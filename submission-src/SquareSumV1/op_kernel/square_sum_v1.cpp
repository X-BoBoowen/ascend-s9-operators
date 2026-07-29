#include "kernel_operator.h"

constexpr uint32_t TILE_OUTPUTS = 1024;
constexpr uint32_t MAX_RANK = 5;
constexpr uint32_t NORMAL_CHUNK = 8192;
constexpr uint32_t LONG_CHUNK = 16384;

// Reductions of at least this many elements use the pipelined
// traversal: double-buffered staging plus a vector accumulator, so
// the scalar unit is read once instead of once per chunk. Below it,
// a reduction fits in a single chunk and the pipelining has nothing
// to overlap, while the extra Add and the accumulator setup/fold
// are pure additions -- those go through the direct traversal. The
// grouped path (fastPath 3) calls the reduction once per group and
// is what makes the short case worth a separate path at all.
// Retune against measured fastPath-3 numbers; a value of 0 forces
// the pipelined path everywhere, one above CHUNK disables it.
constexpr uint64_t PIPELINED_REDUCE_MIN_ELEMENTS = 4096;

union FloatBits {
    float value;
    uint32_t bits;
};

__aicore__ inline float Bfloat16ToFloat(
    const bfloat16_t value)
{
    FloatBits converted;
    converted.bits =
        static_cast<uint32_t>(
            *reinterpret_cast<const uint16_t*>(&value))
        << 16U;
    return converted.value;
}

__aicore__ inline bfloat16_t FloatToBfloat16(
    const float value)
{
    FloatBits converted;
    converted.value = value;
    const uint32_t roundingBias =
        0x7FFFU + ((converted.bits >> 16U) & 1U);
    const uint16_t resultBits = static_cast<uint16_t>(
        (converted.bits + roundingBias) >> 16U);
    bfloat16_t result;
    *reinterpret_cast<uint16_t*>(&result) = resultBits;
    return result;
}

template <typename T>
__aicore__ inline float SquareInInputType(const T value)
{
    const float valueFloat = static_cast<float>(value);
    const T squared =
        static_cast<T>(valueFloat * valueFloat);
    return static_cast<float>(squared);
}

template <>
__aicore__ inline float SquareInInputType<float>(
    const float value)
{
    return value * value;
}

template <>
__aicore__ inline float SquareInInputType<bfloat16_t>(
    const bfloat16_t value)
{
    const float valueFloat = Bfloat16ToFloat(value);
    return Bfloat16ToFloat(
        FloatToBfloat16(valueFloat * valueFloat));
}

template <typename T>
__aicore__ inline T OutputFromFloat(const float value)
{
    return static_cast<T>(value);
}

template <>
__aicore__ inline bfloat16_t OutputFromFloat<bfloat16_t>(
    const float value)
{
    return FloatToBfloat16(value);
}

template <typename T, uint32_t CHUNK>
class KernelSquareSumV1 {
public:
    // inputBuffer_ is split into two staging halves for the
    // contiguous reduction, so a single DMA covers HALF_CHUNK.
    static constexpr uint32_t HALF_CHUNK = CHUNK / 2U;

    __aicore__ inline KernelSquareSumV1() {}

    __aicore__ inline void Init(
        GM_ADDR input,
        GM_ADDR output,
        GM_ADDR workspace,
        const SquareSumV1TilingData& tiling)
    {
        inputElements_ = tiling.inputElements;
        outputElements_ = tiling.outputElements;
        reduceElements_ = tiling.reduceElements;
        innerElements_ = tiling.innerElements;
        trailingReduceElements_ =
            tiling.trailingReduceElements;
        outputRank_ = tiling.outputRank;
        reduceRank_ = tiling.reduceRank;
        fastPath_ = tiling.fastPath;
        reduceMode_ = tiling.reduceMode;
        for (uint32_t axis = 0; axis < MAX_RANK; ++axis) {
            outputDims_[axis] = tiling.outputDims[axis];
            outputInputStrides_[axis] =
                tiling.outputInputStrides[axis];
            reduceDims_[axis] = tiling.reduceDims[axis];
            reduceInputStrides_[axis] =
                tiling.reduceInputStrides[axis];
        }

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        outputs_ = tiling.baseOutputsPerBlock +
            (blockIdx < tiling.extraBlocks ? 1U : 0U);
        firstOutput_ =
            blockIdx * tiling.baseOutputsPerBlock +
            (blockIdx < tiling.extraBlocks
                ? blockIdx
                : tiling.extraBlocks);

        inputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(input),
            inputElements_);
        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(output),
            outputElements_);
        if (reduceMode_ == 2U) {
            partialStride_ =
                (outputElements_ + 7U) / 8U * 8U;
            workspaceGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(workspace),
                partialStride_ * AscendC::GetBlockNum());
        }
        pipe_.InitBuffer(
            outputBuffer_,
            TILE_OUTPUTS * sizeof(T));
        pipe_.InitBuffer(
            inputBuffer_,
            CHUNK * sizeof(T));
        pipe_.InitBuffer(
            floatBuffer_,
            std::is_same<T, float>::value &&
                    CHUNK == LONG_CHUNK
                ? 32U
                : CHUNK * sizeof(float));
        pipe_.InitBuffer(
            reduceWorkBuffer_,
            CHUNK * sizeof(float));
        pipe_.InitBuffer(sumBuffer_, 32);
        mte2ToVEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE2_V));
        // FetchEventID returns the same id for a given hard event,
        // so the ping-pong pair must come from AllocEventID to be
        // distinct from the ids above.
        mte2ToVEventAlt_ = static_cast<event_t>(
            pipe_.AllocEventID<AscendC::HardEvent::MTE2_V>());
        vToMte2EventAlt_ = static_cast<event_t>(
            pipe_.AllocEventID<AscendC::HardEvent::V_MTE2>());
        vToSEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::V_S));
        vToMte2Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::V_MTE2));
        vToMte3Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::V_MTE3));
        mte3ToVEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE3_V));
        sToMte3Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::S_MTE3));
        mte3ToSEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE3_S));
    }

    __aicore__ inline void Process()
    {
        if (reduceMode_ == 2U) {
            ProcessParallelReduction();
            return;
        }
        if constexpr (std::is_same<T, float>::value) {
            if (reduceMode_ == 1U) {
                ProcessAtomicReduction();
                return;
            }
        }
        if (fastPath_ == 1) {
            const uint32_t elementsPerBlock =
                32U / sizeof(T);
            const uint64_t paddedReduce =
                (reduceElements_ + elementsPerBlock - 1U) /
                elementsPerBlock * elementsPerBlock;
            if (reduceElements_ <= 64U &&
                paddedReduce <= CHUNK) {
                ProcessLastAxisRows();
                return;
            }
        }
        if (fastPath_ == 2) {
            ProcessMiddleContiguous();
            return;
        }
        if (fastPath_ == 3 &&
            CanUseGroupedSuffix()) {
            ProcessGroupedSuffix();
            return;
        }
        if (fastPath_ == 4 &&
            CanUseStridedInnerBulk()) {
            ProcessStridedInnerBulk();
            return;
        }
        if (fastPath_ == 4) {
            ProcessMiddleGroup();
            return;
        }
        AscendC::LocalTensor<T> outputLocal =
            outputBuffer_.Get<T>();
        for (uint64_t tileOffset = 0;
             tileOffset < outputs_;
             tileOffset += TILE_OUTPUTS) {
            const uint32_t current = static_cast<uint32_t>(
                outputs_ - tileOffset < TILE_OUTPUTS
                    ? outputs_ - tileOffset
                    : TILE_OUTPUTS);
            const uint64_t outputStart =
                firstOutput_ + tileOffset;
            for (uint32_t element = 0;
                 element < current;
                 ++element) {
                const uint64_t baseInputOffset =
                    BaseInputOffset(outputStart + element);
                float sum = 0.0f;
                if (fastPath_ == 1) {
                    sum = ReduceContiguous(
                        baseInputOffset,
                        reduceElements_);
                } else if (fastPath_ == 3) {
                    const uint64_t groups =
                        reduceElements_ /
                        trailingReduceElements_;
                    for (uint64_t group = 0;
                         group < groups;
                         ++group) {
                        const uint64_t groupBase =
                            baseInputOffset +
                            ReduceInputOffset(
                                group *
                                trailingReduceElements_);
                        sum += ReduceContiguous(
                            groupBase,
                            trailingReduceElements_);
                    }
                } else {
                    for (uint64_t reduceIndex = 0;
                         reduceIndex < reduceElements_;
                         ++reduceIndex) {
                        const uint64_t inputOffset =
                            baseInputOffset +
                            ReduceInputOffset(reduceIndex);
                        sum += SquareInInputType<T>(
                            inputGm_.GetValue(inputOffset));
                    }
                }
                outputLocal.SetValue(
                    element,
                    OutputFromFloat<T>(sum));
            }

            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(
                sToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(
                sToMte3Event_);
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = current * sizeof(T);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(
                outputGm_[outputStart],
                outputLocal,
                copyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(
                mte3ToSEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(
                mte3ToSEvent_);
        }
    }

private:
    __aicore__ inline void ProcessParallelReduction()
    {
        const uint64_t blockIdx = AscendC::GetBlockIdx();
        const uint64_t blockNum = AscendC::GetBlockNum();
        const uint64_t baseReduce = reduceElements_ / blockNum;
        const uint64_t extraReduce = reduceElements_ % blockNum;
        const uint64_t reduceCount =
            baseReduce + (blockIdx < extraReduce ? 1U : 0U);
        const uint64_t reduceStart =
            blockIdx * baseReduce +
            (blockIdx < extraReduce ? blockIdx : extraReduce);

        if (fastPath_ == 1U) {
            ProcessParallelLast(reduceStart, reduceCount);
        } else {
            ProcessParallelMiddle(reduceStart, reduceCount);
        }
        AscendC::SyncAll<true>();
        if (blockIdx == 0U) {
            FinalizeParallelReduction();
        }
    }

    __aicore__ inline void ProcessParallelLast(
        const uint64_t reduceStart,
        const uint64_t reduceCount)
    {
        AscendC::LocalTensor<float> partialLocal =
            floatBuffer_.Get<float>();
        const uint64_t workspaceOffset =
            AscendC::GetBlockIdx() * partialStride_;

        for (uint64_t outputIndex = 0;
             outputIndex < outputElements_;
             ++outputIndex) {
            const uint64_t inputStart =
                BaseInputOffset(outputIndex) + reduceStart;
            partialLocal.SetValue(
                0,
                ReduceContiguous(inputStart, reduceCount));
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(
                sToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(
                sToMte3Event_);
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = sizeof(float);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(
                workspaceGm_[workspaceOffset + outputIndex],
                partialLocal,
                copyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(
                mte3ToSEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(
                mte3ToSEvent_);
        }
    }

    __aicore__ inline void ProcessParallelMiddle(
        const uint64_t reduceStart,
        const uint64_t reduceCount)
    {
        AscendC::LocalTensor<T> inputLocal =
            inputBuffer_.Get<T>();
        AscendC::LocalTensor<float> valueLocal =
            floatBuffer_.Get<float>();
        AscendC::LocalTensor<float> accumulateLocal =
            reduceWorkBuffer_.Get<float>();
        const uint64_t workspaceOffset =
            AscendC::GetBlockIdx() * partialStride_;

        uint64_t outputIndex = 0;
        while (outputIndex < outputElements_) {
            const uint64_t outerIndex =
                outputIndex / innerElements_;
            const uint64_t innerIndex =
                outputIndex % innerElements_;
            uint64_t current64 =
                outputElements_ - outputIndex;
            const uint64_t untilRowEnd =
                innerElements_ - innerIndex;
            if (current64 > untilRowEnd) {
                current64 = untilRowEnd;
            }
            if (current64 > TILE_OUTPUTS) {
                current64 = TILE_OUTPUTS;
            }
            const uint32_t current =
                static_cast<uint32_t>(current64);
            const uint32_t elementsPerBlock =
                32U / sizeof(T);
            const uint32_t paddedInner =
                (current + elementsPerBlock - 1U) /
                elementsPerBlock * elementsPerBlock;
            const uint32_t floatPadded =
                (current + 7U) / 8U * 8U;
            uint32_t reduceRowsPerTile =
                CHUNK / paddedInner;
            if (reduceRowsPerTile == 0U) {
                reduceRowsPerTile = 1U;
            }
            if (reduceRowsPerTile > 4095U) {
                reduceRowsPerTile = 4095U;
            }

            AscendC::Duplicate(
                accumulateLocal,
                0.0f,
                floatPadded);
            uint64_t reduceOffset = 0;
            while (reduceOffset < reduceCount) {
                const uint32_t remainingRows =
                    static_cast<uint32_t>(
                        reduceCount - reduceOffset <
                                reduceRowsPerTile
                            ? reduceCount - reduceOffset
                            : reduceRowsPerTile);
                const uint32_t currentReduceRows =
                    HighestPowerOfTwo(remainingRows);
                const uint32_t inputCount =
                    currentReduceRows * paddedInner;
                const uint64_t inputStart =
                    (outerIndex * reduceElements_ +
                     reduceStart + reduceOffset) *
                        innerElements_ +
                    innerIndex;

                AscendC::DataCopyExtParams copyParams;
                copyParams.blockCount =
                    static_cast<uint16_t>(currentReduceRows);
                copyParams.blockLen = current * sizeof(T);
                copyParams.srcStride =
                    static_cast<uint32_t>(
                        (innerElements_ - current) *
                        sizeof(T));
                copyParams.dstStride = 0;
                AscendC::DataCopyPadExtParams<T> padParams;
                padParams.isPad = paddedInner != current;
                padParams.leftPadding = 0;
                padParams.rightPadding =
                    static_cast<uint8_t>(
                        paddedInner - current);
                const T zero = {};
                padParams.paddingValue = zero;
                AscendC::DataCopyPad(
                    inputLocal,
                    inputGm_[inputStart],
                    copyParams,
                    padParams);
                AscendC::SetFlag<
                    AscendC::HardEvent::MTE2_V>(
                    mte2ToVEvent_);
                AscendC::WaitFlag<
                    AscendC::HardEvent::MTE2_V>(
                    mte2ToVEvent_);

                if constexpr (
                    std::is_same<T, float>::value) {
                    AscendC::Mul(
                        inputLocal,
                        inputLocal,
                        inputLocal,
                        inputCount);
                    ReduceRowsInto(
                        inputLocal,
                        accumulateLocal,
                        currentReduceRows,
                        paddedInner,
                        floatPadded);
                } else if constexpr (
                    std::is_same<T, half>::value) {
                    AscendC::Mul(
                        inputLocal,
                        inputLocal,
                        inputLocal,
                        inputCount);
                    AscendC::Cast(
                        valueLocal,
                        inputLocal,
                        AscendC::RoundMode::CAST_NONE,
                        inputCount);
                    ReduceRowsInto(
                        valueLocal,
                        accumulateLocal,
                        currentReduceRows,
                        paddedInner,
                        floatPadded);
                } else {
                    AscendC::Cast(
                        valueLocal,
                        inputLocal,
                        AscendC::RoundMode::CAST_NONE,
                        inputCount);
                    AscendC::Mul(
                        valueLocal,
                        valueLocal,
                        valueLocal,
                        inputCount);
                    AscendC::Cast(
                        inputLocal,
                        valueLocal,
                        AscendC::RoundMode::CAST_RINT,
                        inputCount);
                    AscendC::Cast(
                        valueLocal,
                        inputLocal,
                        AscendC::RoundMode::CAST_NONE,
                        inputCount);
                    ReduceRowsInto(
                        valueLocal,
                        accumulateLocal,
                        currentReduceRows,
                        paddedInner,
                        floatPadded);
                }
                AscendC::SetFlag<
                    AscendC::HardEvent::V_MTE2>(
                    vToMte2Event_);
                AscendC::WaitFlag<
                    AscendC::HardEvent::V_MTE2>(
                    vToMte2Event_);
                reduceOffset += currentReduceRows;
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::DataCopyExtParams outputCopy;
            outputCopy.blockCount = 1;
            outputCopy.blockLen = current * sizeof(float);
            outputCopy.srcStride = 0;
            outputCopy.dstStride = 0;
            AscendC::DataCopyPad(
                workspaceGm_[workspaceOffset + outputIndex],
                accumulateLocal,
                outputCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                mte3ToVEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                mte3ToVEvent_);
            outputIndex += current;
        }
    }

    __aicore__ inline void FinalizeParallelReduction()
    {
        AscendC::LocalTensor<T> outputLocal =
            outputBuffer_.Get<T>();
        AscendC::LocalTensor<float> valueLocal =
            floatBuffer_.Get<float>();
        AscendC::LocalTensor<float> accumulateLocal =
            reduceWorkBuffer_.Get<float>();
        const uint64_t blockNum = AscendC::GetBlockNum();

        for (uint64_t outputStart = 0;
             outputStart < outputElements_;
             outputStart += TILE_OUTPUTS) {
            const uint32_t current = static_cast<uint32_t>(
                outputElements_ - outputStart < TILE_OUTPUTS
                    ? outputElements_ - outputStart
                    : TILE_OUTPUTS);
            const uint32_t floatPadded =
                (current + 7U) / 8U * 8U;
            AscendC::Duplicate(
                accumulateLocal,
                0.0f,
                floatPadded);

            for (uint64_t block = 0;
                 block < blockNum;
                 ++block) {
                AscendC::DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = current * sizeof(float);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                AscendC::DataCopyPadExtParams<float> padParams;
                padParams.isPad = floatPadded != current;
                padParams.leftPadding = 0;
                padParams.rightPadding =
                    static_cast<uint8_t>(
                        floatPadded - current);
                padParams.paddingValue = 0.0f;
                AscendC::DataCopyPad(
                    valueLocal,
                    workspaceGm_[
                        block * partialStride_ + outputStart],
                    copyParams,
                    padParams);
                AscendC::SetFlag<
                    AscendC::HardEvent::MTE2_V>(
                    mte2ToVEvent_);
                AscendC::WaitFlag<
                    AscendC::HardEvent::MTE2_V>(
                    mte2ToVEvent_);
                AscendC::Add(
                    accumulateLocal,
                    accumulateLocal,
                    valueLocal,
                    floatPadded);
                AscendC::SetFlag<
                    AscendC::HardEvent::V_MTE2>(
                    vToMte2Event_);
                AscendC::WaitFlag<
                    AscendC::HardEvent::V_MTE2>(
                    vToMte2Event_);
            }

            if constexpr (std::is_same<T, float>::value) {
                AscendC::Adds(
                    outputLocal,
                    accumulateLocal,
                    0.0f,
                    floatPadded);
            } else if constexpr (
                std::is_same<T, half>::value) {
                AscendC::Cast(
                    outputLocal,
                    accumulateLocal,
                    AscendC::RoundMode::CAST_NONE,
                    floatPadded);
            } else {
                AscendC::Cast(
                    outputLocal,
                    accumulateLocal,
                    AscendC::RoundMode::CAST_RINT,
                    floatPadded);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::DataCopyExtParams outputCopy;
            outputCopy.blockCount = 1;
            outputCopy.blockLen = current * sizeof(T);
            outputCopy.srcStride = 0;
            outputCopy.dstStride = 0;
            AscendC::DataCopyPad(
                outputGm_[outputStart],
                outputLocal,
                outputCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                mte3ToVEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                mte3ToVEvent_);
        }
    }

    __aicore__ inline void ProcessAtomicReduction()
    {
        AscendC::LocalTensor<T> outputLocal =
            outputBuffer_.Get<T>();
        const uint32_t alignedOutputs =
            static_cast<uint32_t>(
                (outputElements_ + 7U) / 8U * 8U);
        AscendC::Duplicate(
            outputLocal,
            static_cast<T>(0),
            alignedOutputs);

        if (AscendC::GetBlockIdx() == 0U) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::DataCopy(
                outputGm_,
                outputLocal,
                alignedOutputs);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(
                mte3ToSEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(
                mte3ToSEvent_);
        }
        AscendC::SyncAll<true>();

        const uint64_t blockIdx = AscendC::GetBlockIdx();
        const uint64_t blockNum = AscendC::GetBlockNum();
        const uint64_t baseReduce = reduceElements_ / blockNum;
        const uint64_t extraReduce = reduceElements_ % blockNum;
        const uint64_t reduceCount =
            baseReduce + (blockIdx < extraReduce ? 1U : 0U);
        const uint64_t reduceStart =
            blockIdx * baseReduce +
            (blockIdx < extraReduce ? blockIdx : extraReduce);

        for (uint64_t outputIndex = 0;
             outputIndex < outputElements_;
             ++outputIndex) {
            const uint64_t inputStart =
                BaseInputOffset(outputIndex) + reduceStart;
            outputLocal.SetValue(
                outputIndex,
                OutputFromFloat<T>(
                    ReduceContiguous(
                        inputStart,
                        reduceCount)));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(
            sToMte3Event_);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(
            sToMte3Event_);
        AscendC::SetAtomicAdd<T>();
        AscendC::DataCopy(
            outputGm_,
            outputLocal,
            alignedOutputs);
        AscendC::SetAtomicNone();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(
            mte3ToSEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(
            mte3ToSEvent_);
    }

    __aicore__ inline bool CanUseStridedInnerBulk() const
    {
        if (reduceRank_ == 0U ||
            innerElements_ == 0U) {
            return false;
        }
        const uint32_t lastReduceAxis =
            reduceRank_ - 1U;
        if (reduceInputStrides_[lastReduceAxis] !=
            innerElements_) {
            return false;
        }
        return innerElements_ * sizeof(T) <=
            0xFFFFFFFFULL;
    }

    __aicore__ inline void ProcessStridedInnerBulk()
    {
        AscendC::LocalTensor<T> inputLocal =
            inputBuffer_.Get<T>();
        AscendC::LocalTensor<T> outputLocal =
            outputBuffer_.Get<T>();
        AscendC::LocalTensor<float> valueLocal =
            floatBuffer_.Get<float>();
        AscendC::LocalTensor<float> accumulateLocal =
            reduceWorkBuffer_.Get<float>();

        const uint32_t lastReduceAxis =
            reduceRank_ - 1U;
        const uint64_t lastReduceDim =
            reduceDims_[lastReduceAxis];
        const uint64_t outerReduceGroups =
            reduceElements_ / lastReduceDim;

        uint64_t processed = 0;
        while (processed < outputs_) {
            const uint64_t outputIndex =
                firstOutput_ + processed;
            const uint64_t innerIndex =
                outputIndex % innerElements_;
            uint64_t current64 = outputs_ - processed;
            const uint64_t untilRowEnd =
                innerElements_ - innerIndex;
            if (current64 > untilRowEnd) {
                current64 = untilRowEnd;
            }
            if (current64 > TILE_OUTPUTS) {
                current64 = TILE_OUTPUTS;
            }
            const uint32_t current =
                static_cast<uint32_t>(current64);
            const uint32_t elementsPerBlock =
                32U / sizeof(T);
            const uint32_t paddedInner =
                (current + elementsPerBlock - 1U) /
                elementsPerBlock * elementsPerBlock;
            const uint32_t floatPadded =
                (current + 7U) / 8U * 8U;
            uint32_t reduceRowsPerTile =
            CHUNK / paddedInner;
            if (reduceRowsPerTile == 0U) {
                reduceRowsPerTile = 1U;
            }
            if (reduceRowsPerTile > 4095U) {
                reduceRowsPerTile = 4095U;
            }
            AscendC::Duplicate(
                accumulateLocal,
                0.0f,
                floatPadded);
            const uint64_t baseInputOffset =
                BaseInputOffset(outputIndex);
            for (uint64_t group = 0;
                 group < outerReduceGroups;
                 ++group) {
                const uint64_t groupReduceStart =
                    group * lastReduceDim;
                for (uint64_t rowOffset = 0;
                     rowOffset < lastReduceDim;
                     rowOffset += reduceRowsPerTile) {
                    const uint32_t currentReduceRows =
                        static_cast<uint32_t>(
                            lastReduceDim - rowOffset <
                                    reduceRowsPerTile
                                ? lastReduceDim - rowOffset
                                : reduceRowsPerTile);
                    const uint32_t inputCount =
                        currentReduceRows * paddedInner;
                    const uint64_t inputStart =
                        baseInputOffset +
                        ReduceInputOffset(
                            groupReduceStart +
                            rowOffset);

                    AscendC::DataCopyExtParams copyParams;
                    copyParams.blockCount =
                        static_cast<uint16_t>(
                            currentReduceRows);
                    copyParams.blockLen =
                        current * sizeof(T);
                    copyParams.srcStride =
                        static_cast<uint32_t>(
                            (innerElements_ - current) *
                            sizeof(T));
                    copyParams.dstStride = 0;
                    AscendC::DataCopyPadExtParams<T>
                        padParams;
                    padParams.isPad =
                        paddedInner != current;
                    padParams.leftPadding = 0;
                    padParams.rightPadding =
                        static_cast<uint8_t>(
                            paddedInner - current);
                    const T zero = {};
                    padParams.paddingValue = zero;
                    AscendC::DataCopyPad(
                        inputLocal,
                        inputGm_[inputStart],
                        copyParams,
                        padParams);
                    AscendC::SetFlag<
                        AscendC::HardEvent::MTE2_V>(
                        mte2ToVEvent_);
                    AscendC::WaitFlag<
                        AscendC::HardEvent::MTE2_V>(
                        mte2ToVEvent_);

                    if constexpr (
                        std::is_same<T, float>::value) {
                        AscendC::Mul(
                            inputLocal,
                            inputLocal,
                            inputLocal,
                            inputCount);
                        for (uint32_t row = 0;
                             row < currentReduceRows;
                             ++row) {
                            AscendC::Add(
                                accumulateLocal,
                                accumulateLocal,
                                inputLocal[
                                    row * paddedInner],
                                floatPadded);
                        }
                    } else if constexpr (
                        std::is_same<T, half>::value) {
                        AscendC::Mul(
                            inputLocal,
                            inputLocal,
                            inputLocal,
                            inputCount);
                        AscendC::Cast(
                            valueLocal,
                            inputLocal,
                            AscendC::RoundMode::CAST_NONE,
                            inputCount);
                        for (uint32_t row = 0;
                             row < currentReduceRows;
                             ++row) {
                            AscendC::Add(
                                accumulateLocal,
                                accumulateLocal,
                                valueLocal[
                                    row * paddedInner],
                                floatPadded);
                        }
                    } else {
                        AscendC::Cast(
                            valueLocal,
                            inputLocal,
                            AscendC::RoundMode::CAST_NONE,
                            inputCount);
                        AscendC::Mul(
                            valueLocal,
                            valueLocal,
                            valueLocal,
                            inputCount);
                        AscendC::Cast(
                            inputLocal,
                            valueLocal,
                            AscendC::RoundMode::CAST_RINT,
                            inputCount);
                        AscendC::Cast(
                            valueLocal,
                            inputLocal,
                            AscendC::RoundMode::CAST_NONE,
                            inputCount);
                        for (uint32_t row = 0;
                             row < currentReduceRows;
                             ++row) {
                            AscendC::Add(
                                accumulateLocal,
                                accumulateLocal,
                                valueLocal[
                                    row * paddedInner],
                                floatPadded);
                        }
                    }
                    AscendC::SetFlag<
                        AscendC::HardEvent::V_MTE2>(
                        vToMte2Event_);
                    AscendC::WaitFlag<
                        AscendC::HardEvent::V_MTE2>(
                        vToMte2Event_);
                }
            }

            if constexpr (std::is_same<T, float>::value) {
                AscendC::Adds(
                    outputLocal,
                    accumulateLocal,
                    0.0f,
                    floatPadded);
            } else if constexpr (
                std::is_same<T, half>::value) {
                AscendC::Cast(
                    outputLocal,
                    accumulateLocal,
                    AscendC::RoundMode::CAST_NONE,
                    floatPadded);
            } else {
                AscendC::Cast(
                    outputLocal,
                    accumulateLocal,
                    AscendC::RoundMode::CAST_RINT,
                    floatPadded);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::DataCopyExtParams outputCopy;
            outputCopy.blockCount = 1;
            outputCopy.blockLen = current * sizeof(T);
            outputCopy.srcStride = 0;
            outputCopy.dstStride = 0;
            AscendC::DataCopyPad(
                outputGm_[outputIndex],
                outputLocal,
                outputCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                mte3ToVEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                mte3ToVEvent_);
            processed += current;
        }
    }

    __aicore__ inline bool CanUseGroupedSuffix() const
    {
        uint64_t expectedStride = 1;
        uint32_t firstTrailingAxis = reduceRank_;
        for (int32_t axis =
                 static_cast<int32_t>(reduceRank_) - 1;
             axis >= 0;
             --axis) {
            if (reduceInputStrides_[axis] !=
                expectedStride) {
                break;
            }
            expectedStride *= reduceDims_[axis];
            firstTrailingAxis =
                static_cast<uint32_t>(axis);
        }
        if (firstTrailingAxis == 0U ||
            firstTrailingAxis >= reduceRank_ ||
            expectedStride != trailingReduceElements_) {
            return false;
        }

        const uint32_t batchAxis =
            firstTrailingAxis - 1U;
        const uint64_t elementsPerBlock =
            32U / sizeof(T);
        const uint64_t paddedTail =
            (trailingReduceElements_ +
             elementsPerBlock - 1U) /
            elementsPerBlock * elementsPerBlock;
        if (reduceDims_[batchAxis] == 0U ||
            paddedTail == 0U ||
            paddedTail > CHUNK ||
            reduceInputStrides_[batchAxis] <
                trailingReduceElements_) {
            return false;
        }
        const uint64_t sourceGapBytes =
            (reduceInputStrides_[batchAxis] -
             trailingReduceElements_) *
            sizeof(T);
        return sourceGapBytes <= 0xFFFFFFFFULL;
    }

    __aicore__ inline void ProcessGroupedSuffix()
    {
        AscendC::LocalTensor<T> inputLocal =
            inputBuffer_.Get<T>();
        AscendC::LocalTensor<T> outputLocal =
            outputBuffer_.Get<T>();
        AscendC::LocalTensor<float> floatLocal =
            floatBuffer_.Get<float>();
        AscendC::LocalTensor<float> workLocal =
            reduceWorkBuffer_.Get<float>();
        AscendC::LocalTensor<float> sumLocal =
            sumBuffer_.Get<float>();

        uint64_t expectedStride = 1;
        uint32_t firstTrailingAxis = reduceRank_;
        for (int32_t axis =
                 static_cast<int32_t>(reduceRank_) - 1;
             axis >= 0;
             --axis) {
            if (reduceInputStrides_[axis] !=
                expectedStride) {
                break;
            }
            expectedStride *= reduceDims_[axis];
            firstTrailingAxis =
                static_cast<uint32_t>(axis);
        }
        const uint32_t batchAxis =
            firstTrailingAxis - 1U;
        const uint64_t batchDim =
            reduceDims_[batchAxis];
        const uint64_t outerGroups =
            reduceElements_ /
            (batchDim * trailingReduceElements_);
        const uint32_t elementsPerBlock =
            32U / sizeof(T);
        const uint32_t paddedTail =
            static_cast<uint32_t>(
                (trailingReduceElements_ +
                 elementsPerBlock - 1U) /
                elementsPerBlock * elementsPerBlock);
        uint32_t batchRowsPerTile =
            CHUNK / paddedTail;
        if (batchRowsPerTile == 0U) {
            batchRowsPerTile = 1U;
        }
        if (batchRowsPerTile > 4095U) {
            batchRowsPerTile = 4095U;
        }
        const uint32_t sourceGapBytes =
            static_cast<uint32_t>(
                (reduceInputStrides_[batchAxis] -
                 trailingReduceElements_) *
                sizeof(T));

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockLen =
            static_cast<uint32_t>(
                trailingReduceElements_ * sizeof(T));
        copyParams.srcStride = sourceGapBytes;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> padParams;
        padParams.isPad =
            paddedTail != trailingReduceElements_;
        padParams.leftPadding = 0;
        padParams.rightPadding =
            static_cast<uint8_t>(
                paddedTail - trailingReduceElements_);
        const T zero = {};
        padParams.paddingValue = zero;

        for (uint64_t tileOffset = 0;
             tileOffset < outputs_;
             tileOffset += TILE_OUTPUTS) {
            const uint32_t current =
                static_cast<uint32_t>(
                    outputs_ - tileOffset <
                            TILE_OUTPUTS
                        ? outputs_ - tileOffset
                        : TILE_OUTPUTS);
            const uint64_t outputStart =
                firstOutput_ + tileOffset;
            for (uint32_t element = 0;
                 element < current;
                 ++element) {
                const uint64_t baseInputOffset =
                    BaseInputOffset(
                        outputStart + element);
                float total = 0.0f;
                for (uint64_t group = 0;
                     group < outerGroups;
                     ++group) {
                    const uint64_t groupReduceStart =
                        group * batchDim *
                        trailingReduceElements_;
                    for (uint64_t batchOffset = 0;
                         batchOffset < batchDim;
                         batchOffset +=
                             batchRowsPerTile) {
                        const uint32_t currentRows =
                            static_cast<uint32_t>(
                                batchDim - batchOffset <
                                        batchRowsPerTile
                                    ? batchDim -
                                        batchOffset
                                    : batchRowsPerTile);
                        const uint32_t inputCount =
                            currentRows * paddedTail;
                        const uint64_t inputStart =
                            baseInputOffset +
                            ReduceInputOffset(
                                groupReduceStart +
                                batchOffset *
                                    trailingReduceElements_);
                        copyParams.blockCount =
                            static_cast<uint16_t>(
                                currentRows);
                        AscendC::DataCopyPad(
                            inputLocal,
                            inputGm_[inputStart],
                            copyParams,
                            padParams);
                        AscendC::SetFlag<
                            AscendC::HardEvent::MTE2_V>(
                            mte2ToVEvent_);
                        AscendC::WaitFlag<
                            AscendC::HardEvent::MTE2_V>(
                            mte2ToVEvent_);

                        if constexpr (
                            std::is_same<
                                T,
                                float>::value) {
                            AscendC::Mul(
                                inputLocal,
                                inputLocal,
                                inputLocal,
                                inputCount);
                            AscendC::ReduceSum(
                                sumLocal,
                                inputLocal,
                                workLocal,
                                inputCount);
                        } else if constexpr (
                            std::is_same<
                                T,
                                half>::value) {
                            AscendC::Mul(
                                inputLocal,
                                inputLocal,
                                inputLocal,
                                inputCount);
                            AscendC::Cast(
                                floatLocal,
                                inputLocal,
                                AscendC::RoundMode::
                                    CAST_NONE,
                                inputCount);
                            AscendC::ReduceSum(
                                sumLocal,
                                floatLocal,
                                workLocal,
                                inputCount);
                        } else {
                            AscendC::Cast(
                                floatLocal,
                                inputLocal,
                                AscendC::RoundMode::
                                    CAST_NONE,
                                inputCount);
                            AscendC::Mul(
                                floatLocal,
                                floatLocal,
                                floatLocal,
                                inputCount);
                            AscendC::Cast(
                                inputLocal,
                                floatLocal,
                                AscendC::RoundMode::CAST_RINT,
                                inputCount);
                            AscendC::Cast(
                                floatLocal,
                                inputLocal,
                                AscendC::RoundMode::CAST_NONE,
                                inputCount);
                            AscendC::ReduceSum(
                                sumLocal,
                                floatLocal,
                                workLocal,
                                inputCount);
                        }
                        AscendC::SetFlag<
                            AscendC::HardEvent::V_S>(
                            vToSEvent_);
                        AscendC::WaitFlag<
                            AscendC::HardEvent::V_S>(
                            vToSEvent_);
                        total += sumLocal.GetValue(0);
                        AscendC::SetFlag<
                            AscendC::HardEvent::V_MTE2>(
                            vToMte2Event_);
                        AscendC::WaitFlag<
                            AscendC::HardEvent::V_MTE2>(
                            vToMte2Event_);
                    }
                }
                outputLocal.SetValue(
                    element,
                    OutputFromFloat<T>(total));
            }

            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(
                sToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(
                sToMte3Event_);
            AscendC::DataCopyExtParams outputCopy;
            outputCopy.blockCount = 1;
            outputCopy.blockLen = current * sizeof(T);
            outputCopy.srcStride = 0;
            outputCopy.dstStride = 0;
            AscendC::DataCopyPad(
                outputGm_[outputStart],
                outputLocal,
                outputCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(
                mte3ToSEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(
                mte3ToSEvent_);
        }
    }

    __aicore__ inline void ProcessLastAxisRows()
    {
        AscendC::LocalTensor<T> inputLocal =
            inputBuffer_.Get<T>();
        AscendC::LocalTensor<T> outputLocal =
            outputBuffer_.Get<T>();
        AscendC::LocalTensor<float> floatLocal =
            floatBuffer_.Get<float>();
        AscendC::LocalTensor<float> sumLocal =
            reduceWorkBuffer_.Get<float>();

        const uint32_t elementsPerBlock = 32U / sizeof(T);
        const uint32_t paddedReduce = static_cast<uint32_t>(
            (reduceElements_ + elementsPerBlock - 1U) /
            elementsPerBlock * elementsPerBlock);
        uint32_t tileRows =
            paddedReduce == 0U
                ? 1U
                : CHUNK / paddedReduce;
        if (tileRows == 0U) {
            tileRows = 1U;
        }
        // WholeReduceSum repeatTimes is encoded in the vector instruction's
        // 8-bit repeat field. Larger values silently wrap on 910B4.
        if (tileRows > 255U) {
            tileRows = 255U;
        }

        for (uint64_t rowOffset = 0;
             rowOffset < outputs_;
             rowOffset += tileRows) {
            const uint32_t currentRows = static_cast<uint32_t>(
                outputs_ - rowOffset < tileRows
                    ? outputs_ - rowOffset
                    : tileRows);
            const uint32_t inputCount =
                currentRows * paddedReduce;
            const uint64_t firstRow =
                firstOutput_ + rowOffset;

            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount =
                static_cast<uint16_t>(currentRows);
            copyParams.blockLen =
                static_cast<uint32_t>(
                    reduceElements_ * sizeof(T));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<T> padParams;
            padParams.isPad =
                paddedReduce != reduceElements_;
            padParams.leftPadding = 0;
            padParams.rightPadding =
                static_cast<uint8_t>(
                    paddedReduce - reduceElements_);
            const T zero = {};
            padParams.paddingValue = zero;
            AscendC::DataCopyPad(
                inputLocal,
                inputGm_[firstRow * reduceElements_],
                copyParams,
                padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
                mte2ToVEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                mte2ToVEvent_);

            if constexpr (std::is_same<T, float>::value) {
                AscendC::Mul(
                    inputLocal,
                    inputLocal,
                    inputLocal,
                    inputCount);
                AscendC::WholeReduceSum<float>(
                    outputLocal,
                    inputLocal,
                    static_cast<int32_t>(reduceElements_),
                    static_cast<int32_t>(currentRows),
                    1,
                    1,
                    static_cast<int32_t>(paddedReduce / 8U));
            } else if constexpr (
                std::is_same<T, half>::value) {
                AscendC::Mul(
                    inputLocal,
                    inputLocal,
                    inputLocal,
                    inputCount);
                AscendC::Cast(
                    floatLocal,
                    inputLocal,
                    AscendC::RoundMode::CAST_NONE,
                    inputCount);
                AscendC::WholeReduceSum<float>(
                    sumLocal,
                    floatLocal,
                    static_cast<int32_t>(reduceElements_),
                    static_cast<int32_t>(currentRows),
                    1,
                    1,
                    static_cast<int32_t>(paddedReduce / 8U));
                AscendC::Cast(
                    outputLocal,
                    sumLocal,
                    AscendC::RoundMode::CAST_NONE,
                    static_cast<int32_t>(currentRows));
            } else {
                AscendC::Cast(
                    floatLocal,
                    inputLocal,
                    AscendC::RoundMode::CAST_NONE,
                    inputCount);
                AscendC::Mul(
                    floatLocal,
                    floatLocal,
                    floatLocal,
                    inputCount);
                AscendC::Cast(
                    inputLocal,
                    floatLocal,
                    AscendC::RoundMode::CAST_RINT,
                    inputCount);
                AscendC::Cast(
                    floatLocal,
                    inputLocal,
                    AscendC::RoundMode::CAST_NONE,
                    inputCount);
                AscendC::WholeReduceSum<float>(
                    sumLocal,
                    floatLocal,
                    static_cast<int32_t>(reduceElements_),
                    static_cast<int32_t>(currentRows),
                    1,
                    1,
                    static_cast<int32_t>(paddedReduce / 8U));
                AscendC::Cast(
                    outputLocal,
                    sumLocal,
                    AscendC::RoundMode::CAST_RINT,
                    static_cast<int32_t>(currentRows));
            }

            if (rowOffset + currentRows < outputs_) {
                // inputLocal is reused by the next MTE2 tile. Do not let that
                // transfer overwrite data while this tile's vector work lives.
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                    vToMte2Event_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                    vToMte2Event_);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::DataCopyExtParams outputCopy;
            outputCopy.blockCount = 1;
            outputCopy.blockLen =
                currentRows * sizeof(T);
            outputCopy.srcStride = 0;
            outputCopy.dstStride = 0;
            AscendC::DataCopyPad(
                outputGm_[firstRow],
                outputLocal,
                outputCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                mte3ToVEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                mte3ToVEvent_);
        }
    }

    __aicore__ inline uint32_t HighestPowerOfTwo(
        const uint32_t value) const
    {
        uint32_t result = 1U;
        while ((result << 1U) <= value) {
            result <<= 1U;
        }
        return result;
    }

    __aicore__ inline void ReduceRowsInto(
        AscendC::LocalTensor<float> values,
        AscendC::LocalTensor<float> accumulate,
        const uint32_t rows,
        const uint32_t paddedInner,
        const uint32_t floatPadded)
    {
        for (uint32_t activeRows = rows;
             activeRows > 1U;
             activeRows >>= 1U) {
            const uint32_t halfRows = activeRows >> 1U;
            AscendC::Add(
                values,
                values,
                values[halfRows * paddedInner],
                halfRows * paddedInner);
        }
        AscendC::Add(
            accumulate,
            accumulate,
            values,
            floatPadded);
    }

    __aicore__ inline void ProcessMiddleContiguous()
    {
        AscendC::LocalTensor<T> inputLocal =
            inputBuffer_.Get<T>();
        AscendC::LocalTensor<T> outputLocal =
            outputBuffer_.Get<T>();
        AscendC::LocalTensor<float> valueLocal =
            floatBuffer_.Get<float>();
        AscendC::LocalTensor<float> accumulateLocal =
            reduceWorkBuffer_.Get<float>();

        uint64_t processed = 0;
        while (processed < outputs_) {
            const uint64_t outputIndex =
                firstOutput_ + processed;
            const uint64_t outerIndex =
                outputIndex / innerElements_;
            const uint64_t innerIndex =
                outputIndex % innerElements_;
            uint64_t current64 = outputs_ - processed;
            const uint64_t untilRowEnd =
                innerElements_ - innerIndex;
            if (current64 > untilRowEnd) {
                current64 = untilRowEnd;
            }
            if (current64 > TILE_OUTPUTS) {
                current64 = TILE_OUTPUTS;
            }
            const uint32_t current =
                static_cast<uint32_t>(current64);
            const uint32_t elementsPerBlock =
                32U / sizeof(T);
            const uint32_t paddedInner =
                (current + elementsPerBlock - 1U) /
                elementsPerBlock * elementsPerBlock;
            const uint32_t floatPadded =
                (current + 7U) / 8U * 8U;
            uint32_t reduceRowsPerTile =
                CHUNK / paddedInner;
            if (reduceRowsPerTile == 0U) {
                reduceRowsPerTile = 1U;
            }
            if (reduceRowsPerTile > 4095U) {
                reduceRowsPerTile = 4095U;
            }

            AscendC::Duplicate(
                accumulateLocal,
                0.0f,
                floatPadded);
            uint64_t reduceOffset = 0;
            while (reduceOffset < reduceElements_) {
                const uint32_t remainingRows =
                    static_cast<uint32_t>(
                        reduceElements_ - reduceOffset <
                                reduceRowsPerTile
                            ? reduceElements_ - reduceOffset
                            : reduceRowsPerTile);
                const uint32_t currentReduceRows =
                    HighestPowerOfTwo(remainingRows);
                const uint32_t inputCount =
                    currentReduceRows * paddedInner;
                const uint64_t inputStart =
                    (outerIndex * reduceElements_ +
                     reduceOffset) *
                        innerElements_ +
                    innerIndex;

                AscendC::DataCopyExtParams copyParams;
                copyParams.blockCount =
                    static_cast<uint16_t>(currentReduceRows);
                copyParams.blockLen =
                    current * sizeof(T);
                copyParams.srcStride =
                    static_cast<uint32_t>(
                        (innerElements_ - current) *
                        sizeof(T));
                copyParams.dstStride = 0;
                AscendC::DataCopyPadExtParams<T> padParams;
                padParams.isPad = paddedInner != current;
                padParams.leftPadding = 0;
                padParams.rightPadding =
                    static_cast<uint8_t>(
                        paddedInner - current);
                const T zero = {};
                padParams.paddingValue = zero;
                AscendC::DataCopyPad(
                    inputLocal,
                    inputGm_[inputStart],
                    copyParams,
                    padParams);
                AscendC::SetFlag<
                    AscendC::HardEvent::MTE2_V>(
                    mte2ToVEvent_);
                AscendC::WaitFlag<
                    AscendC::HardEvent::MTE2_V>(
                    mte2ToVEvent_);

                if constexpr (
                    std::is_same<T, float>::value) {
                    AscendC::Mul(
                        inputLocal,
                        inputLocal,
                        inputLocal,
                        inputCount);
                    ReduceRowsInto(
                        inputLocal,
                        accumulateLocal,
                        currentReduceRows,
                        paddedInner,
                        floatPadded);
                } else if constexpr (
                    std::is_same<T, half>::value) {
                    AscendC::Mul(
                        inputLocal,
                        inputLocal,
                        inputLocal,
                        inputCount);
                    AscendC::Cast(
                        valueLocal,
                        inputLocal,
                        AscendC::RoundMode::CAST_NONE,
                        inputCount);
                    ReduceRowsInto(
                        valueLocal,
                        accumulateLocal,
                        currentReduceRows,
                        paddedInner,
                        floatPadded);
                } else {
                    AscendC::Cast(
                        valueLocal,
                        inputLocal,
                        AscendC::RoundMode::CAST_NONE,
                        inputCount);
                    AscendC::Mul(
                        valueLocal,
                        valueLocal,
                        valueLocal,
                        inputCount);
                    AscendC::Cast(
                        inputLocal,
                        valueLocal,
                        AscendC::RoundMode::CAST_RINT,
                        inputCount);
                    AscendC::Cast(
                        valueLocal,
                        inputLocal,
                        AscendC::RoundMode::CAST_NONE,
                        inputCount);
                    ReduceRowsInto(
                        valueLocal,
                        accumulateLocal,
                        currentReduceRows,
                        paddedInner,
                        floatPadded);
                }
                AscendC::SetFlag<
                    AscendC::HardEvent::V_MTE2>(
                    vToMte2Event_);
                AscendC::WaitFlag<
                    AscendC::HardEvent::V_MTE2>(
                    vToMte2Event_);
                reduceOffset += currentReduceRows;
            }

            if constexpr (std::is_same<T, float>::value) {
                AscendC::Adds(
                    outputLocal,
                    accumulateLocal,
                    0.0f,
                    floatPadded);
            } else if constexpr (
                std::is_same<T, half>::value) {
                AscendC::Cast(
                    outputLocal,
                    accumulateLocal,
                    AscendC::RoundMode::CAST_NONE,
                    floatPadded);
            } else {
                AscendC::Cast(
                    outputLocal,
                    accumulateLocal,
                    AscendC::RoundMode::CAST_RINT,
                    floatPadded);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::DataCopyExtParams outputCopy;
            outputCopy.blockCount = 1;
            outputCopy.blockLen = current * sizeof(T);
            outputCopy.srcStride = 0;
            outputCopy.dstStride = 0;
            AscendC::DataCopyPad(
                outputGm_[outputIndex],
                outputLocal,
                outputCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                mte3ToVEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                mte3ToVEvent_);
            processed += current;
        }
    }

    __aicore__ inline void ProcessMiddleGroup()
    {
        AscendC::LocalTensor<T> inputLocal =
            inputBuffer_.Get<T>();
        AscendC::LocalTensor<T> outputLocal =
            outputBuffer_.Get<T>();
        AscendC::LocalTensor<float> valueLocal =
            floatBuffer_.Get<float>();
        AscendC::LocalTensor<float> accumulateLocal =
            reduceWorkBuffer_.Get<float>();

        uint64_t processed = 0;
        while (processed < outputs_) {
            const uint64_t outputIndex =
                firstOutput_ + processed;
            const uint64_t outerIndex =
                outputIndex / innerElements_;
            const uint64_t innerIndex =
                outputIndex % innerElements_;
            uint64_t current64 = outputs_ - processed;
            const uint64_t untilRowEnd =
                innerElements_ - innerIndex;
            if (current64 > untilRowEnd) {
                current64 = untilRowEnd;
            }
            if (current64 > TILE_OUTPUTS) {
                current64 = TILE_OUTPUTS;
            }
            const uint32_t current =
                static_cast<uint32_t>(current64);
            const uint32_t floatPadded =
                (current + 7U) / 8U * 8U;
            AscendC::Duplicate(
                accumulateLocal,
                0.0f,
                floatPadded);

            for (uint64_t reduceIndex = 0;
                 reduceIndex < reduceElements_;
                 ++reduceIndex) {
                uint64_t inputStart = 0;
                if (fastPath_ == 2) {
                    inputStart =
                        (outerIndex * reduceElements_ +
                         reduceIndex) *
                            innerElements_ +
                        innerIndex;
                } else {
                    inputStart =
                        BaseInputOffset(outputIndex) +
                        ReduceInputOffset(reduceIndex);
                }
                const uint32_t elementsPerBlock =
                    32U / sizeof(T);
                const uint32_t inputPadded =
                    (current + elementsPerBlock - 1U) /
                    elementsPerBlock * elementsPerBlock;
                AscendC::DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = current * sizeof(T);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                AscendC::DataCopyPadExtParams<T> padParams;
                padParams.isPad = inputPadded != current;
                padParams.leftPadding = 0;
                padParams.rightPadding =
                    static_cast<uint8_t>(
                        inputPadded - current);
                const T zero = {};
                padParams.paddingValue = zero;
                AscendC::DataCopyPad(
                    inputLocal,
                    inputGm_[inputStart],
                    copyParams,
                    padParams);
                AscendC::SetFlag<
                    AscendC::HardEvent::MTE2_V>(
                    mte2ToVEvent_);
                AscendC::WaitFlag<
                    AscendC::HardEvent::MTE2_V>(
                    mte2ToVEvent_);

                if constexpr (
                    std::is_same<T, float>::value) {
                    AscendC::Mul(
                        inputLocal,
                        inputLocal,
                        inputLocal,
                        inputPadded);
                    AscendC::Add(
                        accumulateLocal,
                        accumulateLocal,
                        inputLocal,
                        floatPadded);
                } else if constexpr (
                    std::is_same<T, half>::value) {
                    AscendC::Mul(
                        inputLocal,
                        inputLocal,
                        inputLocal,
                        inputPadded);
                    AscendC::Cast(
                        valueLocal,
                        inputLocal,
                        AscendC::RoundMode::CAST_NONE,
                        inputPadded);
                    AscendC::Add(
                        accumulateLocal,
                        accumulateLocal,
                        valueLocal,
                        floatPadded);
                } else {
                    AscendC::Cast(
                        valueLocal,
                        inputLocal,
                        AscendC::RoundMode::CAST_NONE,
                        inputPadded);
                    AscendC::Mul(
                        valueLocal,
                        valueLocal,
                        valueLocal,
                        inputPadded);
                    AscendC::Cast(
                        inputLocal,
                        valueLocal,
                        AscendC::RoundMode::CAST_RINT,
                        inputPadded);
                    AscendC::Cast(
                        valueLocal,
                        inputLocal,
                        AscendC::RoundMode::CAST_NONE,
                        inputPadded);
                    AscendC::Add(
                        accumulateLocal,
                        accumulateLocal,
                        valueLocal,
                        floatPadded);
                }
                AscendC::SetFlag<
                    AscendC::HardEvent::V_MTE2>(
                    vToMte2Event_);
                AscendC::WaitFlag<
                    AscendC::HardEvent::V_MTE2>(
                    vToMte2Event_);
            }

            if constexpr (std::is_same<T, float>::value) {
                AscendC::Adds(
                    outputLocal,
                    accumulateLocal,
                    0.0f,
                    floatPadded);
            } else if constexpr (
                std::is_same<T, half>::value) {
                AscendC::Cast(
                    outputLocal,
                    accumulateLocal,
                    AscendC::RoundMode::CAST_NONE,
                    floatPadded);
            } else {
                AscendC::Cast(
                    outputLocal,
                    accumulateLocal,
                    AscendC::RoundMode::CAST_RINT,
                    floatPadded);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::DataCopyExtParams outputCopy;
            outputCopy.blockCount = 1;
            outputCopy.blockLen = current * sizeof(T);
            outputCopy.srcStride = 0;
            outputCopy.dstStride = 0;
            AscendC::DataCopyPad(
                outputGm_[outputIndex],
                outputLocal,
                outputCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                mte3ToVEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                mte3ToVEvent_);
            processed += current;
        }
    }

    // Both traversals below compute the same sum of squares over
    // [inputStart, inputStart + elementCount) and differ only in
    // how they move data and where partial sums live.
    __aicore__ inline float ReduceContiguous(
        const uint64_t inputStart,
        const uint64_t elementCount)
    {
        if (elementCount < PIPELINED_REDUCE_MIN_ELEMENTS) {
            return ReduceContiguousDirect(inputStart, elementCount);
        }
        return ReduceContiguousPipelined(inputStart, elementCount);
    }

    __aicore__ inline float ReduceContiguousPipelined(
        const uint64_t inputStart,
        const uint64_t elementCount)
    {
        AscendC::LocalTensor<T> inputLocal =
            inputBuffer_.Get<T>();
        AscendC::LocalTensor<float> floatLocal =
            floatBuffer_.Get<float>();
        AscendC::LocalTensor<float> workLocal =
            reduceWorkBuffer_.Get<float>();
        AscendC::LocalTensor<float> sumLocal =
            sumBuffer_.Get<float>();

        // The two halves of inputBuffer_ ping-pong so that the
        // MTE2 fetch of chunk n+1 overlaps the vector work of
        // chunk n. Partial sums stay in a float vector for the
        // whole traversal; the scalar unit is touched once, after
        // the loop, because a per-chunk V_S round trip stalls
        // MTE2 issue and that dominates at these sizes.
        AscendC::LocalTensor<T> buffers[2] = {
            inputLocal[0], inputLocal[HALF_CHUNK]};
        const event_t loadEvents[2] = {
            mte2ToVEvent_, mte2ToVEventAlt_};
        const event_t freeEvents[2] = {
            vToMte2Event_, vToMte2EventAlt_};

        // The accumulator only needs as many lanes as the widest
        // chunk actually uses. Sizing it to HALF_CHUNK regardless
        // would make the Duplicate and the final ReduceSum cost
        // the same for a 64-element reduction as for a full one,
        // which is a large loss on the short repeated calls that
        // the grouped path (fastPath 3) makes.
        const uint32_t accumWidth = PadToBlockFloat(
            static_cast<uint32_t>(
                elementCount < HALF_CHUNK ? elementCount
                                          : HALF_CHUNK));
        AscendC::Duplicate(workLocal, 0.0f, accumWidth);

        const uint64_t chunks =
            (elementCount + HALF_CHUNK - 1U) / HALF_CHUNK;
        uint32_t slot = 0;
        IssueChunkLoad(buffers[0], inputStart, 0, elementCount,
                       loadEvents[0]);

        for (uint64_t index = 0; index < chunks; ++index) {
            const uint64_t offset = index * HALF_CHUNK;
            const uint32_t current = static_cast<uint32_t>(
                elementCount - offset < HALF_CHUNK
                    ? elementCount - offset
                    : HALF_CHUNK);
            const uint32_t padded = PadToBlock(current);
            const uint32_t next = slot ^ 1U;

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                loadEvents[slot]);
            if (index + 1U < chunks) {
                // Iteration 0 prefetches into a half nobody has
                // consumed yet, so there is no V_MTE2 to wait on.
                if (index > 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                        freeEvents[next]);
                }
                IssueChunkLoad(buffers[next], inputStart,
                               offset + HALF_CHUNK, elementCount,
                               loadEvents[next]);
            }

            AccumulateSquares(buffers[slot], floatLocal, workLocal,
                              current, padded);

            if (index + 2U < chunks) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                    freeEvents[slot]);
            }
            slot = next;
        }

        // The staging halves are dead now, so the larger of them
        // serves as ReduceSum scratch; floatBuffer_ is only 32B on
        // the float LONG_CHUNK instantiation.
        AscendC::LocalTensor<float> foldWork =
            inputBuffer_.Get<float>();
        AscendC::ReduceSum(
            sumLocal, workLocal, foldWork, accumWidth);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(vToSEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(vToSEvent_);
        return sumLocal.GetValue(0);
    }

    // Single-buffered traversal reducing each chunk to a scalar as
    // it lands. Uses the full CHUNK staging area, since there is no
    // second half to reserve, so a sub-threshold reduction is
    // always one DMA and one ReduceSum.
    __aicore__ inline float ReduceContiguousDirect(
        const uint64_t inputStart,
        const uint64_t elementCount)
    {
        AscendC::LocalTensor<T> inputLocal =
            inputBuffer_.Get<T>();
        AscendC::LocalTensor<float> floatLocal =
            floatBuffer_.Get<float>();
        AscendC::LocalTensor<float> workLocal =
            reduceWorkBuffer_.Get<float>();
        AscendC::LocalTensor<float> sumLocal =
            sumBuffer_.Get<float>();
        float total = 0.0f;

        for (uint64_t offset = 0; offset < elementCount;
             offset += CHUNK) {
            const uint32_t current = static_cast<uint32_t>(
                elementCount - offset < CHUNK
                    ? elementCount - offset
                    : CHUNK);
            const uint32_t padded = PadToBlock(current);

            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = current * sizeof(T);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<T> padParams;
            padParams.isPad = padded != current;
            padParams.leftPadding = 0;
            padParams.rightPadding =
                static_cast<uint8_t>(padded - current);
            const T zero = {};
            padParams.paddingValue = zero;
            AscendC::DataCopyPad(
                inputLocal, inputGm_[inputStart + offset],
                copyParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
                mte2ToVEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                mte2ToVEvent_);

            SquareIntoFloat(inputLocal, floatLocal, padded);
            if constexpr (std::is_same<T, float>::value) {
                AscendC::ReduceSum(
                    sumLocal, inputLocal, workLocal, current);
            } else {
                AscendC::ReduceSum(
                    sumLocal, floatLocal, workLocal, current);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_S>(vToSEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(vToSEvent_);
            total += sumLocal.GetValue(0);

            if (offset + CHUNK < elementCount) {
                // inputLocal was overwritten by the squaring above
                // and is refilled next iteration; on the float
                // instantiation that write is the vector op the
                // next DMA must not race.
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                    vToMte2Event_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                    vToMte2Event_);
            }
        }
        return total;
    }

    // Rounds up to a 32B block of floats, which is what the
    // accumulator and the final fold are sized in. Distinct from
    // PadToBlock, which is in units of T.
    __aicore__ inline uint32_t PadToBlockFloat(
        const uint32_t count) const
    {
        constexpr uint32_t floatsPerBlock = 32U / sizeof(float);
        return (count + floatsPerBlock - 1U) / floatsPerBlock *
               floatsPerBlock;
    }

    __aicore__ inline uint32_t PadToBlock(
        const uint32_t count) const
    {
        const uint32_t elementsPerBlock = 32U / sizeof(T);
        return (count + elementsPerBlock - 1U) /
               elementsPerBlock * elementsPerBlock;
    }

    __aicore__ inline void IssueChunkLoad(
        const AscendC::LocalTensor<T>& dst,
        const uint64_t inputStart,
        const uint64_t offset,
        const uint64_t elementCount,
        const event_t loadEvent)
    {
        const uint32_t current = static_cast<uint32_t>(
            elementCount - offset < HALF_CHUNK
                ? elementCount - offset
                : HALF_CHUNK);
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = current * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> padParams;
        padParams.isPad = PadToBlock(current) != current;
        padParams.leftPadding = 0;
        padParams.rightPadding = static_cast<uint8_t>(
            PadToBlock(current) - current);
        const T zero = {};
        padParams.paddingValue = zero;
        AscendC::DataCopyPad(
            dst, inputGm_[inputStart + offset], copyParams,
            padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(loadEvent);
    }

    // Squares `padded` elements of `src` in place. The result is
    // left in `src` for float and in `floatLocal` otherwise. The
    // bf16 round trip through T reproduces the reference operator's
    // intermediate rounding, so it must stay even though it looks
    // redundant next to the float path.
    __aicore__ inline void SquareIntoFloat(
        const AscendC::LocalTensor<T>& src,
        const AscendC::LocalTensor<float>& floatLocal,
        const uint32_t padded)
    {
        if constexpr (std::is_same<T, float>::value) {
            AscendC::Mul(src, src, src, padded);
        } else if constexpr (std::is_same<T, half>::value) {
            AscendC::Mul(src, src, src, padded);
            AscendC::Cast(
                floatLocal, src,
                AscendC::RoundMode::CAST_NONE, padded);
        } else {
            AscendC::Cast(
                floatLocal, src,
                AscendC::RoundMode::CAST_NONE, padded);
            AscendC::Mul(
                floatLocal, floatLocal, floatLocal, padded);
            AscendC::Cast(
                src, floatLocal,
                AscendC::RoundMode::CAST_RINT, padded);
            AscendC::Cast(
                floatLocal, src,
                AscendC::RoundMode::CAST_NONE, padded);
        }
    }

    // Adds the squares of the first `current` elements of `src`
    // into `accum`. Only `current` lanes are touched, so the
    // block padding beyond the tail never reaches the sum.
    __aicore__ inline void AccumulateSquares(
        const AscendC::LocalTensor<T>& src,
        const AscendC::LocalTensor<float>& floatLocal,
        const AscendC::LocalTensor<float>& accum,
        const uint32_t current,
        const uint32_t padded)
    {
        SquareIntoFloat(src, floatLocal, padded);
        if constexpr (std::is_same<T, float>::value) {
            AscendC::Add(accum, accum, src, current);
        } else {
            AscendC::Add(accum, accum, floatLocal, current);
        }
    }

    __aicore__ inline uint64_t BaseInputOffset(
        uint64_t outputIndex) const
    {
        uint64_t inputOffset = 0;
        for (int32_t axis =
                 static_cast<int32_t>(outputRank_) - 1;
             axis >= 0;
             --axis) {
            const uint64_t coordinate =
                outputIndex % outputDims_[axis];
            outputIndex /= outputDims_[axis];
            inputOffset +=
                coordinate * outputInputStrides_[axis];
        }
        return inputOffset;
    }

    __aicore__ inline uint64_t ReduceInputOffset(
        uint64_t reduceIndex) const
    {
        uint64_t inputOffset = 0;
        for (int32_t axis =
                 static_cast<int32_t>(reduceRank_) - 1;
             axis >= 0;
             --axis) {
            const uint64_t coordinate =
                reduceIndex % reduceDims_[axis];
            reduceIndex /= reduceDims_[axis];
            inputOffset +=
                coordinate * reduceInputStrides_[axis];
        }
        return inputOffset;
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceWorkBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuffer_;
    AscendC::GlobalTensor<T> inputGm_;
    AscendC::GlobalTensor<T> outputGm_;
    AscendC::GlobalTensor<float> workspaceGm_;
    uint64_t inputElements_;
    uint64_t outputElements_;
    uint64_t reduceElements_;
    uint64_t innerElements_;
    uint64_t trailingReduceElements_;
    uint64_t firstOutput_;
    uint64_t outputs_;
    uint32_t outputRank_;
    uint32_t reduceRank_;
    uint32_t fastPath_;
    uint32_t reduceMode_;
    uint64_t partialStride_;
    uint64_t outputDims_[MAX_RANK];
    uint64_t outputInputStrides_[MAX_RANK];
    uint64_t reduceDims_[MAX_RANK];
    uint64_t reduceInputStrides_[MAX_RANK];
    event_t sToMte3Event_;
    event_t mte3ToSEvent_;
    event_t mte2ToVEvent_;
    event_t mte2ToVEventAlt_;
    event_t vToMte2EventAlt_;
    event_t vToSEvent_;
    event_t vToMte2Event_;
    event_t vToMte3Event_;
    event_t mte3ToVEvent_;
};

extern "C" __global__ __aicore__ void square_sum_v1(
    GM_ADDR input,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA(tilingData, tiling);
        if (tilingData.outputElements == 0) {
            return;
        }
        GM_ADDR userWorkspace =
            tilingData.reduceMode == 2U
                ? AscendC::GetUserWorkspace(workspace)
                : workspace;
        KernelSquareSumV1<DTYPE_INPUT, NORMAL_CHUNK> op;
        op.Init(input, output, userWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        GET_TILING_DATA(tilingData, tiling);
        if (tilingData.outputElements == 0) {
            return;
        }
        GM_ADDR userWorkspace =
            tilingData.reduceMode == 2U
                ? AscendC::GetUserWorkspace(workspace)
                : workspace;
        KernelSquareSumV1<DTYPE_INPUT, LONG_CHUNK> op;
        op.Init(input, output, userWorkspace, tilingData);
        op.Process();
    }
}
