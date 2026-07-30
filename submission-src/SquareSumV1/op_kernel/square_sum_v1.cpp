#include "kernel_operator.h"

constexpr uint32_t TILE_OUTPUTS = 1024;
constexpr uint32_t MAX_RANK = 5;
constexpr uint32_t NORMAL_CHUNK = 8192;
constexpr uint32_t LONG_CHUNK = 16384;

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

template <typename T, uint32_t CHUNK, bool TREE_FINALIZE>
class KernelSquareSumV1 {
public:
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
        if (reduceMode_ != 0U) {
            partialStride_ =
                (outputElements_ + 7U) / 8U * 8U;
            workspaceGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(workspace),
                partialStride_ *
                    (reduceMode_ == 2U
                        ? AscendC::GetBlockNum()
                        : 1U));
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
        vToSEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::V_S));
        vToMte2Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::V_MTE2));
        vToMte3Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::V_MTE3));
        mte3ToVEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE3_V));
        mte2ToMte3Event_ = static_cast<event_t>(
            pipe_.FetchEventID(
                AscendC::HardEvent::MTE2_MTE3));
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
            if constexpr (TREE_FINALIZE) {
                FinalizeParallelReductionTree();
            } else {
                FinalizeParallelReductionSequential();
            }
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

    __aicore__ inline void FinalizeParallelReductionSequential()
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
                copyParams.blockLen =
                    current * sizeof(float);
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

    __aicore__ inline void FinalizeParallelReductionTree()
    {
        AscendC::LocalTensor<T> outputLocal =
            outputBuffer_.Get<T>();
        AscendC::LocalTensor<float> valueLocal =
            floatBuffer_.Get<float>();
        AscendC::LocalTensor<float> accumulateLocal =
            reduceWorkBuffer_.Get<float>();
        const uint32_t blockNum =
            AscendC::GetBlockNum();
        uint32_t reductionRows = 1U;
        while (reductionRows < blockNum) {
            reductionRows <<= 1U;
        }
        uint32_t finalTileOutputs =
            CHUNK / reductionRows;
        if (finalTileOutputs > TILE_OUTPUTS) {
            finalTileOutputs = TILE_OUTPUTS;
        }
        finalTileOutputs =
            finalTileOutputs / 8U * 8U;
        if (finalTileOutputs == 0U) {
            finalTileOutputs = 8U;
        }

        for (uint64_t outputStart = 0;
             outputStart < outputElements_;
             outputStart += finalTileOutputs) {
            const uint32_t current = static_cast<uint32_t>(
                outputElements_ - outputStart <
                        finalTileOutputs
                    ? outputElements_ - outputStart
                    : finalTileOutputs);
            const uint32_t floatPadded =
                (current + 7U) / 8U * 8U;
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount =
                static_cast<uint16_t>(blockNum);
            copyParams.blockLen =
                current * sizeof(float);
            copyParams.srcStride =
                static_cast<uint32_t>(
                    (partialStride_ - current) *
                    sizeof(float));
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
                workspaceGm_[outputStart],
                copyParams,
                padParams);
            AscendC::Duplicate(
                accumulateLocal,
                0.0f,
                floatPadded);
            if (reductionRows > blockNum) {
                AscendC::Duplicate(
                    valueLocal[
                        blockNum * floatPadded],
                    0.0f,
                    (reductionRows - blockNum) *
                        floatPadded);
            }
            AscendC::SetFlag<
                AscendC::HardEvent::MTE2_V>(
                mte2ToVEvent_);
            AscendC::WaitFlag<
                AscendC::HardEvent::MTE2_V>(
                mte2ToVEvent_);
            ReduceRowsInto(
                valueLocal,
                accumulateLocal,
                reductionRows,
                floatPadded,
                floatPadded);
            AscendC::SetFlag<
                AscendC::HardEvent::V_MTE2>(
                vToMte2Event_);
            AscendC::WaitFlag<
                AscendC::HardEvent::V_MTE2>(
                vToMte2Event_);

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
                workspaceGm_,
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
            workspaceGm_,
            outputLocal,
            alignedOutputs);
        AscendC::SetAtomicNone();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(
            mte3ToSEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(
            mte3ToSEvent_);
        AscendC::SyncAll<true>();

        if (blockIdx == 0U) {
            AscendC::DataCopy(
                outputLocal,
                workspaceGm_,
                alignedOutputs);
            AscendC::SetFlag<
                AscendC::HardEvent::MTE2_MTE3>(
                mte2ToMte3Event_);
            AscendC::WaitFlag<
                AscendC::HardEvent::MTE2_MTE3>(
                mte2ToMte3Event_);
            AscendC::DataCopyExtParams outputCopy;
            outputCopy.blockCount = 1;
            outputCopy.blockLen =
                outputElements_ * sizeof(T);
            outputCopy.srcStride = 0;
            outputCopy.dstStride = 0;
            AscendC::DataCopyPad(
                outputGm_,
                outputLocal,
                outputCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(
                mte3ToSEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(
                mte3ToSEvent_);
        }
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
            reduceRowsPerTile =
                HighestPowerOfTwo(reduceRowsPerTile);
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
                uint64_t rowOffset = 0;
                while (rowOffset < lastReduceDim) {
                    const uint32_t remainingRows =
                        static_cast<uint32_t>(
                            lastReduceDim - rowOffset <
                                    reduceRowsPerTile
                                ? lastReduceDim - rowOffset
                                : reduceRowsPerTile);
                    const uint32_t currentReduceRows =
                        remainingRows;
                    uint32_t reductionRows = 1U;
                    while (reductionRows <
                           currentReduceRows) {
                        reductionRows <<= 1U;
                    }
                    const uint32_t inputCount =
                        currentReduceRows * paddedInner;
                    const uint32_t paddedRowElements =
                        (reductionRows -
                         currentReduceRows) *
                        paddedInner;
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
                        if (paddedRowElements != 0U) {
                            AscendC::Duplicate(
                                inputLocal[inputCount],
                                0.0f,
                                paddedRowElements);
                        }
                        ReduceRowsInto(
                            inputLocal,
                            accumulateLocal,
                            reductionRows,
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
                        if (paddedRowElements != 0U) {
                            AscendC::Duplicate(
                                valueLocal[inputCount],
                                0.0f,
                                paddedRowElements);
                        }
                        ReduceRowsInto(
                            valueLocal,
                            accumulateLocal,
                            reductionRows,
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
                        if (paddedRowElements != 0U) {
                            AscendC::Duplicate(
                                valueLocal[inputCount],
                                0.0f,
                                paddedRowElements);
                        }
                        ReduceRowsInto(
                            valueLocal,
                            accumulateLocal,
                            reductionRows,
                            paddedInner,
                            floatPadded);
                    }
                    AscendC::SetFlag<
                        AscendC::HardEvent::V_MTE2>(
                        vToMte2Event_);
                    AscendC::WaitFlag<
                        AscendC::HardEvent::V_MTE2>(
                        vToMte2Event_);
                    rowOffset += currentReduceRows;
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

    __aicore__ inline float ReduceContiguous(
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

        for (uint64_t offset = 0;
             offset < elementCount;
              offset += CHUNK) {
            const uint32_t current = static_cast<uint32_t>(
                elementCount - offset < CHUNK
                    ? elementCount - offset
                    : CHUNK);
            const uint32_t elementsPerBlock =
                32U / sizeof(T);
            const uint32_t padded =
                (current + elementsPerBlock - 1U) /
                elementsPerBlock * elementsPerBlock;

            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = current * sizeof(T);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<T> padParams;
            padParams.isPad = padded != current;
            padParams.leftPadding = 0;
            padParams.rightPadding = static_cast<uint8_t>(
                padded - current);
            const T zero = {};
            padParams.paddingValue = zero;
            AscendC::DataCopyPad(
                inputLocal,
                inputGm_[inputStart + offset],
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
                    padded);
                AscendC::ReduceSum(
                    sumLocal,
                    inputLocal,
                    workLocal,
                    current);
            } else if constexpr (
                std::is_same<T, half>::value) {
                AscendC::Mul(
                    inputLocal,
                    inputLocal,
                    inputLocal,
                    padded);
                AscendC::Cast(
                    floatLocal,
                    inputLocal,
                    AscendC::RoundMode::CAST_NONE,
                    padded);
                AscendC::ReduceSum(
                    sumLocal,
                    floatLocal,
                    workLocal,
                    current);
            } else {
                AscendC::Cast(
                    floatLocal,
                    inputLocal,
                    AscendC::RoundMode::CAST_NONE,
                    padded);
                AscendC::Mul(
                    floatLocal,
                    floatLocal,
                    floatLocal,
                    padded);
                AscendC::Cast(
                    inputLocal,
                    floatLocal,
                    AscendC::RoundMode::CAST_RINT,
                    padded);
                AscendC::Cast(
                    floatLocal,
                    inputLocal,
                    AscendC::RoundMode::CAST_NONE,
                    padded);
                AscendC::ReduceSum(
                    sumLocal,
                    floatLocal,
                    workLocal,
                    current);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_S>(
                vToSEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(
                vToSEvent_);
            total += sumLocal.GetValue(0);
        }
        return total;
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
    event_t vToSEvent_;
    event_t vToMte2Event_;
    event_t vToMte3Event_;
    event_t mte3ToVEvent_;
    event_t mte2ToMte3Event_;
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
            tilingData.reduceMode != 0U
                ? AscendC::GetUserWorkspace(workspace)
                : workspace;
        KernelSquareSumV1<
            DTYPE_INPUT,
            NORMAL_CHUNK,
            false> op;
        op.Init(input, output, userWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        GET_TILING_DATA(tilingData, tiling);
        if (tilingData.outputElements == 0) {
            return;
        }
        GM_ADDR userWorkspace =
            tilingData.reduceMode != 0U
                ? AscendC::GetUserWorkspace(workspace)
                : workspace;
        KernelSquareSumV1<
            DTYPE_INPUT,
            LONG_CHUNK,
            false> op;
        op.Init(input, output, userWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        GET_TILING_DATA(tilingData, tiling);
        if (tilingData.outputElements == 0) {
            return;
        }
        GM_ADDR userWorkspace =
            tilingData.reduceMode != 0U
                ? AscendC::GetUserWorkspace(workspace)
                : workspace;
        KernelSquareSumV1<
            DTYPE_INPUT,
            NORMAL_CHUNK,
            true> op;
        op.Init(input, output, userWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        GET_TILING_DATA(tilingData, tiling);
        if (tilingData.outputElements == 0) {
            return;
        }
        GM_ADDR userWorkspace =
            tilingData.reduceMode != 0U
                ? AscendC::GetUserWorkspace(workspace)
                : workspace;
        KernelSquareSumV1<
            DTYPE_INPUT,
            LONG_CHUNK,
            true> op;
        op.Init(input, output, userWorkspace, tilingData);
        op.Process();
    }
}
