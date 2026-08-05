#include "kernel_operator.h"

constexpr uint32_t TILE_OUTPUTS = 1024;
constexpr uint32_t MAX_RANK = 5;
constexpr uint32_t NORMAL_CHUNK = 8192;
constexpr uint32_t LONG_CHUNK = 16384;
constexpr uint32_t FP32_LONG_TREE_OUTPUTS = 232;
constexpr uint32_t FP32_LONG_TREE_FLOAT_ELEMENTS =
    64U * FP32_LONG_TREE_OUTPUTS;

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

template <
    typename T,
    uint32_t CHUNK,
    bool TREE_FINALIZE,
    bool GROUPED_VECTOR8,
    bool GROUPED_VECTOR_ADAPTIVE,
    uint32_t GROUPED_VECTOR_WIDTH,
    bool EMPTY_REDUCTION = false>
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
        const uint64_t scheduledOutputs =
            tiling.baseOutputsPerBlock +
            (blockIdx < tiling.extraBlocks ? 1U : 0U);
        const uint64_t firstScheduledOutput =
            blockIdx * tiling.baseOutputsPerBlock +
            (blockIdx < tiling.extraBlocks
                ? blockIdx
                : tiling.extraBlocks);
        outputs_ = scheduledOutputs;
        firstOutput_ = firstScheduledOutput;

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
                ? (TREE_FINALIZE
                    ? FP32_LONG_TREE_FLOAT_ELEMENTS *
                        sizeof(float)
                    : TILE_OUTPUTS * sizeof(float))
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
        if constexpr (EMPTY_REDUCTION) {
            ProcessEmptyReduction();
            return;
        }
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
        if constexpr (GROUPED_VECTOR_ADAPTIVE) {
            if (fastPath_ == 1U &&
                reduceElements_ > 4096U) {
                ProcessLastAxisVector4Long();
            } else if (trailingReduceElements_ >= 1024U) {
                ProcessGroupedSuffixVector8LongTail();
            } else {
                ProcessGroupedSuffixVector8FlatRows();
            }
            return;
        }
        if constexpr (GROUPED_VECTOR8) {
            if (fastPath_ == 1U &&
                reduceElements_ > 64U) {
                ProcessLastAxisVectorLong(
                    GROUPED_VECTOR_WIDTH,
                    CHUNK / GROUPED_VECTOR_WIDTH);
                return;
            }
            if (trailingReduceElements_ >= 1024U) {
                ProcessGroupedSuffixVector8LongTail();
                return;
            }
            const uint32_t elementsPerBlock =
                32U / sizeof(T);
            if constexpr (GROUPED_VECTOR_WIDTH < 8U) {
                ProcessGroupedSuffixVector8FlatRows();
            } else if (trailingReduceElements_ <= 64U &&
                       trailingReduceElements_ %
                               elementsPerBlock == 0U) {
                ProcessGroupedSuffixVector8();
            } else if (trailingReduceElements_ <= 64U) {
                ProcessGroupedSuffixVector8FlatRows();
            } else {
                ProcessGroupedSuffixVector8FlatRows();
            }
            return;
        }
        if (fastPath_ == 1) {
            const uint32_t elementsPerBlock =
                32U / sizeof(T);
            const uint64_t paddedReduce =
                (reduceElements_ + elementsPerBlock - 1U) /
                elementsPerBlock * elementsPerBlock;
            if (reduceElements_ > 64U &&
                reduceElements_ <= 4096U &&
                outputElements_ >= 32U) {
                ProcessLastAxisSegmentedRows();
                return;
            }
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
    __aicore__ inline void ProcessEmptyReduction()
    {
        AscendC::LocalTensor<T> outputLocal =
            outputBuffer_.Get<T>();
        const uint32_t elementsPerBlock =
            32U / sizeof(T);

        for (uint64_t tileOffset = 0;
             tileOffset < outputs_;
             tileOffset += TILE_OUTPUTS) {
            const uint32_t current = static_cast<uint32_t>(
                outputs_ - tileOffset < TILE_OUTPUTS
                    ? outputs_ - tileOffset
                    : TILE_OUTPUTS);
            const uint32_t padded =
                (current + elementsPerBlock - 1U) /
                elementsPerBlock * elementsPerBlock;
            AscendC::Duplicate(
                outputLocal,
                static_cast<T>(0),
                padded);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);

            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = current * sizeof(T);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(
                outputGm_[firstOutput_ + tileOffset],
                outputLocal,
                copyParams);
            if (tileOffset + current < outputs_) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                    mte3ToVEvent_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                    mte3ToVEvent_);
            }
        }
    }

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
        // ReduceContiguous reuses floatBuffer_ for half/BF16 conversion,
        // so batching several outputs there corrupts earlier partials.
        AscendC::LocalTensor<float> partialLocal =
            outputBuffer_.Get<float>();
        const uint64_t workspaceOffset =
            AscendC::GetBlockIdx() * partialStride_;

        for (uint64_t outputIndex = 0;
             outputIndex < outputElements_;
             ++outputIndex) {
            const uint64_t inputStart =
                BaseInputOffset(outputIndex) + reduceStart;
            partialLocal.SetValue(
                outputIndex,
                ReduceContiguous(inputStart, reduceCount));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(
            sToMte3Event_);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(
            sToMte3Event_);
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(
            outputElements_ * sizeof(float));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(
            workspaceGm_[workspaceOffset],
            partialLocal,
            copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(
            mte3ToSEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(
            mte3ToSEvent_);
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
            const bool compactFullInner8 =
                innerElements_ == 8U &&
                innerIndex == 0U &&
                current == 8U;
            const uint32_t paddedInner =
                compactFullInner8
                    ? current
                    : (current + elementsPerBlock - 1U) /
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

            uint64_t reduceOffset = 0;
            while (reduceOffset < reduceCount) {
                const uint32_t remainingRows =
                    static_cast<uint32_t>(
                        reduceCount - reduceOffset <
                                reduceRowsPerTile
                            ? reduceCount - reduceOffset
                            : reduceRowsPerTile);
                const uint32_t currentReduceRows =
                    remainingRows;
                uint32_t reductionRows = 1U;
                while (reductionRows < currentReduceRows) {
                    reductionRows <<= 1U;
                }
                const uint32_t inputCount =
                    currentReduceRows * paddedInner;
                const uint32_t paddedRowElements =
                    (reductionRows - currentReduceRows) *
                    paddedInner;
                const uint64_t inputStart =
                    (outerIndex * reduceElements_ +
                     reduceStart + reduceOffset) *
                        innerElements_ +
                    innerIndex;

                AscendC::DataCopyExtParams copyParams;
                copyParams.blockCount = compactFullInner8
                    ? 1U
                    : static_cast<uint16_t>(currentReduceRows);
                copyParams.blockLen = compactFullInner8
                    ? currentReduceRows * current * sizeof(T)
                    : current * sizeof(T);
                copyParams.srcStride = compactFullInner8
                    ? 0U
                    : static_cast<uint32_t>(
                        (innerElements_ - current) * sizeof(T));
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
                        floatPadded,
                        reduceOffset == 0U);
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
                        floatPadded,
                        reduceOffset == 0U);
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
                        floatPadded,
                        reduceOffset == 0U);
                }
                if (reduceOffset + currentReduceRows <
                        reduceCount ||
                    outputIndex + current < outputElements_) {
                    AscendC::SetFlag<
                        AscendC::HardEvent::V_MTE2>(
                        vToMte2Event_);
                    AscendC::WaitFlag<
                        AscendC::HardEvent::V_MTE2>(
                        vToMte2Event_);
                }
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
                if (block == 0U) {
                    AscendC::Adds(
                        accumulateLocal,
                        valueLocal,
                        0.0f,
                        floatPadded);
                } else {
                    AscendC::Add(
                        accumulateLocal,
                        accumulateLocal,
                        valueLocal,
                        floatPadded);
                }
                if (block + 1U < blockNum) {
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
        const uint32_t blockNum =
            AscendC::GetBlockNum();
        uint32_t reductionRows = 1U;
        while (reductionRows < blockNum) {
            reductionRows <<= 1U;
        }
        uint32_t finalTileOutputs =
            CHUNK / reductionRows;
        if constexpr (
            std::is_same<T, float>::value &&
            CHUNK == LONG_CHUNK) {
            const uint32_t capacityOutputs =
                FP32_LONG_TREE_FLOAT_ELEMENTS /
                reductionRows;
            if (finalTileOutputs > capacityOutputs) {
                finalTileOutputs = capacityOutputs;
            }
        }
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
            ReduceRowsInPlace(
                valueLocal,
                reductionRows,
                floatPadded);

            if constexpr (std::is_same<T, float>::value) {
                AscendC::Adds(
                    outputLocal,
                    valueLocal,
                    0.0f,
                    floatPadded);
            } else if constexpr (
                std::is_same<T, half>::value) {
                AscendC::Cast(
                    outputLocal,
                    valueLocal,
                    AscendC::RoundMode::CAST_NONE,
                    floatPadded);
            } else {
                AscendC::Cast(
                    outputLocal,
                    valueLocal,
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
                            floatPadded,
                            group == 0U && rowOffset == 0U);
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
                            floatPadded,
                            group == 0U && rowOffset == 0U);
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
                            floatPadded,
                            group == 0U && rowOffset == 0U);
                    }
                    if (rowOffset + currentReduceRows <
                             lastReduceDim ||
                        group + 1U < outerReduceGroups ||
                        processed + current < outputs_) {
                        AscendC::SetFlag<
                            AscendC::HardEvent::V_MTE2>(
                            vToMte2Event_);
                        AscendC::WaitFlag<
                            AscendC::HardEvent::V_MTE2>(
                            vToMte2Event_);
                    }
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

    __aicore__ inline uint64_t GroupedMaskBits(
        const uint32_t lane,
        const uint32_t count) const
    {
        if (count == 64U) {
            return ~static_cast<uint64_t>(0U);
        }
        return ((static_cast<uint64_t>(1U) << count) - 1U)
               << lane;
    }

    __aicore__ inline void ReduceGroupedMaskedRows(
        const AscendC::LocalTensor<float>& partialLocal,
        const AscendC::LocalTensor<float>& sourceLocal,
        const uint32_t activeOutputs,
        const uint32_t trailing,
        const uint32_t paddedVector,
        const uint32_t batchRowsPerTile,
        const uint32_t currentRows,
        const uint32_t paddedPartialsPerOutput)
    {
        for (uint32_t output = 0;
             output < activeOutputs;
             ++output) {
            const uint32_t outputStartElement =
                output * trailing;
            const uint32_t alignedWindowStart =
                outputStartElement / 8U * 8U;
            const uint32_t latestWindowStart =
                paddedVector - 64U;
            const uint32_t firstWindowStart =
                alignedWindowStart < latestWindowStart
                    ? alignedWindowStart
                    : latestWindowStart;
            const uint32_t firstLane =
                outputStartElement - firstWindowStart;
            const uint32_t firstCount =
                trailing < 64U - firstLane
                    ? trailing
                    : 64U - firstLane;
            const uint64_t firstMask[1] = {
                GroupedMaskBits(firstLane, firstCount)};
            AscendC::WholeReduceSum<float>(
                partialLocal[
                    output * paddedPartialsPerOutput],
                sourceLocal[firstWindowStart],
                firstMask,
                static_cast<int32_t>(currentRows),
                1,
                1,
                static_cast<int32_t>(paddedVector / 8U));

            const uint32_t remaining =
                trailing - firstCount;
            if (remaining > 0U) {
                const uint32_t secondWindowStart =
                    firstWindowStart + 8U;
                const uint32_t secondLane =
                    outputStartElement + firstCount -
                    secondWindowStart;
                const uint64_t secondMask[1] = {
                    GroupedMaskBits(secondLane, remaining)};
                AscendC::WholeReduceSum<float>(
                    partialLocal[
                        output * paddedPartialsPerOutput +
                        batchRowsPerTile],
                    sourceLocal[secondWindowStart],
                    secondMask,
                    static_cast<int32_t>(currentRows),
                    1,
                    1,
                    static_cast<int32_t>(paddedVector / 8U));
            }
        }
    }

    __aicore__ inline void ReduceGroupedMediumRows(
        const AscendC::LocalTensor<float>& partialLocal,
        const AscendC::LocalTensor<float>& sourceLocal,
        const uint32_t activeOutputs,
        const uint32_t trailing,
        const uint32_t paddedVector,
        const uint32_t batchRowsPerTile,
        const uint32_t currentRows,
        const uint32_t paddedPartialsPerOutput)
    {
        const uint32_t latestWindowStart =
            paddedVector - 64U;
        for (uint32_t output = 0;
             output < activeOutputs;
             ++output) {
            uint32_t cursor = output * trailing;
            const uint32_t outputEnd = cursor + trailing;
            uint32_t window = 0U;
            while (cursor < outputEnd) {
                const uint32_t alignedWindowStart =
                    cursor / 8U * 8U;
                const uint32_t windowStart =
                    alignedWindowStart < latestWindowStart
                        ? alignedWindowStart
                        : latestWindowStart;
                const uint32_t lane = cursor - windowStart;
                const uint32_t available = 64U - lane;
                const uint32_t remaining = outputEnd - cursor;
                const uint32_t count =
                    remaining < available
                        ? remaining
                        : available;
                const uint64_t mask[1] = {
                    GroupedMaskBits(lane, count)};
                AscendC::WholeReduceSum<float>(
                    partialLocal[
                        output * paddedPartialsPerOutput +
                        window * batchRowsPerTile],
                    sourceLocal[windowStart],
                    mask,
                    static_cast<int32_t>(currentRows),
                    1,
                    1,
                    static_cast<int32_t>(paddedVector / 8U));
                cursor += count;
                ++window;
            }
        }
    }

    __aicore__ inline void ProcessGroupedSuffixVector8LongTail()
    {
        constexpr uint32_t VECTOR_OUTPUTS =
            GROUPED_VECTOR_WIDTH;
        constexpr uint32_t ROW_CHUNK =
            CHUNK / GROUPED_VECTOR_WIDTH;
        constexpr uint32_t SEGMENT_ELEMENTS = 64U;
        AscendC::LocalTensor<T> inputLocal =
            inputBuffer_.Get<T>();
        AscendC::LocalTensor<T> outputLocal =
            outputBuffer_.Get<T>();
        AscendC::LocalTensor<float> floatLocal =
            floatBuffer_.Get<float>();
        AscendC::LocalTensor<float> partialLocal =
            reduceWorkBuffer_.Get<float>();
        AscendC::LocalTensor<float> accumulateLocal =
            sumBuffer_.Get<float>();

        uint64_t expectedStride = 1U;
        uint32_t firstTrailingAxis = reduceRank_;
        for (int32_t axis =
                 static_cast<int32_t>(reduceRank_) - 1;
             axis >= 0;
             --axis) {
            if (reduceInputStrides_[axis] != expectedStride) {
                break;
            }
            expectedStride *= reduceDims_[axis];
            firstTrailingAxis = static_cast<uint32_t>(axis);
        }
        const uint32_t batchAxis = firstTrailingAxis - 1U;
        const uint64_t batchDim = reduceDims_[batchAxis];
        const uint64_t outerGroups =
            reduceElements_ /
            (batchDim * trailingReduceElements_);
        const uint32_t trailing = static_cast<uint32_t>(
            trailingReduceElements_);
        const uint32_t elementsPerBlock = 32U / sizeof(T);
        const T zero = {};

        uint64_t adaptiveInnerDim = 0U;
        uint64_t adaptiveTasksPerRow = 0U;
        if constexpr (GROUPED_VECTOR_ADAPTIVE) {
            for (int32_t axis =
                     static_cast<int32_t>(outputRank_) - 1;
                 axis >= 0;
                 --axis) {
                if (outputDims_[axis] > 1U &&
                    outputInputStrides_[axis] != 0U) {
                    adaptiveInnerDim = outputDims_[axis];
                    break;
                }
            }
            adaptiveTasksPerRow =
                (adaptiveInnerDim + VECTOR_OUTPUTS - 1U) /
                VECTOR_OUTPUTS;
        }

        for (uint64_t taskOffset = 0;
             taskOffset < outputs_;
             ++taskOffset) {
            const uint64_t taskIndex = firstOutput_ + taskOffset;
            uint64_t outputStart = taskIndex * VECTOR_OUTPUTS;
            uint32_t activeOutputs = VECTOR_OUTPUTS;
            if constexpr (GROUPED_VECTOR_ADAPTIVE) {
                const uint64_t outputRow =
                    taskIndex / adaptiveTasksPerRow;
                const uint64_t taskInRow =
                    taskIndex % adaptiveTasksPerRow;
                const uint64_t outputInRow =
                    taskInRow * VECTOR_OUTPUTS;
                outputStart =
                    outputRow * adaptiveInnerDim + outputInRow;
                activeOutputs = static_cast<uint32_t>(
                    adaptiveInnerDim - outputInRow < VECTOR_OUTPUTS
                        ? adaptiveInnerDim - outputInRow
                        : VECTOR_OUTPUTS);
            }

            const uint64_t baseInputOffset =
                BaseInputOffset(outputStart);

            for (uint64_t group = 0;
                 group < outerGroups;
                 ++group) {
                const uint64_t groupReduceStart =
                    group * batchDim * trailingReduceElements_;
                for (uint64_t batchOffset = 0;
                     batchOffset < batchDim;
                     ++batchOffset) {
                    const uint64_t batchReduceStart =
                        groupReduceStart +
                        batchOffset * trailingReduceElements_;
                    for (uint64_t tailOffset = 0;
                         tailOffset < trailing;
                         tailOffset += ROW_CHUNK) {
                        const uint32_t current =
                            static_cast<uint32_t>(
                                trailing - tailOffset < ROW_CHUNK
                                    ? trailing - tailOffset
                                    : ROW_CHUNK);
                        const uint32_t alignedCurrent =
                            (current + elementsPerBlock - 1U) /
                            elementsPerBlock * elementsPerBlock;
                        const uint32_t currentRowElements =
                            VECTOR_OUTPUTS == 1U
                                ? (current + SEGMENT_ELEMENTS - 1U) /
                                    SEGMENT_ELEMENTS * SEGMENT_ELEMENTS
                                : ROW_CHUNK;
                        const uint32_t currentSegmentsPerRow =
                            currentRowElements / SEGMENT_ELEMENTS;
                        const uint32_t inputCount =
                            activeOutputs * currentRowElements;

                        if (current != ROW_CHUNK) {
                            AscendC::Duplicate(
                                inputLocal,
                                zero,
                                inputCount);
                            AscendC::SetFlag<
                                AscendC::HardEvent::V_MTE2>(
                                vToMte2Event_);
                            AscendC::WaitFlag<
                                AscendC::HardEvent::V_MTE2>(
                                vToMte2Event_);
                        }

                        AscendC::DataCopyExtParams copyParams;
                        copyParams.blockCount =
                            static_cast<uint16_t>(activeOutputs);
                        copyParams.blockLen =
                            current * sizeof(T);
                        copyParams.srcStride =
                            VECTOR_OUTPUTS == 1U
                                ? 0U
                                : (trailing - current) * sizeof(T);
                        copyParams.dstStride =
                            VECTOR_OUTPUTS == 1U
                                ? 0U
                                : (ROW_CHUNK - alignedCurrent) *
                                    sizeof(T) / 32U;
                        AscendC::DataCopyPadExtParams<T> padParams;
                        padParams.isPad = alignedCurrent != current;
                        padParams.leftPadding = 0;
                        padParams.rightPadding =
                            static_cast<uint8_t>(
                                alignedCurrent - current);
                        padParams.paddingValue = zero;
                        const uint64_t inputStart =
                            baseInputOffset +
                            ReduceInputOffset(batchReduceStart) +
                            tailOffset;
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
                            if (currentSegmentsPerRow > 64U &&
                                currentSegmentsPerRow % 64U != 0U) {
                                AscendC::Duplicate(
                                    partialLocal,
                                    0.0f,
                                    (currentSegmentsPerRow + 63U) /
                                        64U * 64U);
                            }
                            ReduceLastAxisLongPartials(
                                partialLocal,
                                inputLocal,
                                activeOutputs,
                                currentRowElements,
                                currentSegmentsPerRow);
                            if (currentSegmentsPerRow > 64U) {
                                AccumulateLastAxisLongLarge(
                                    partialLocal,
                                    floatLocal,
                                    accumulateLocal,
                                    currentSegmentsPerRow,
                                    group == 0U &&
                                        batchOffset == 0U &&
                                        tailOffset == 0U);
                            } else {
                                AscendC::WholeReduceSum<float>(
                                    outputLocal,
                                    partialLocal,
                                    static_cast<int32_t>(
                                        currentSegmentsPerRow),
                                    static_cast<int32_t>(
                                        activeOutputs),
                                    1,
                                    1,
                                    (currentSegmentsPerRow + 7U) / 8U);
                                if (group == 0U &&
                                    batchOffset == 0U &&
                                    tailOffset == 0U) {
                                    AscendC::Adds(
                                        accumulateLocal,
                                        outputLocal,
                                        0.0f,
                                        activeOutputs);
                                } else {
                                    AscendC::Add(
                                        accumulateLocal,
                                        accumulateLocal,
                                        outputLocal,
                                        activeOutputs);
                                }
                            }
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
                            if (currentSegmentsPerRow > 64U &&
                                currentSegmentsPerRow % 64U != 0U) {
                                AscendC::Duplicate(
                                    partialLocal,
                                    0.0f,
                                    (currentSegmentsPerRow + 63U) /
                                        64U * 64U);
                            }
                            ReduceLastAxisLongPartials(
                                partialLocal,
                                floatLocal,
                                activeOutputs,
                                currentRowElements,
                                currentSegmentsPerRow);
                            if (currentSegmentsPerRow > 64U) {
                                AccumulateLastAxisLongLarge(
                                    partialLocal,
                                    floatLocal,
                                    accumulateLocal,
                                    currentSegmentsPerRow,
                                    group == 0U &&
                                        batchOffset == 0U &&
                                        tailOffset == 0U);
                            } else {
                                AscendC::WholeReduceSum<float>(
                                    floatLocal,
                                    partialLocal,
                                    static_cast<int32_t>(
                                        currentSegmentsPerRow),
                                    static_cast<int32_t>(
                                        activeOutputs),
                                    1,
                                    1,
                                    (currentSegmentsPerRow + 7U) / 8U);
                                if (group == 0U &&
                                    batchOffset == 0U &&
                                    tailOffset == 0U) {
                                    AscendC::Adds(
                                        accumulateLocal,
                                        floatLocal,
                                        0.0f,
                                        activeOutputs);
                                } else {
                                    AscendC::Add(
                                        accumulateLocal,
                                        accumulateLocal,
                                        floatLocal,
                                        activeOutputs);
                                }
                            }
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
                            if (currentSegmentsPerRow > 64U &&
                                currentSegmentsPerRow % 64U != 0U) {
                                AscendC::Duplicate(
                                    partialLocal,
                                    0.0f,
                                    (currentSegmentsPerRow + 63U) /
                                        64U * 64U);
                            }
                            ReduceLastAxisLongPartials(
                                partialLocal,
                                floatLocal,
                                activeOutputs,
                                currentRowElements,
                                currentSegmentsPerRow);
                            if (currentSegmentsPerRow > 64U) {
                                AccumulateLastAxisLongLarge(
                                    partialLocal,
                                    floatLocal,
                                    accumulateLocal,
                                    currentSegmentsPerRow,
                                    group == 0U &&
                                        batchOffset == 0U &&
                                        tailOffset == 0U);
                            } else {
                                AscendC::WholeReduceSum<float>(
                                    floatLocal,
                                    partialLocal,
                                    static_cast<int32_t>(
                                        currentSegmentsPerRow),
                                    static_cast<int32_t>(
                                        activeOutputs),
                                    1,
                                    1,
                                    (currentSegmentsPerRow + 7U) / 8U);
                                if (group == 0U &&
                                    batchOffset == 0U &&
                                    tailOffset == 0U) {
                                    AscendC::Adds(
                                        accumulateLocal,
                                        floatLocal,
                                        0.0f,
                                        activeOutputs);
                                } else {
                                    AscendC::Add(
                                        accumulateLocal,
                                        accumulateLocal,
                                        floatLocal,
                                        activeOutputs);
                                }
                            }
                        }

                        const bool hasNext =
                            tailOffset + current < trailing ||
                            batchOffset + 1U < batchDim ||
                            group + 1U < outerGroups;
                        uint32_t nextCurrent = 0U;
                        if (hasNext) {
                            const uint64_t nextTailOffset =
                                tailOffset + current < trailing
                                    ? tailOffset + current
                                    : 0U;
                            nextCurrent = static_cast<uint32_t>(
                                trailing - nextTailOffset < ROW_CHUNK
                                    ? trailing - nextTailOffset
                                    : ROW_CHUNK);
                        }
                        if (hasNext && nextCurrent == ROW_CHUNK) {
                            AscendC::SetFlag<
                                AscendC::HardEvent::V_MTE2>(
                                vToMte2Event_);
                            AscendC::WaitFlag<
                                AscendC::HardEvent::V_MTE2>(
                                vToMte2Event_);
                        }
                    }
                }
            }

            if constexpr (std::is_same<T, float>::value) {
                AscendC::Adds(
                    outputLocal,
                    accumulateLocal,
                    0.0f,
                    activeOutputs);
            } else if constexpr (std::is_same<T, half>::value) {
                AscendC::Cast(
                    outputLocal,
                    accumulateLocal,
                    AscendC::RoundMode::CAST_NONE,
                    activeOutputs);
            } else {
                AscendC::Cast(
                    outputLocal,
                    accumulateLocal,
                    AscendC::RoundMode::CAST_RINT,
                    activeOutputs);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::DataCopyExtParams outputCopy;
            outputCopy.blockCount = 1;
            outputCopy.blockLen = activeOutputs * sizeof(T);
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

    __aicore__ inline void ProcessGroupedSuffixVector8FlatRows()
    {
        constexpr uint32_t VECTOR_OUTPUTS =
            GROUPED_VECTOR_WIDTH;
        AscendC::LocalTensor<T> inputLocal =
            inputBuffer_.Get<T>();
        AscendC::LocalTensor<T> outputLocal =
            outputBuffer_.Get<T>();
        AscendC::LocalTensor<float> floatLocal =
            floatBuffer_.Get<float>();
        AscendC::LocalTensor<float> partialLocal =
            reduceWorkBuffer_.Get<float>();
        AscendC::LocalTensor<float> accumulateLocal =
            sumBuffer_.Get<float>();

        uint64_t expectedStride = 1U;
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
        const uint32_t trailing =
            static_cast<uint32_t>(
                trailingReduceElements_);
        const uint32_t elementsPerBlock =
            32U / sizeof(T);

        uint64_t adaptiveInnerDim = 0U;
        uint64_t adaptiveTasksPerRow = 0U;
        if constexpr (GROUPED_VECTOR_ADAPTIVE) {
            for (int32_t axis =
                     static_cast<int32_t>(outputRank_) - 1;
                 axis >= 0;
                 --axis) {
                if (outputDims_[axis] > 1U &&
                    outputInputStrides_[axis] != 0U) {
                    adaptiveInnerDim = outputDims_[axis];
                    break;
                }
            }
            adaptiveTasksPerRow =
                (adaptiveInnerDim + VECTOR_OUTPUTS - 1U) /
                VECTOR_OUTPUTS;
        }

        for (uint64_t taskOffset = 0;
             taskOffset < outputs_;
             ++taskOffset) {
            const uint64_t taskIndex =
                firstOutput_ + taskOffset;
            uint64_t outputStart =
                taskIndex * VECTOR_OUTPUTS;
            uint32_t activeOutputs = VECTOR_OUTPUTS;
            if constexpr (GROUPED_VECTOR_ADAPTIVE) {
                const uint64_t outputRow =
                    taskIndex / adaptiveTasksPerRow;
                const uint64_t taskInRow =
                    taskIndex % adaptiveTasksPerRow;
                const uint64_t outputInRow =
                    taskInRow * VECTOR_OUTPUTS;
                outputStart =
                    outputRow * adaptiveInnerDim +
                    outputInRow;
                activeOutputs = static_cast<uint32_t>(
                    adaptiveInnerDim - outputInRow <
                            VECTOR_OUTPUTS
                        ? adaptiveInnerDim - outputInRow
                        : VECTOR_OUTPUTS);
            }

            const uint32_t vectorElements =
                activeOutputs * trailing;
            constexpr uint32_t WINDOWS_PER_OUTPUT = 2U;
            const bool mediumTail = trailing > 64U;
            const uint32_t maxWindows =
                mediumTail
                    ? (trailing + 70U) / 64U
                    : WINDOWS_PER_OUTPUT;
            const uint32_t alignedVector =
                (vectorElements + elementsPerBlock - 1U) /
                elementsPerBlock * elementsPerBlock;
            const uint32_t paddedVector =
                alignedVector < 64U ? 64U : alignedVector;
            uint32_t batchRowsPerTile =
                CHUNK / paddedVector;
            const uint32_t partialsPerOutput =
                batchRowsPerTile * maxWindows;
            const uint32_t paddedPartialsPerOutput =
                (partialsPerOutput + 63U) / 64U * 64U;
            const uint32_t secondPartialsPerOutput =
                paddedPartialsPerOutput / 64U;
            const uint32_t alignedSecondPartials =
                (secondPartialsPerOutput + 7U) / 8U * 8U;
            const uint32_t partialBufferCount =
                activeOutputs * paddedPartialsPerOutput;
            const T zero = {};
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockLen =
                vectorElements * sizeof(T);
            copyParams.srcStride = static_cast<uint32_t>(
                (reduceInputStrides_[batchAxis] -
                 vectorElements) * sizeof(T));
            copyParams.dstStride = static_cast<uint32_t>(
                (paddedVector - alignedVector) * sizeof(T) /
                32U);
            AscendC::DataCopyPadExtParams<T> padParams;
            padParams.isPad = alignedVector != vectorElements;
            padParams.leftPadding = 0;
            padParams.rightPadding = static_cast<uint8_t>(
                alignedVector - vectorElements);
            padParams.paddingValue = zero;

            const uint64_t baseInputOffset =
                BaseInputOffset(outputStart);
            for (uint64_t group = 0;
                 group < outerGroups;
                 ++group) {
                const uint64_t groupReduceStart =
                    group * batchDim *
                    trailingReduceElements_;
                for (uint64_t batchOffset = 0;
                     batchOffset < batchDim;
                     batchOffset += batchRowsPerTile) {
                    const uint32_t currentRows =
                        static_cast<uint32_t>(
                            batchDim - batchOffset <
                                    batchRowsPerTile
                                ? batchDim - batchOffset
                                : batchRowsPerTile);
                    const uint32_t currentInputCount =
                        currentRows * paddedVector;
                    const uint64_t inputStart =
                        baseInputOffset +
                        ReduceInputOffset(
                            groupReduceStart +
                            batchOffset *
                                trailingReduceElements_);

                    if (paddedVector != alignedVector) {
                        AscendC::Duplicate(
                            inputLocal,
                            zero,
                            currentInputCount);
                    }
                    AscendC::Duplicate(
                        partialLocal,
                        0.0f,
                        partialBufferCount);
                    AscendC::SetFlag<
                        AscendC::HardEvent::V_MTE2>(
                        vToMte2Event_);
                    AscendC::WaitFlag<
                        AscendC::HardEvent::V_MTE2>(
                        vToMte2Event_);
                    copyParams.blockCount =
                        static_cast<uint16_t>(currentRows);
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
                            currentInputCount);
                        if (mediumTail) {
                            ReduceGroupedMediumRows(
                                partialLocal,
                                inputLocal,
                                activeOutputs,
                                trailing,
                                paddedVector,
                                batchRowsPerTile,
                                currentRows,
                                paddedPartialsPerOutput);
                        } else {
                            ReduceGroupedMaskedRows(
                                partialLocal,
                                inputLocal,
                                activeOutputs,
                                trailing,
                                paddedVector,
                                batchRowsPerTile,
                                currentRows,
                                paddedPartialsPerOutput);
                        }
                    } else if constexpr (
                        std::is_same<T, half>::value) {
                        AscendC::Mul(
                            inputLocal,
                            inputLocal,
                            inputLocal,
                            currentInputCount);
                        AscendC::Cast(
                            floatLocal,
                            inputLocal,
                            AscendC::RoundMode::CAST_NONE,
                            currentInputCount);
                        if (mediumTail) {
                            ReduceGroupedMediumRows(
                                partialLocal,
                                floatLocal,
                                activeOutputs,
                                trailing,
                                paddedVector,
                                batchRowsPerTile,
                                currentRows,
                                paddedPartialsPerOutput);
                        } else {
                            ReduceGroupedMaskedRows(
                                partialLocal,
                                floatLocal,
                                activeOutputs,
                                trailing,
                                paddedVector,
                                batchRowsPerTile,
                                currentRows,
                                paddedPartialsPerOutput);
                        }
                    } else {
                        AscendC::Cast(
                            floatLocal,
                            inputLocal,
                            AscendC::RoundMode::CAST_NONE,
                            currentInputCount);
                        AscendC::Mul(
                            floatLocal,
                            floatLocal,
                            floatLocal,
                            currentInputCount);
                        AscendC::Cast(
                            inputLocal,
                            floatLocal,
                            AscendC::RoundMode::CAST_RINT,
                            currentInputCount);
                        AscendC::Cast(
                            floatLocal,
                            inputLocal,
                            AscendC::RoundMode::CAST_NONE,
                            currentInputCount);
                        if (mediumTail) {
                            ReduceGroupedMediumRows(
                                partialLocal,
                                floatLocal,
                                activeOutputs,
                                trailing,
                                paddedVector,
                                batchRowsPerTile,
                                currentRows,
                                paddedPartialsPerOutput);
                        } else {
                            ReduceGroupedMaskedRows(
                                partialLocal,
                                floatLocal,
                                activeOutputs,
                                trailing,
                                paddedVector,
                                batchRowsPerTile,
                                currentRows,
                                paddedPartialsPerOutput);
                        }
                    }
                    if (mediumTail &&
                        paddedPartialsPerOutput == 64U) {
                        AscendC::WholeReduceSum<float>(
                            floatLocal,
                            partialLocal,
                            64,
                            static_cast<int32_t>(activeOutputs),
                            1,
                            1,
                            8);
                        if (group == 0U && batchOffset == 0U) {
                            AscendC::Adds(
                                accumulateLocal,
                                floatLocal,
                                0.0f,
                                activeOutputs);
                        } else {
                            AscendC::Add(
                                accumulateLocal,
                                accumulateLocal,
                                floatLocal,
                                activeOutputs);
                        }
                    } else {
                        AscendC::Duplicate(
                            floatLocal,
                            0.0f,
                            activeOutputs *
                                alignedSecondPartials);
                        for (uint32_t output = 0;
                             output < activeOutputs;
                             ++output) {
                            AscendC::WholeReduceSum<float>(
                                floatLocal[
                                    output *
                                        alignedSecondPartials],
                                partialLocal[
                                    output *
                                        paddedPartialsPerOutput],
                                64,
                                static_cast<int32_t>(
                                    secondPartialsPerOutput),
                                1,
                                1,
                                8);
                        }
                        AscendC::WholeReduceSum<float>(
                            partialLocal,
                            floatLocal,
                            static_cast<int32_t>(
                                alignedSecondPartials),
                            static_cast<int32_t>(activeOutputs),
                            1,
                            1,
                            static_cast<int32_t>(
                                alignedSecondPartials / 8U));
                        if (group == 0U && batchOffset == 0U) {
                            AscendC::Adds(
                                accumulateLocal,
                                partialLocal,
                                0.0f,
                                activeOutputs);
                        } else {
                            AscendC::Add(
                                accumulateLocal,
                                accumulateLocal,
                                partialLocal,
                                activeOutputs);
                        }
                    }
                }
            }

            if constexpr (std::is_same<T, float>::value) {
                AscendC::Adds(
                    outputLocal,
                    accumulateLocal,
                    0.0f,
                    activeOutputs);
            } else if constexpr (
                std::is_same<T, half>::value) {
                AscendC::Cast(
                    outputLocal,
                    accumulateLocal,
                    AscendC::RoundMode::CAST_NONE,
                    activeOutputs);
            } else {
                AscendC::Cast(
                    outputLocal,
                    accumulateLocal,
                    AscendC::RoundMode::CAST_RINT,
                    activeOutputs);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::DataCopyExtParams outputCopy;
            outputCopy.blockCount = 1;
            outputCopy.blockLen =
                activeOutputs * sizeof(T);
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

    __aicore__ inline void ProcessGroupedSuffixVector8()
    {
        constexpr uint32_t VECTOR_OUTPUTS =
            GROUPED_VECTOR_WIDTH;
        AscendC::LocalTensor<T> inputLocal =
            inputBuffer_.Get<T>();
        AscendC::LocalTensor<T> outputLocal =
            outputBuffer_.Get<T>();
        AscendC::LocalTensor<float> floatLocal =
            floatBuffer_.Get<float>();
        AscendC::LocalTensor<float> partialLocal =
            reduceWorkBuffer_.Get<float>();
        AscendC::LocalTensor<float> accumulateLocal =
            sumBuffer_.Get<float>();

        uint64_t expectedStride = 1U;
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
        const uint32_t trailing =
            static_cast<uint32_t>(
                trailingReduceElements_);
        const uint32_t vectorInputElements =
            VECTOR_OUTPUTS * trailing;
        uint32_t batchRowsPerTile =
            CHUNK / vectorInputElements;
        if (batchRowsPerTile > 31U) {
            batchRowsPerTile = 31U;
        }
        if (batchRowsPerTile == 0U) {
            batchRowsPerTile = 1U;
        }
        const uint32_t sourceGapBytes =
            static_cast<uint32_t>(
                (reduceInputStrides_[batchAxis] -
                 vectorInputElements) *
                sizeof(T));

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockLen =
            vectorInputElements * sizeof(T);
        copyParams.srcStride = sourceGapBytes;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        const T zero = {};
        padParams.paddingValue = zero;

        for (uint64_t taskOffset = 0;
             taskOffset < outputs_;
             ++taskOffset) {
            const uint64_t outputStart =
                (firstOutput_ + taskOffset) *
                VECTOR_OUTPUTS;
            const uint64_t baseInputOffset =
                BaseInputOffset(outputStart);
            for (uint64_t group = 0;
                 group < outerGroups;
                 ++group) {
                const uint64_t groupReduceStart =
                    group * batchDim *
                    trailingReduceElements_;
                for (uint64_t batchOffset = 0;
                     batchOffset < batchDim;
                     batchOffset += batchRowsPerTile) {
                    const uint32_t currentRows =
                        static_cast<uint32_t>(
                            batchDim - batchOffset <
                                    batchRowsPerTile
                                ? batchDim - batchOffset
                                : batchRowsPerTile);
                    uint32_t reductionRows = 1U;
                    while (reductionRows < currentRows) {
                        reductionRows <<= 1U;
                    }
                    const uint32_t inputCount =
                        currentRows * vectorInputElements;
                    const uint32_t partialCount =
                        currentRows * VECTOR_OUTPUTS;
                    const uint32_t paddedPartialCount =
                        (reductionRows - currentRows) *
                        VECTOR_OUTPUTS;
                    const uint64_t inputStart =
                        baseInputOffset +
                        ReduceInputOffset(
                            groupReduceStart +
                            batchOffset *
                                trailingReduceElements_);
                    copyParams.blockCount =
                        static_cast<uint16_t>(currentRows);
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
                        AscendC::WholeReduceSum<float>(
                            partialLocal,
                            inputLocal,
                            static_cast<int32_t>(trailing),
                            static_cast<int32_t>(
                                currentRows *
                                VECTOR_OUTPUTS),
                            1,
                            1,
                            static_cast<int32_t>(
                                trailing / 8U));
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
                            partialLocal,
                            floatLocal,
                            static_cast<int32_t>(trailing),
                            static_cast<int32_t>(
                                currentRows *
                                VECTOR_OUTPUTS),
                            1,
                            1,
                            static_cast<int32_t>(
                                trailing / 8U));
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
                            partialLocal,
                            floatLocal,
                            static_cast<int32_t>(trailing),
                            static_cast<int32_t>(
                                currentRows *
                                VECTOR_OUTPUTS),
                            1,
                            1,
                            static_cast<int32_t>(
                                trailing / 8U));
                    }
                    if (paddedPartialCount != 0U) {
                        AscendC::Duplicate(
                            partialLocal[partialCount],
                            0.0f,
                            paddedPartialCount);
                    }
                    ReduceRowsInto(
                        partialLocal,
                        accumulateLocal,
                        reductionRows,
                        VECTOR_OUTPUTS,
                        VECTOR_OUTPUTS,
                        group == 0U && batchOffset == 0U);
                    if (batchOffset + currentRows < batchDim ||
                        group + 1U < outerGroups ||
                        taskOffset + 1U < outputs_) {
                        AscendC::SetFlag<
                            AscendC::HardEvent::V_MTE2>(
                            vToMte2Event_);
                        AscendC::WaitFlag<
                            AscendC::HardEvent::V_MTE2>(
                            vToMte2Event_);
                    }
                }
            }

            if constexpr (std::is_same<T, float>::value) {
                AscendC::Adds(
                    outputLocal,
                    accumulateLocal,
                    0.0f,
                    VECTOR_OUTPUTS);
            } else if constexpr (
                std::is_same<T, half>::value) {
                AscendC::Cast(
                    outputLocal,
                    accumulateLocal,
                    AscendC::RoundMode::CAST_NONE,
                    VECTOR_OUTPUTS);
            } else {
                AscendC::Cast(
                    outputLocal,
                    accumulateLocal,
                    AscendC::RoundMode::CAST_RINT,
                    VECTOR_OUTPUTS);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::DataCopyExtParams outputCopy;
            outputCopy.blockCount = 1;
            outputCopy.blockLen =
                VECTOR_OUTPUTS * sizeof(T);
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

    __aicore__ inline void ReduceLastAxisLongPartials(
        const AscendC::LocalTensor<float>& partialLocal,
        const AscendC::LocalTensor<float>& sourceLocal,
        const uint32_t activeOutputs,
        const uint32_t rowChunk,
        const uint32_t segmentsPerRow) const
    {
        constexpr uint32_t SEGMENT_ELEMENTS = 64U;
        if (activeOutputs == 1U &&
            segmentsPerRow == 256U) {
            AscendC::WholeReduceSum<float>(
                partialLocal,
                sourceLocal,
                static_cast<int32_t>(SEGMENT_ELEMENTS),
                255,
                1,
                1,
                SEGMENT_ELEMENTS / 8U);
            AscendC::WholeReduceSum<float>(
                partialLocal[255U],
                sourceLocal[255U * SEGMENT_ELEMENTS],
                static_cast<int32_t>(SEGMENT_ELEMENTS),
                1,
                1,
                1,
                SEGMENT_ELEMENTS / 8U);
            return;
        }
        const uint32_t firstGroupLimit =
            255U / segmentsPerRow;
        const uint32_t firstOutputs =
            activeOutputs < firstGroupLimit
                ? activeOutputs
                : firstGroupLimit;
        AscendC::WholeReduceSum<float>(
            partialLocal,
            sourceLocal,
            static_cast<int32_t>(SEGMENT_ELEMENTS),
            static_cast<int32_t>(
                firstOutputs * segmentsPerRow),
            1,
            1,
            SEGMENT_ELEMENTS / 8U);
        if (activeOutputs > firstOutputs) {
            AscendC::WholeReduceSum<float>(
                partialLocal[
                    firstOutputs * segmentsPerRow],
                sourceLocal[
                    firstOutputs * rowChunk],
                static_cast<int32_t>(SEGMENT_ELEMENTS),
                static_cast<int32_t>(
                    (activeOutputs - firstOutputs) *
                    segmentsPerRow),
                1,
                1,
                SEGMENT_ELEMENTS / 8U);
        }
    }

    __aicore__ inline void AccumulateLastAxisLongLarge(
        const AscendC::LocalTensor<float>& partialLocal,
        const AscendC::LocalTensor<float>& scratchLocal,
        const AscendC::LocalTensor<float>& accumulateLocal,
        const uint32_t segmentsPerRow,
        const bool initialize)
    {
        const uint32_t partialGroups =
            (segmentsPerRow + 63U) / 64U;
        AscendC::WholeReduceSum<float>(
            scratchLocal,
            partialLocal,
            64,
            static_cast<int32_t>(partialGroups),
            1,
            1,
            8);
        AscendC::WholeReduceSum<float>(
            partialLocal,
            scratchLocal,
            static_cast<int32_t>(partialGroups),
            1,
            1,
            1,
            1);
        if (initialize) {
            AscendC::Adds(
                accumulateLocal,
                partialLocal,
                0.0f,
                1);
        } else {
            AscendC::Add(
                accumulateLocal,
                accumulateLocal,
                partialLocal,
                1);
        }
    }

    __aicore__ inline void ProcessLastAxisVector8Long()
    {
        ProcessLastAxisVectorLong(8U, 2048U);
    }

    __aicore__ inline void ProcessLastAxisVector4Long()
    {
        ProcessLastAxisVectorLong(4U, 4096U);
    }

    __aicore__ inline void ProcessLastAxisVectorLong(
        const uint32_t vectorOutputs,
        const uint32_t rowChunk)
    {
        constexpr uint32_t SEGMENT_ELEMENTS = 64U;
        const uint32_t segmentsPerRow =
            rowChunk / SEGMENT_ELEMENTS;
        AscendC::LocalTensor<T> inputLocal =
            inputBuffer_.Get<T>();
        AscendC::LocalTensor<T> outputLocal =
            outputBuffer_.Get<T>();
        AscendC::LocalTensor<float> floatLocal =
            floatBuffer_.Get<float>();
        AscendC::LocalTensor<float> partialLocal =
            reduceWorkBuffer_.Get<float>();
        AscendC::LocalTensor<float> accumulateLocal =
            sumBuffer_.Get<float>();
        const uint32_t elementsPerBlock =
            32U / sizeof(T);
        const T zero = {};
        const bool directSingleChunk =
            vectorOutputs == 1U && reduceElements_ <= 4096U;

        for (uint64_t taskOffset = 0;
             taskOffset < outputs_;
             ++taskOffset) {
            const uint64_t taskIndex =
                firstOutput_ + taskOffset;
            const uint64_t outputStart =
                taskIndex * vectorOutputs;
            const uint32_t activeOutputs =
                static_cast<uint32_t>(
                    outputElements_ - outputStart < vectorOutputs
                        ? outputElements_ - outputStart
                        : vectorOutputs);
            const uint32_t inputCapacity =
                activeOutputs * rowChunk;
            for (uint64_t reduceOffset = 0;
                 reduceOffset < reduceElements_;
                 reduceOffset += rowChunk) {
                const uint32_t current =
                    static_cast<uint32_t>(
                        reduceElements_ - reduceOffset < rowChunk
                            ? reduceElements_ - reduceOffset
                            : rowChunk);
                const uint32_t alignedCurrent =
                    (current + elementsPerBlock - 1U) /
                    elementsPerBlock * elementsPerBlock;
                const uint32_t currentRowElements =
                    vectorOutputs == 1U
                        ? (current + SEGMENT_ELEMENTS - 1U) /
                            SEGMENT_ELEMENTS * SEGMENT_ELEMENTS
                        : rowChunk;
                const uint32_t currentSegmentsPerRow =
                    currentRowElements / SEGMENT_ELEMENTS;
                const uint32_t currentInputCount =
                    vectorOutputs == 1U
                        ? currentRowElements
                        : inputCapacity;

                if (currentRowElements != alignedCurrent) {
                    // Full chunks are completely overwritten by MTE2. Only
                    // the final partial chunk needs its unused row tails
                    // cleared before the fixed-width segmented reduction.
                    AscendC::Duplicate(
                        inputLocal,
                        zero,
                        currentInputCount);
                    AscendC::SetFlag<
                        AscendC::HardEvent::V_MTE2>(
                        vToMte2Event_);
                    AscendC::WaitFlag<
                        AscendC::HardEvent::V_MTE2>(
                        vToMte2Event_);
                }

                AscendC::DataCopyExtParams copyParams;
                copyParams.blockCount =
                    static_cast<uint16_t>(activeOutputs);
                copyParams.blockLen =
                    current * sizeof(T);
                copyParams.srcStride =
                    vectorOutputs == 1U
                        ? 0U
                        : static_cast<uint32_t>(
                            (reduceElements_ - current) * sizeof(T));
                copyParams.dstStride =
                    vectorOutputs == 1U
                        ? 0U
                        : static_cast<uint32_t>(
                            (rowChunk - alignedCurrent) * sizeof(T) /
                            32U);
                AscendC::DataCopyPadExtParams<T> padParams;
                padParams.isPad = alignedCurrent != current;
                padParams.leftPadding = 0;
                padParams.rightPadding =
                    static_cast<uint8_t>(
                        alignedCurrent - current);
                padParams.paddingValue = zero;
                AscendC::DataCopyPad(
                    inputLocal,
                    inputGm_[
                        outputStart * reduceElements_ +
                        reduceOffset],
                    copyParams,
                    padParams);
                AscendC::SetFlag<
                    AscendC::HardEvent::MTE2_V>(
                    mte2ToVEvent_);
                AscendC::WaitFlag<
                    AscendC::HardEvent::MTE2_V>(
                    mte2ToVEvent_);

                if constexpr (std::is_same<T, float>::value) {
                    AscendC::Mul(
                        inputLocal,
                        inputLocal,
                        inputLocal,
                        currentInputCount);
                    if (currentSegmentsPerRow > 64U &&
                        currentSegmentsPerRow % 64U != 0U) {
                        AscendC::Duplicate(
                            partialLocal,
                            0.0f,
                            (currentSegmentsPerRow + 63U) /
                                64U * 64U);
                    }
                    ReduceLastAxisLongPartials(
                        partialLocal,
                        inputLocal,
                        activeOutputs,
                        currentRowElements,
                        currentSegmentsPerRow);
                    if (currentSegmentsPerRow > 64U) {
                        AccumulateLastAxisLongLarge(
                            partialLocal,
                            floatLocal,
                            accumulateLocal,
                            currentSegmentsPerRow,
                            reduceOffset == 0U);
                    } else {
                        AscendC::WholeReduceSum<float>(
                            outputLocal,
                            partialLocal,
                            static_cast<int32_t>(
                                currentSegmentsPerRow),
                            static_cast<int32_t>(activeOutputs),
                            1,
                            1,
                            (currentSegmentsPerRow + 7U) / 8U);
                        if (!directSingleChunk) {
                            if (reduceOffset == 0U) {
                                AscendC::Adds(
                                    accumulateLocal,
                                    outputLocal,
                                    0.0f,
                                    activeOutputs);
                            } else {
                                AscendC::Add(
                                    accumulateLocal,
                                    accumulateLocal,
                                    outputLocal,
                                    activeOutputs);
                            }
                        }
                    }
                } else if constexpr (
                    std::is_same<T, half>::value) {
                    AscendC::Mul(
                        inputLocal,
                        inputLocal,
                        inputLocal,
                        currentInputCount);
                    AscendC::Cast(
                        floatLocal,
                        inputLocal,
                        AscendC::RoundMode::CAST_NONE,
                        currentInputCount);
                    if (currentSegmentsPerRow > 64U &&
                        currentSegmentsPerRow % 64U != 0U) {
                        AscendC::Duplicate(
                            partialLocal,
                            0.0f,
                            (currentSegmentsPerRow + 63U) /
                                64U * 64U);
                    }
                    ReduceLastAxisLongPartials(
                        partialLocal,
                        floatLocal,
                        activeOutputs,
                        currentRowElements,
                        currentSegmentsPerRow);
                    if (currentSegmentsPerRow > 64U) {
                        AccumulateLastAxisLongLarge(
                            partialLocal,
                            floatLocal,
                            accumulateLocal,
                            currentSegmentsPerRow,
                            reduceOffset == 0U);
                    } else {
                        AscendC::WholeReduceSum<float>(
                            floatLocal,
                            partialLocal,
                            static_cast<int32_t>(
                                currentSegmentsPerRow),
                            static_cast<int32_t>(activeOutputs),
                            1,
                            1,
                            (currentSegmentsPerRow + 7U) / 8U);
                        if (!directSingleChunk) {
                            if (reduceOffset == 0U) {
                                AscendC::Adds(
                                    accumulateLocal,
                                    floatLocal,
                                    0.0f,
                                    activeOutputs);
                            } else {
                                AscendC::Add(
                                    accumulateLocal,
                                    accumulateLocal,
                                    floatLocal,
                                    activeOutputs);
                            }
                        }
                    }
                } else {
                    AscendC::Cast(
                        floatLocal,
                        inputLocal,
                        AscendC::RoundMode::CAST_NONE,
                        currentInputCount);
                    AscendC::Mul(
                        floatLocal,
                        floatLocal,
                        floatLocal,
                        currentInputCount);
                    AscendC::Cast(
                        inputLocal,
                        floatLocal,
                        AscendC::RoundMode::CAST_RINT,
                        currentInputCount);
                    AscendC::Cast(
                        floatLocal,
                        inputLocal,
                        AscendC::RoundMode::CAST_NONE,
                        currentInputCount);
                    if (currentSegmentsPerRow > 64U &&
                        currentSegmentsPerRow % 64U != 0U) {
                        AscendC::Duplicate(
                            partialLocal,
                            0.0f,
                            (currentSegmentsPerRow + 63U) /
                                64U * 64U);
                    }
                    ReduceLastAxisLongPartials(
                        partialLocal,
                        floatLocal,
                        activeOutputs,
                        currentRowElements,
                        currentSegmentsPerRow);
                    if (currentSegmentsPerRow > 64U) {
                        AccumulateLastAxisLongLarge(
                            partialLocal,
                            floatLocal,
                            accumulateLocal,
                            currentSegmentsPerRow,
                            reduceOffset == 0U);
                    } else {
                        AscendC::WholeReduceSum<float>(
                            floatLocal,
                            partialLocal,
                            static_cast<int32_t>(
                                currentSegmentsPerRow),
                            static_cast<int32_t>(activeOutputs),
                            1,
                            1,
                            (currentSegmentsPerRow + 7U) / 8U);
                        if (!directSingleChunk) {
                            if (reduceOffset == 0U) {
                                AscendC::Adds(
                                    accumulateLocal,
                                    floatLocal,
                                    0.0f,
                                    activeOutputs);
                            } else {
                                AscendC::Add(
                                    accumulateLocal,
                                    accumulateLocal,
                                    floatLocal,
                                    activeOutputs);
                            }
                        }
                    }
                }
                if (reduceOffset + rowChunk < reduceElements_ ||
                    taskOffset + 1U < outputs_) {
                    AscendC::SetFlag<
                        AscendC::HardEvent::V_MTE2>(
                        vToMte2Event_);
                    AscendC::WaitFlag<
                        AscendC::HardEvent::V_MTE2>(
                        vToMte2Event_);
                }
            }

            if constexpr (std::is_same<T, float>::value) {
                if (!directSingleChunk) {
                    AscendC::Adds(
                        outputLocal,
                        accumulateLocal,
                        0.0f,
                        activeOutputs);
                }
            } else if constexpr (
                std::is_same<T, half>::value) {
                if (directSingleChunk) {
                    AscendC::Cast(
                        outputLocal,
                        floatLocal,
                        AscendC::RoundMode::CAST_NONE,
                        activeOutputs);
                } else {
                    AscendC::Cast(
                        outputLocal,
                        accumulateLocal,
                        AscendC::RoundMode::CAST_NONE,
                        activeOutputs);
                }
            } else {
                if (directSingleChunk) {
                    AscendC::Cast(
                        outputLocal,
                        floatLocal,
                        AscendC::RoundMode::CAST_RINT,
                        activeOutputs);
                } else {
                    AscendC::Cast(
                        outputLocal,
                        accumulateLocal,
                        AscendC::RoundMode::CAST_RINT,
                        activeOutputs);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::DataCopyExtParams outputCopy;
            outputCopy.blockCount = 1;
            outputCopy.blockLen =
                activeOutputs * sizeof(T);
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

    __aicore__ inline void ProcessLastAxisSegmentedRows()
    {
        AscendC::LocalTensor<T> inputLocal =
            inputBuffer_.Get<T>();
        AscendC::LocalTensor<T> outputLocal =
            outputBuffer_.Get<T>();
        AscendC::LocalTensor<float> floatLocal =
            floatBuffer_.Get<float>();
        AscendC::LocalTensor<float> partialLocal =
            reduceWorkBuffer_.Get<float>();

        uint32_t paddedReduce = 128U;
        if (reduceElements_ > 512U) {
            paddedReduce = static_cast<uint32_t>(
                (reduceElements_ + 511U) / 512U * 512U);
        } else if (reduceElements_ > 256U) {
            paddedReduce = 512U;
        } else if (reduceElements_ > 128U) {
            paddedReduce = 256U;
        }
        const uint32_t segmentElements =
            paddedReduce <= 256U
                ? paddedReduce / 8U
                : 64U;
        const uint32_t segmentsPerRow =
            paddedReduce / segmentElements;
        uint32_t tileRows = CHUNK / paddedReduce;
        if (tileRows > 31U) {
            tileRows = 31U;
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
            const uint32_t elementsPerBlock =
                32U / sizeof(T);
            const uint32_t alignedReduce =
                static_cast<uint32_t>(
                    (reduceElements_ + elementsPerBlock - 1U) /
                    elementsPerBlock * elementsPerBlock);
            const T zero = {};

            if (paddedReduce != alignedReduce) {
                // When the copied rows already fill the segmented row
                // stride, MTE2 overwrites the whole input tile and no
                // pre-clear is required.
                AscendC::Duplicate(
                    inputLocal,
                    zero,
                    inputCount);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                    vToMte2Event_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                    vToMte2Event_);
            }

            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount =
                static_cast<uint16_t>(currentRows);
            copyParams.blockLen =
                static_cast<uint32_t>(
                    reduceElements_ * sizeof(T));
            copyParams.srcStride = 0;
            copyParams.dstStride =
                static_cast<uint32_t>(
                    (paddedReduce - alignedReduce) *
                    sizeof(T) / 32U);
            AscendC::DataCopyPadExtParams<T> padParams;
            padParams.isPad =
                alignedReduce != reduceElements_;
            padParams.leftPadding = 0;
            padParams.rightPadding =
                static_cast<uint8_t>(
                    alignedReduce - reduceElements_);
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
                    partialLocal,
                    inputLocal,
                    static_cast<int32_t>(segmentElements),
                    static_cast<int32_t>(
                        currentRows * segmentsPerRow),
                    1,
                    1,
                    static_cast<int32_t>(
                        segmentElements / 8U));
                AscendC::WholeReduceSum<float>(
                    outputLocal,
                    partialLocal,
                    static_cast<int32_t>(segmentsPerRow),
                    static_cast<int32_t>(currentRows),
                    1,
                    1,
                    static_cast<int32_t>(
                        segmentsPerRow / 8U));
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
                    partialLocal,
                    floatLocal,
                    static_cast<int32_t>(segmentElements),
                    static_cast<int32_t>(
                        currentRows * segmentsPerRow),
                    1,
                    1,
                    static_cast<int32_t>(
                        segmentElements / 8U));
                AscendC::WholeReduceSum<float>(
                    floatLocal,
                    partialLocal,
                    static_cast<int32_t>(segmentsPerRow),
                    static_cast<int32_t>(currentRows),
                    1,
                    1,
                    static_cast<int32_t>(
                        segmentsPerRow / 8U));
                AscendC::Cast(
                    outputLocal,
                    floatLocal,
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
                    partialLocal,
                    floatLocal,
                    static_cast<int32_t>(segmentElements),
                    static_cast<int32_t>(
                        currentRows * segmentsPerRow),
                    1,
                    1,
                    static_cast<int32_t>(
                        segmentElements / 8U));
                AscendC::WholeReduceSum<float>(
                    floatLocal,
                    partialLocal,
                    static_cast<int32_t>(segmentsPerRow),
                    static_cast<int32_t>(currentRows),
                    1,
                    1,
                    static_cast<int32_t>(
                        segmentsPerRow / 8U));
                AscendC::Cast(
                    outputLocal,
                    floatLocal,
                    AscendC::RoundMode::CAST_RINT,
                    static_cast<int32_t>(currentRows));
            }

            if (paddedReduce == alignedReduce &&
                rowOffset + currentRows < outputs_) {
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

    __aicore__ inline void ReduceRowsInPlace(
        AscendC::LocalTensor<float> values,
        const uint32_t rows,
        const uint32_t paddedInner)
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
    }

    __aicore__ inline void ReduceRowsInto(
        AscendC::LocalTensor<float> values,
        AscendC::LocalTensor<float> accumulate,
        const uint32_t rows,
        const uint32_t paddedInner,
        const uint32_t floatPadded,
        const bool initialize)
    {
        ReduceRowsInPlace(values, rows, paddedInner);
        if (initialize) {
            AscendC::Adds(
                accumulate,
                values,
                0.0f,
                floatPadded);
        } else {
            AscendC::Add(
                accumulate,
                accumulate,
                values,
                floatPadded);
        }
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
            reduceRowsPerTile =
                HighestPowerOfTwo(reduceRowsPerTile);
            uint64_t reduceOffset = 0;
            while (reduceOffset < reduceElements_) {
                const uint32_t remainingRows =
                    static_cast<uint32_t>(
                        reduceElements_ - reduceOffset <
                                reduceRowsPerTile
                            ? reduceElements_ - reduceOffset
                            : reduceRowsPerTile);
                const uint32_t currentReduceRows =
                    remainingRows;
                uint32_t reductionRows = 1U;
                while (reductionRows < currentReduceRows) {
                    reductionRows <<= 1U;
                }
                const uint32_t inputCount =
                    currentReduceRows * paddedInner;
                const uint32_t paddedRowElements =
                    (reductionRows - currentReduceRows) *
                    paddedInner;
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
                        floatPadded,
                        reduceOffset == 0U);
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
                        floatPadded,
                        reduceOffset == 0U);
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
                        floatPadded,
                        reduceOffset == 0U);
                }
                if (reduceOffset + currentReduceRows <
                        reduceElements_ ||
                    processed + current < outputs_) {
                    AscendC::SetFlag<
                        AscendC::HardEvent::V_MTE2>(
                        vToMte2Event_);
                    AscendC::WaitFlag<
                        AscendC::HardEvent::V_MTE2>(
                        vToMte2Event_);
                }
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
                if (reduceIndex + 1U < reduceElements_) {
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
            false,
            false,
            false,
            8U> op;
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
            false,
            false,
            false,
            8U> op;
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
            true,
            false,
            false,
            8U> op;
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
            true,
            false,
            false,
            8U> op;
        op.Init(input, output, userWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
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
            false,
            true,
            false,
            8U> op;
        op.Init(input, output, userWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(6)) {
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
            false,
            false,
            true,
            8U> op;
        op.Init(input, output, userWorkspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(7)) {
        GET_TILING_DATA(tilingData, tiling);
        if (tilingData.outputElements == 0) {
            return;
        }
        KernelSquareSumV1<
            DTYPE_INPUT,
            LONG_CHUNK,
            false,
            true,
            false,
            8U> op;
        op.Init(input, output, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(8)) {
        GET_TILING_DATA(tilingData, tiling);
        if (tilingData.outputElements == 0) {
            return;
        }
        KernelSquareSumV1<
            DTYPE_INPUT,
            LONG_CHUNK,
            false,
            false,
            true,
            8U> op;
        op.Init(input, output, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(9)) {
        GET_TILING_DATA(tilingData, tiling);
        if (tilingData.outputElements == 0) {
            return;
        }
        KernelSquareSumV1<
            DTYPE_INPUT,
            NORMAL_CHUNK,
            false,
            true,
            false,
            4U> op;
        op.Init(input, output, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(10)) {
        GET_TILING_DATA(tilingData, tiling);
        if (tilingData.outputElements == 0) {
            return;
        }
        KernelSquareSumV1<
            DTYPE_INPUT,
            NORMAL_CHUNK,
            false,
            true,
            false,
            2U> op;
        op.Init(input, output, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(11)) {
        GET_TILING_DATA(tilingData, tiling);
        if (tilingData.outputElements == 0) {
            return;
        }
        KernelSquareSumV1<
            DTYPE_INPUT,
            NORMAL_CHUNK,
            false,
            true,
            false,
            1U> op;
        op.Init(input, output, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(12)) {
        GET_TILING_DATA(tilingData, tiling);
        if (tilingData.outputElements == 0) {
            return;
        }
        KernelSquareSumV1<
            DTYPE_INPUT,
            LONG_CHUNK,
            false,
            true,
            false,
            1U> op;
        op.Init(input, output, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(13)) {
        GET_TILING_DATA(tilingData, tiling);
        if (tilingData.outputElements == 0) {
            return;
        }
        KernelSquareSumV1<
            DTYPE_INPUT,
            NORMAL_CHUNK,
            false,
            false,
            false,
            8U,
            true> op;
        op.Init(input, output, workspace, tilingData);
        op.Process();
    }
}
