#include "kernel_operator.h"

constexpr uint32_t TILE_OUTPUTS = 1024;
constexpr uint32_t MAX_RANK = 5;
constexpr uint32_t FAST_CHUNK = 8192;

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

template <typename T>
class KernelSquareSumV1 {
public:
    __aicore__ inline KernelSquareSumV1() {}

    __aicore__ inline void Init(
        GM_ADDR input,
        GM_ADDR output,
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
        pipe_.InitBuffer(
            outputBuffer_,
            TILE_OUTPUTS * sizeof(T));
        pipe_.InitBuffer(
            inputBuffer_,
            FAST_CHUNK * sizeof(T));
        pipe_.InitBuffer(
            floatBuffer_,
            FAST_CHUNK * sizeof(float));
        pipe_.InitBuffer(
            reduceWorkBuffer_,
            FAST_CHUNK * sizeof(float));
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
        sToMte3Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::S_MTE3));
        mte3ToSEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE3_S));
    }

    __aicore__ inline void Process()
    {
        if (fastPath_ == 1) {
            const uint32_t elementsPerBlock =
                32U / sizeof(T);
            const uint64_t paddedReduce =
                (reduceElements_ + elementsPerBlock - 1U) /
                elementsPerBlock * elementsPerBlock;
            if (reduceElements_ <= 64U &&
                paddedReduce <= FAST_CHUNK) {
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
                FAST_CHUNK / paddedInner;
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
            paddedTail > FAST_CHUNK ||
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
            FAST_CHUNK / paddedTail;
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
                : FAST_CHUNK / paddedReduce;
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
                FAST_CHUNK / paddedInner;
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
            for (uint64_t reduceOffset = 0;
                 reduceOffset < reduceElements_;
                 reduceOffset += reduceRowsPerTile) {
                const uint32_t currentReduceRows =
                    static_cast<uint32_t>(
                        reduceElements_ - reduceOffset <
                                reduceRowsPerTile
                            ? reduceElements_ - reduceOffset
                            : reduceRowsPerTile);
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
                    for (uint32_t row = 0;
                         row < currentReduceRows;
                         ++row) {
                        AscendC::Add(
                            accumulateLocal,
                            accumulateLocal,
                            inputLocal[row * paddedInner],
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
                            valueLocal[row * paddedInner],
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
                    for (uint32_t row = 0;
                         row < currentReduceRows;
                         ++row) {
                        AscendC::Add(
                            accumulateLocal,
                            accumulateLocal,
                            valueLocal[row * paddedInner],
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
                        floatPadded);
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
             offset += FAST_CHUNK) {
            const uint32_t current = static_cast<uint32_t>(
                elementCount - offset < FAST_CHUNK
                    ? elementCount - offset
                    : FAST_CHUNK);
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
};

extern "C" __global__ __aicore__ void square_sum_v1(
    GM_ADDR input,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.outputElements == 0) {
        return;
    }
    KernelSquareSumV1<DTYPE_INPUT> op;
    op.Init(input, output, tilingData);
    op.Process();
}
