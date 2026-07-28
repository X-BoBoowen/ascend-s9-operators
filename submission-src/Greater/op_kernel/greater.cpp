#include "kernel_operator.h"

constexpr uint32_t TILE_ELEMENTS = 4096;
constexpr uint32_t COMPARE_ALIGNMENT_ELEMENTS = 128;
constexpr uint32_t MAX_RANK = 8;

template <typename T>
class KernelGreater {
public:
    __aicore__ inline KernelGreater() {}

    __aicore__ inline void Init(
        GM_ADDR self,
        GM_ADDR other,
        GM_ADDR output,
        const GreaterTilingData& tiling,
        uint32_t blockIdx)
    {
        rank_ = tiling.rank;
        outputElements_ = tiling.outputElements;
        selfContiguous_ = tiling.selfContiguous != 0;
        otherContiguous_ = tiling.otherContiguous != 0;
        for (uint32_t axis = 0; axis < MAX_RANK; ++axis) {
            outputDims_[axis] = tiling.outputDims[axis];
            selfStrides_[axis] = tiling.selfStrides[axis];
            otherStrides_[axis] = tiling.otherStrides[axis];
        }
        selfScalar_ = false;
        otherScalar_ = false;
        selfRunElements_ = 1;
        otherRunElements_ = 1;
        selfConstantRunElements_ = 1;
        otherConstantRunElements_ = 1;
        if (!selfContiguous_) {
            selfScalar_ = true;
            for (uint32_t axis = 0; axis < rank_; ++axis) {
                selfScalar_ =
                    selfScalar_ && selfStrides_[axis] == 0;
            }
            uint64_t expectedStride = 1;
            for (int32_t axis = static_cast<int32_t>(rank_) - 1;
                 axis >= 0;
                 --axis) {
                if (selfStrides_[axis] != expectedStride) {
                    break;
                }
                selfRunElements_ *= outputDims_[axis];
                expectedStride *= outputDims_[axis];
            }
            for (int32_t axis = static_cast<int32_t>(rank_) - 1;
                 axis >= 0;
                 --axis) {
                if (selfStrides_[axis] != 0) {
                    break;
                }
                selfConstantRunElements_ *= outputDims_[axis];
            }
        }
        if (!otherContiguous_) {
            otherScalar_ = true;
            for (uint32_t axis = 0; axis < rank_; ++axis) {
                otherScalar_ =
                    otherScalar_ && otherStrides_[axis] == 0;
            }
            uint64_t expectedStride = 1;
            for (int32_t axis = static_cast<int32_t>(rank_) - 1;
                 axis >= 0;
                 --axis) {
                if (otherStrides_[axis] != expectedStride) {
                    break;
                }
                otherRunElements_ *= outputDims_[axis];
                expectedStride *= outputDims_[axis];
            }
            for (int32_t axis = static_cast<int32_t>(rank_) - 1;
                 axis >= 0;
                 --axis) {
                if (otherStrides_[axis] != 0) {
                    break;
                }
                otherConstantRunElements_ *= outputDims_[axis];
            }
        }

        const uint64_t extraElements =
            blockIdx < tiling.extraBlocks
                ? tiling.partitionUnitElements
                : 0;
        elements_ = tiling.baseElementsPerBlock + extraElements;
        firstElement_ =
            blockIdx * tiling.baseElementsPerBlock +
            (blockIdx < tiling.extraBlocks
                 ? blockIdx * tiling.partitionUnitElements
                 : tiling.extraBlocks *
                     tiling.partitionUnitElements);

        selfGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(self),
            outputElements_);
        otherGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(other),
            outputElements_);
        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(output),
            outputElements_);

        pipe_.InitBuffer(
            selfBuffer_,
            TILE_ELEMENTS * sizeof(T));
        pipe_.InitBuffer(
            otherBuffer_,
            TILE_ELEMENTS * sizeof(T));
        pipe_.InitBuffer(
            floatSelfBuffer_,
            TILE_ELEMENTS * sizeof(float));
        pipe_.InitBuffer(
            floatOtherBuffer_,
            TILE_ELEMENTS * sizeof(float));
        pipe_.InitBuffer(
            halfSelfBuffer_,
            TILE_ELEMENTS * sizeof(half));
        pipe_.InitBuffer(
            halfOtherBuffer_,
            TILE_ELEMENTS * sizeof(half));
        pipe_.InitBuffer(
            maskBuffer_,
            TILE_ELEMENTS / 8);
        pipe_.InitBuffer(
            onesBuffer_,
            TILE_ELEMENTS * sizeof(half));
        pipe_.InitBuffer(
            selectedBuffer_,
            TILE_ELEMENTS * sizeof(half));
        pipe_.InitBuffer(
            outputBuffer_,
            TILE_ELEMENTS * sizeof(uint8_t));

        const uint32_t onesElements = static_cast<uint32_t>(
            elements_ < TILE_ELEMENTS
                ? (elements_ + COMPARE_ALIGNMENT_ELEMENTS - 1) /
                    COMPARE_ALIGNMENT_ELEMENTS *
                    COMPARE_ALIGNMENT_ELEMENTS
                : TILE_ELEMENTS);
        AscendC::Duplicate(
            onesBuffer_.Get<half>(),
            static_cast<half>(1),
            onesElements);
        AscendC::PipeBarrier<PIPE_V>();

        mte2ToVEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE2_V));
        vToMte2Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::V_MTE2));
        sToVEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::S_V));
        vToSEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::V_S));
        vToMte3Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::V_MTE3));
        mte3ToVEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE3_V));
    }

    __aicore__ inline void Process()
    {
        uint32_t tileElements = TILE_ELEMENTS;
        for (uint64_t offset = 0;
             offset < elements_;
             offset += tileElements) {
            const uint32_t current = static_cast<uint32_t>(
                elements_ - offset < tileElements
                    ? elements_ - offset
                    : tileElements);
            ProcessTile(firstElement_ + offset, current);
        }
    }

private:
    __aicore__ inline uint64_t BroadcastOffset(
        uint64_t outputIndex,
        const uint64_t* strides) const
    {
        uint64_t inputOffset = 0;
        for (int32_t axis = static_cast<int32_t>(rank_) - 1;
             axis >= 0;
             --axis) {
            const uint64_t coordinate =
                outputIndex % outputDims_[axis];
            outputIndex /= outputDims_[axis];
            inputOffset += coordinate * strides[axis];
        }
        return inputOffset;
    }

    __aicore__ inline uint32_t LoadInput(
        const AscendC::GlobalTensor<T>& source,
        AscendC::LocalTensor<T>& local,
        uint64_t outputStart,
        uint32_t current,
        uint32_t aligned,
        bool contiguous,
        bool scalar,
        uint64_t runElements,
        uint64_t constantRunElements,
        const uint64_t* strides)
    {
        if (contiguous) {
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = current * sizeof(T);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<T> padParams;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = static_cast<T>(0);
            AscendC::DataCopyPad(
                local,
                source[outputStart],
                copyParams,
                padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
                mte2ToVEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                mte2ToVEvent_);
            return 1;
        }

        if (scalar) {
            const T value = source.GetValue(0);
            if constexpr (AscendC::IsSameType<T, int8_t>::value) {
                AscendC::SetFlag<AscendC::HardEvent::S_V>(
                    sToVEvent_);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(
                    sToVEvent_);
                const uint16_t byteValue =
                    static_cast<uint8_t>(value);
                const uint16_t packedValue =
                    byteValue | (byteValue << 8);
                AscendC::Duplicate(
                    local.template ReinterpretCast<uint16_t>(),
                    packedValue,
                    aligned / 2);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::S_V>(
                    sToVEvent_);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(
                    sToVEvent_);
                AscendC::Duplicate(local, value, aligned);
            }
            return 0;
        }

        if (constantRunElements > 1 &&
            outputStart % constantRunElements == 0 &&
            current % constantRunElements == 0 &&
            (AscendC::IsSameType<T, int8_t>::value ||
             constantRunElements * sizeof(T) % 32 == 0)) {
            uint64_t outputIndex = outputStart;
            uint32_t localOffset = 0;
            while (localOffset < current) {
                const T value = source.GetValue(
                    BroadcastOffset(outputIndex, strides));
                if constexpr (
                    AscendC::IsSameType<T, int8_t>::value) {
                    for (uint32_t index = 0;
                         index < constantRunElements;
                         ++index) {
                        local.SetValue(localOffset + index, value);
                    }
                } else {
                    AscendC::SetFlag<AscendC::HardEvent::S_V>(
                        sToVEvent_);
                    AscendC::WaitFlag<AscendC::HardEvent::S_V>(
                        sToVEvent_);
                    AscendC::Duplicate(
                        local[localOffset],
                        value,
                        constantRunElements);
                }
                outputIndex += constantRunElements;
                localOffset += constantRunElements;
            }
            if constexpr (
                AscendC::IsSameType<T, int8_t>::value) {
                AscendC::SetFlag<AscendC::HardEvent::S_V>(
                    sToVEvent_);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(
                    sToVEvent_);
            }
            if constexpr (AscendC::IsSameType<T, int8_t>::value) {
                return 2;
            }
            return 0;
        }

        if (runElements > 1 &&
            outputStart % runElements == 0 &&
            current % runElements == 0 &&
            runElements * sizeof(T) % 32 == 0) {
            uint64_t outputIndex = outputStart;
            uint32_t localOffset = 0;
            while (localOffset < current) {
                const uint64_t positionInRun =
                    outputIndex % runElements;
                const uint64_t remainingInRun =
                    runElements - positionInRun;
                const uint32_t segment = static_cast<uint32_t>(
                    remainingInRun < current - localOffset
                        ? remainingInRun
                        : current - localOffset);
                AscendC::DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = segment * sizeof(T);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                AscendC::DataCopyPadExtParams<T> padParams;
                padParams.isPad = true;
                padParams.leftPadding = 0;
                padParams.rightPadding = 0;
                padParams.paddingValue = static_cast<T>(0);
                AscendC::DataCopyPad(
                    local[localOffset],
                    source[BroadcastOffset(outputIndex, strides)],
                    copyParams,
                    padParams);
                outputIndex += segment;
                localOffset += segment;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
                mte2ToVEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                mte2ToVEvent_);
            return 1;
        }

        uint64_t coordinates[MAX_RANK] = {};
        uint64_t remaining = outputStart;
        uint64_t sourceOffset = 0;
        for (int32_t axis = static_cast<int32_t>(rank_) - 1;
             axis >= 0;
             --axis) {
            coordinates[axis] = remaining % outputDims_[axis];
            remaining /= outputDims_[axis];
            sourceOffset += coordinates[axis] * strides[axis];
        }
        for (uint32_t index = 0; index < current; ++index) {
            local.SetValue(index, source.GetValue(sourceOffset));
            if (index + 1 == current) {
                break;
            }
            for (int32_t axis = static_cast<int32_t>(rank_) - 1;
                 axis >= 0;
                 --axis) {
                ++coordinates[axis];
                sourceOffset += strides[axis];
                if (coordinates[axis] < outputDims_[axis]) {
                    break;
                }
                coordinates[axis] = 0;
                sourceOffset -= outputDims_[axis] * strides[axis];
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(sToVEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(sToVEvent_);
        return 2;
    }

    __aicore__ inline void MaskToOutput(
        const AscendC::LocalTensor<uint8_t>& mask,
        uint32_t aligned,
        AscendC::LocalTensor<uint8_t>& outputLocal)
    {
        AscendC::LocalTensor<half> ones =
            onesBuffer_.Get<half>();
        AscendC::LocalTensor<half> selected =
            selectedBuffer_.Get<half>();

        AscendC::Select(
            selected,
            mask,
            ones,
            static_cast<half>(0),
            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
            aligned);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(
            outputLocal,
            selected,
            AscendC::RoundMode::CAST_RINT,
            aligned);
    }

    template <typename CompareT>
    __aicore__ inline void VectorCompare(
        const AscendC::LocalTensor<CompareT>& selfLocal,
        const AscendC::LocalTensor<CompareT>& otherLocal,
        uint32_t aligned,
        AscendC::LocalTensor<uint8_t>& outputLocal)
    {
        AscendC::LocalTensor<uint8_t> mask =
            maskBuffer_.Get<uint8_t>();
        AscendC::Compare(
            mask,
            selfLocal,
            otherLocal,
            AscendC::CMPMODE::GT,
            aligned);
        AscendC::PipeBarrier<PIPE_V>();
        MaskToOutput(mask, aligned, outputLocal);
    }

    __aicore__ inline void VectorCompareInt32(
        const AscendC::LocalTensor<int32_t>& selfLocal,
        const AscendC::LocalTensor<int32_t>& otherLocal,
        uint32_t aligned,
        AscendC::LocalTensor<uint8_t>& outputLocal)
    {
        AscendC::LocalTensor<int32_t> maximum =
            floatSelfBuffer_.Get<int32_t>();
        AscendC::LocalTensor<half> greaterOrEqual =
            halfSelfBuffer_.Get<half>();
        AscendC::LocalTensor<half> equal =
            halfOtherBuffer_.Get<half>();
        AscendC::LocalTensor<half> ones =
            onesBuffer_.Get<half>();
        AscendC::LocalTensor<half> selected =
            selectedBuffer_.Get<half>();
        AscendC::LocalTensor<uint8_t> greaterOrEqualMask =
            maskBuffer_.Get<uint8_t>();
        AscendC::LocalTensor<uint8_t> equalMask =
            floatOtherBuffer_.Get<uint8_t>();

        AscendC::Max(
            maximum,
            selfLocal,
            otherLocal,
            static_cast<int32_t>(aligned));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(
            greaterOrEqualMask,
            maximum,
            selfLocal,
            AscendC::CMPMODE::EQ,
            aligned);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(
            equalMask,
            selfLocal,
            otherLocal,
            AscendC::CMPMODE::EQ,
            aligned);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(
            greaterOrEqual,
            greaterOrEqualMask,
            ones,
            static_cast<half>(0),
            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
            aligned);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(
            equal,
            equalMask,
            ones,
            static_cast<half>(0),
            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
            aligned);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(
            selected,
            greaterOrEqual,
            equal,
            aligned);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(
            outputLocal,
            selected,
            AscendC::RoundMode::CAST_RINT,
            aligned);
    }

    __aicore__ inline void ProcessTile(
        uint64_t outputStart,
        uint32_t current)
    {
        const uint32_t aligned =
            (current + COMPARE_ALIGNMENT_ELEMENTS - 1) /
            COMPARE_ALIGNMENT_ELEMENTS *
            COMPARE_ALIGNMENT_ELEMENTS;
        AscendC::LocalTensor<T> selfLocal =
            selfBuffer_.Get<T>();
        AscendC::LocalTensor<T> otherLocal =
            otherBuffer_.Get<T>();
        AscendC::LocalTensor<uint8_t> outputLocal =
            outputBuffer_.Get<uint8_t>();

        const uint32_t selfLoadPipeline = LoadInput(
            selfGm_,
            selfLocal,
            outputStart,
            current,
            aligned,
            selfContiguous_,
            selfScalar_,
            selfRunElements_,
            selfConstantRunElements_,
            selfStrides_);
        const uint32_t otherLoadPipeline = LoadInput(
            otherGm_,
            otherLocal,
            outputStart,
            current,
            aligned,
            otherContiguous_,
            otherScalar_,
            otherRunElements_,
            otherConstantRunElements_,
            otherStrides_);

        if constexpr (AscendC::IsSameType<T, int32_t>::value) {
            VectorCompareInt32(
                selfLocal,
                otherLocal,
                aligned,
                outputLocal);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
        } else if constexpr (
            AscendC::IsSameType<T, float>::value) {
            VectorCompare(
                selfLocal,
                otherLocal,
                aligned,
                outputLocal);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
        } else if constexpr (
            AscendC::IsSameType<T, half>::value) {
            VectorCompare(
                selfLocal,
                otherLocal,
                aligned,
                outputLocal);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
        } else if constexpr (
            AscendC::IsSameType<T, int8_t>::value) {
            AscendC::LocalTensor<half> selfHalf =
                halfSelfBuffer_.Get<half>();
            AscendC::LocalTensor<half> otherHalf =
                halfOtherBuffer_.Get<half>();
            AscendC::Cast(
                selfHalf,
                selfLocal,
                AscendC::RoundMode::CAST_NONE,
                aligned);
            AscendC::Cast(
                otherHalf,
                otherLocal,
                AscendC::RoundMode::CAST_NONE,
                aligned);
            AscendC::PipeBarrier<PIPE_V>();
            VectorCompare(
                selfHalf,
                otherHalf,
                aligned,
                outputLocal);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
        } else {
            AscendC::LocalTensor<float> selfFloat =
                floatSelfBuffer_.Get<float>();
            AscendC::LocalTensor<float> otherFloat =
                floatOtherBuffer_.Get<float>();
            AscendC::Cast(
                selfFloat,
                selfLocal,
                AscendC::RoundMode::CAST_NONE,
                aligned);
            AscendC::Cast(
                otherFloat,
                otherLocal,
                AscendC::RoundMode::CAST_NONE,
                aligned);
            AscendC::PipeBarrier<PIPE_V>();
            VectorCompare(
                selfFloat,
                otherFloat,
                aligned,
                outputLocal);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
        }

        const uint32_t loadPipelines =
            selfLoadPipeline | otherLoadPipeline;
        if ((loadPipelines & 1U) != 0U) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                vToMte2Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                vToMte2Event_);
        }
        if ((loadPipelines & 2U) != 0U) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(
                vToSEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(
                vToSEvent_);
        }

        AscendC::DataCopyExtParams outputCopyParams;
        outputCopyParams.blockCount = 1;
        outputCopyParams.blockLen = current;
        outputCopyParams.srcStride = 0;
        outputCopyParams.dstStride = 0;
        AscendC::DataCopyPad(
            outputGm_[outputStart],
            outputLocal,
            outputCopyParams);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
            mte3ToVEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
            mte3ToVEvent_);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selfBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> otherBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatSelfBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatOtherBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> halfSelfBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> halfOtherBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> onesBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selectedBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuffer_;
    AscendC::GlobalTensor<T> selfGm_;
    AscendC::GlobalTensor<T> otherGm_;
    AscendC::GlobalTensor<uint8_t> outputGm_;
    uint64_t outputElements_;
    uint64_t elements_;
    uint64_t firstElement_;
    uint32_t rank_;
    bool selfContiguous_;
    bool otherContiguous_;
    bool selfScalar_;
    bool otherScalar_;
    uint64_t selfRunElements_;
    uint64_t otherRunElements_;
    uint64_t selfConstantRunElements_;
    uint64_t otherConstantRunElements_;
    uint64_t outputDims_[MAX_RANK];
    uint64_t selfStrides_[MAX_RANK];
    uint64_t otherStrides_[MAX_RANK];
    event_t mte2ToVEvent_;
    event_t vToMte2Event_;
    event_t sToVEvent_;
    event_t vToSEvent_;
    event_t vToMte3Event_;
    event_t mte3ToVEvent_;
};

extern "C" __global__ __aicore__ void greater(
    GM_ADDR self,
    GM_ADDR other,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.outputElements == 0) {
        return;
    }
    KernelGreater<DTYPE_SELF> op;
    op.Init(
        self,
        other,
        output,
        tilingData,
        AscendC::GetBlockIdx());
    op.Process();
}
