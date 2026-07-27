#include "kernel_operator.h"

constexpr uint32_t INNER_CHUNK = 256;
constexpr uint32_t MAX_DIM_GROUP = 64;
constexpr uint32_t MAX_INDEX_COUNT = 8000;
constexpr uint32_t ACCUM_ELEMENTS =
    INNER_CHUNK * MAX_DIM_GROUP;

template <typename T, bool USE_BATCHED_INT8>
class KernelIndexAdd {
public:
    __aicore__ inline KernelIndexAdd() {}

    __aicore__ inline void Init(
        GM_ADDR self,
        GM_ADDR index,
        GM_ADDR source,
        GM_ADDR output,
        const IndexAddTilingData& tiling)
    {
        totalElements_ = tiling.totalElements;
        sourceElements_ = tiling.sourceElements;
        dimSize_ = tiling.dimSize;
        inner_ = tiling.inner;
        indexCount_ = tiling.indexCount;
        taskCount_ = tiling.taskCount;
        dimGroup_ = tiling.dimGroup;
        dimGroups_ = tiling.dimGroups;
        innerChunks_ = tiling.innerChunks;
        blockIdx_ = AscendC::GetBlockIdx();
        blockNum_ = AscendC::GetBlockNum();

        selfGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(self),
            totalElements_);
        indexGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(index),
            indexCount_);
        sourceGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(source),
            sourceElements_);
        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(output),
            totalElements_);

        constexpr bool isBatchedInt8 =
            AscendC::IsSameType<T, int8_t>::value &&
            USE_BATCHED_INT8;
        constexpr uint32_t accumElementBytes =
            isBatchedInt8
                ? sizeof(half)
                : sizeof(T);
        pipe_.InitBuffer(
            accumBuffer_,
            ACCUM_ELEMENTS * accumElementBytes);
        pipe_.InitBuffer(
            sourceBuffer_,
            INNER_CHUNK * sizeof(T));
        pipe_.InitBuffer(
            floatAccumBuffer_,
            INNER_CHUNK * sizeof(float));
        pipe_.InitBuffer(
            floatSourceBuffer_,
            INNER_CHUNK * sizeof(float));
        pipe_.InitBuffer(
            indexBuffer_,
            MAX_INDEX_COUNT * sizeof(int32_t));
        pipe_.InitBuffer(
            int8MaskBuffer_,
            INNER_CHUNK / 8);
        pipe_.InitBuffer(
            int8WorkBuffer_,
            INNER_CHUNK * sizeof(half));
        if constexpr (isBatchedInt8) {
            pipe_.InitBuffer(
                int8OutputBuffer_,
                ACCUM_ELEMENTS * sizeof(int8_t));
        }
        mte2ToVEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE2_V));
        mte2ToSEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE2_S));
        vToMte2Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::V_MTE2));
        sToMte2Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::S_MTE2));
        vToMte3Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::V_MTE3));
        sToMte3Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::S_MTE3));
        mte3ToMte2Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<int32_t> indexLocal =
            indexBuffer_.Get<int32_t>();
        if (indexCount_ != 0) {
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen =
                indexCount_ * sizeof(int32_t);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<int32_t> padding;
            padding.isPad = false;
            padding.leftPadding = 0;
            padding.rightPadding = 0;
            padding.paddingValue = 0;
            AscendC::DataCopyPad(
                indexLocal,
                indexGm_,
                copyParams,
                padding);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(
                mte2ToSEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(
                mte2ToSEvent_);
        }
        for (uint64_t task = blockIdx_;
             task < taskCount_;
             task += blockNum_) {
            ProcessTask(task);
        }
    }

private:
    __aicore__ inline void CopyIn(
        AscendC::LocalTensor<T>& local,
        uint32_t localOffset,
        const AscendC::GlobalTensor<T>& source,
        uint64_t sourceOffset,
        uint32_t count)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = count * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = static_cast<T>(0);
        AscendC::DataCopyPad(
            local[localOffset],
            source[sourceOffset],
            copyParams,
            padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
            mte2ToVEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            mte2ToVEvent_);
    }

    __aicore__ inline void WrapInt8(
        AscendC::LocalTensor<half>& values,
        uint32_t alignedCount)
    {
        AscendC::LocalTensor<half> scratch =
            floatSourceBuffer_.Get<half>();
        AscendC::LocalTensor<half> adjusted =
            int8WorkBuffer_.Get<half>();
        AscendC::LocalTensor<uint8_t> mask =
            int8MaskBuffer_.Get<uint8_t>();
        AscendC::Adds(
            adjusted,
            values,
            static_cast<half>(-256),
            alignedCount);
        AscendC::CompareScalar(
            mask,
            values,
            static_cast<half>(127),
            AscendC::CMPMODE::GT,
            alignedCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(
            scratch,
            mask,
            adjusted,
            values,
            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
            alignedCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(
            adjusted,
            scratch,
            static_cast<half>(256),
            alignedCount);
        AscendC::CompareScalar(
            mask,
            scratch,
            static_cast<half>(-128),
            AscendC::CMPMODE::LT,
            alignedCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(
            values,
            mask,
            adjusted,
            scratch,
            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
            alignedCount);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AddValues(
        AscendC::LocalTensor<T>& accum,
        uint32_t accumOffset,
        const AscendC::LocalTensor<T>& source,
        uint32_t count)
    {
        if constexpr (AscendC::IsSameType<T, int8_t>::value) {
            const uint32_t alignedCount =
                (count + 127) / 128 * 128;
            AscendC::LocalTensor<half> accumHalf =
                floatAccumBuffer_.Get<half>();
            AscendC::LocalTensor<half> sourceHalf =
                floatSourceBuffer_.Get<half>();
            AscendC::Cast(
                accumHalf,
                accum[accumOffset],
                AscendC::RoundMode::CAST_NONE,
                alignedCount);
            AscendC::Cast(
                sourceHalf,
                source,
                AscendC::RoundMode::CAST_NONE,
                alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(
                accumHalf,
                accumHalf,
                sourceHalf,
                alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            WrapInt8(accumHalf, alignedCount);
            AscendC::Cast(
                accum[accumOffset],
                accumHalf,
                AscendC::RoundMode::CAST_RINT,
                alignedCount);
        } else if constexpr (
            AscendC::IsSameType<T, bfloat16_t>::value) {
            AscendC::LocalTensor<float> accumFloat =
                floatAccumBuffer_.Get<float>();
            AscendC::LocalTensor<float> sourceFloat =
                floatSourceBuffer_.Get<float>();
            AscendC::Cast(
                accumFloat,
                accum[accumOffset],
                AscendC::RoundMode::CAST_NONE,
                count);
            AscendC::Cast(
                sourceFloat,
                source,
                AscendC::RoundMode::CAST_NONE,
                count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(
                accumFloat,
                accumFloat,
                sourceFloat,
                count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                accum[accumOffset],
                accumFloat,
                AscendC::RoundMode::CAST_RINT,
                count);
        } else {
            AscendC::Add(
                accum[accumOffset],
                accum[accumOffset],
                source,
                count);
        }
    }

    __aicore__ inline void CopyOut(
        AscendC::GlobalTensor<T>& destination,
        uint64_t destinationOffset,
        const AscendC::LocalTensor<T>& local,
        uint32_t localOffset,
        uint32_t count)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = count * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(
            destination[destinationOffset],
            local[localOffset],
            copyParams);
    }

    __aicore__ inline void ProcessInt8Task(uint64_t task)
    {
        const uint32_t innerChunkIndex =
            static_cast<uint32_t>(task % innerChunks_);
        task /= innerChunks_;
        const uint32_t dimGroupIndex =
            static_cast<uint32_t>(task % dimGroups_);
        const uint64_t outerIndex = task / dimGroups_;
        const uint64_t innerStart =
            static_cast<uint64_t>(innerChunkIndex) * INNER_CHUNK;
        const uint32_t current = static_cast<uint32_t>(
            inner_ - innerStart < INNER_CHUNK
                ? inner_ - innerStart
                : INNER_CHUNK);
        const uint32_t aligned =
            (current + 127) / 128 * 128;
        const uint64_t groupStart =
            static_cast<uint64_t>(dimGroupIndex) * dimGroup_;
        const uint32_t groupCount = static_cast<uint32_t>(
            dimSize_ - groupStart < dimGroup_
                ? dimSize_ - groupStart
                : dimGroup_);

        AscendC::LocalTensor<half> accumHalf =
            accumBuffer_.Get<half>();
        AscendC::LocalTensor<int8_t> sourceLocal =
            sourceBuffer_.Get<int8_t>();
        AscendC::LocalTensor<half> sourceHalf =
            floatAccumBuffer_.Get<half>();
        AscendC::LocalTensor<int8_t> outputLocal =
            int8OutputBuffer_.Get<int8_t>();
        uint8_t pendingAdds[MAX_DIM_GROUP];

        for (uint32_t localRow = 0;
             localRow < groupCount;
             ++localRow) {
            pendingAdds[localRow] = 0;
            const uint64_t destinationRow =
                groupStart + localRow;
            const uint64_t offset =
                (outerIndex * dimSize_ + destinationRow) *
                    inner_ +
                innerStart;
            CopyIn(
                sourceLocal,
                0,
                selfGm_,
                offset,
                current);
            AscendC::Cast(
                accumHalf[localRow * INNER_CHUNK],
                sourceLocal,
                AscendC::RoundMode::CAST_NONE,
                aligned);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                vToMte2Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                vToMte2Event_);
        }

        for (uint64_t indexPosition = 0;
             indexPosition < indexCount_;
             ++indexPosition) {
            const int32_t destinationIndex =
                indexBuffer_.Get<int32_t>().GetValue(
                    indexPosition);
            if (destinationIndex < 0 ||
                static_cast<uint64_t>(destinationIndex) < groupStart ||
                static_cast<uint64_t>(destinationIndex) >=
                    groupStart + groupCount) {
                continue;
            }
            const uint32_t localRow = static_cast<uint32_t>(
                static_cast<uint64_t>(destinationIndex) - groupStart);
            const uint64_t sourceOffset =
                (outerIndex * indexCount_ + indexPosition) *
                    inner_ +
                innerStart;
            CopyIn(
                sourceLocal,
                0,
                sourceGm_,
                sourceOffset,
                current);
            AscendC::Cast(
                sourceHalf,
                sourceLocal,
                AscendC::RoundMode::CAST_NONE,
                aligned);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(
                accumHalf[localRow * INNER_CHUNK],
                accumHalf[localRow * INNER_CHUNK],
                sourceHalf,
                aligned);
            AscendC::PipeBarrier<PIPE_V>();
            ++pendingAdds[localRow];
            if (pendingAdds[localRow] == 2) {
                AscendC::LocalTensor<half> row =
                    accumHalf[localRow * INNER_CHUNK];
                WrapInt8(row, aligned);
                pendingAdds[localRow] = 0;
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                vToMte2Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                vToMte2Event_);
        }

        for (uint32_t localRow = 0;
             localRow < groupCount;
             ++localRow) {
            AscendC::LocalTensor<half> row =
                accumHalf[localRow * INNER_CHUNK];
            WrapInt8(row, aligned);
            AscendC::Cast(
                outputLocal[localRow * INNER_CHUNK],
                row,
                AscendC::RoundMode::CAST_RINT,
                aligned);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
            vToMte3Event_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
            vToMte3Event_);
        for (uint32_t localRow = 0;
             localRow < groupCount;
             ++localRow) {
            const uint64_t destinationRow =
                groupStart + localRow;
            const uint64_t offset =
                (outerIndex * dimSize_ + destinationRow) *
                    inner_ +
                innerStart;
            CopyOut(
                outputGm_,
                offset,
                outputLocal,
                localRow * INNER_CHUNK,
                current);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
            mte3ToMte2Event_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
            mte3ToMte2Event_);
    }

    __aicore__ inline void ProcessTask(uint64_t task)
    {
        if constexpr (
            AscendC::IsSameType<T, int8_t>::value &&
            USE_BATCHED_INT8) {
            ProcessInt8Task(task);
            return;
        }
        const uint32_t innerChunkIndex =
            static_cast<uint32_t>(task % innerChunks_);
        task /= innerChunks_;
        const uint32_t dimGroupIndex =
            static_cast<uint32_t>(task % dimGroups_);
        const uint64_t outerIndex = task / dimGroups_;
        const uint64_t innerStart =
            static_cast<uint64_t>(innerChunkIndex) * INNER_CHUNK;
        const uint32_t current = static_cast<uint32_t>(
            inner_ - innerStart < INNER_CHUNK
                ? inner_ - innerStart
                : INNER_CHUNK);
        const uint64_t groupStart =
            static_cast<uint64_t>(dimGroupIndex) * dimGroup_;
        const uint32_t groupCount = static_cast<uint32_t>(
            dimSize_ - groupStart < dimGroup_
                ? dimSize_ - groupStart
                : dimGroup_);

        AscendC::LocalTensor<T> accum =
            accumBuffer_.Get<T>();
        AscendC::LocalTensor<T> sourceLocal =
            sourceBuffer_.Get<T>();

        for (uint32_t localRow = 0;
             localRow < groupCount;
             ++localRow) {
            const uint64_t destinationRow =
                groupStart + localRow;
            const uint64_t offset =
                (outerIndex * dimSize_ + destinationRow) *
                    inner_ +
                innerStart;
            CopyIn(
                accum,
                localRow * INNER_CHUNK,
                selfGm_,
                offset,
                current);
        }

        for (uint64_t indexPosition = 0;
             indexPosition < indexCount_;
             ++indexPosition) {
            const int32_t destinationIndex =
                indexBuffer_.Get<int32_t>().GetValue(
                    indexPosition);
            if (destinationIndex < 0 ||
                static_cast<uint64_t>(destinationIndex) < groupStart ||
                static_cast<uint64_t>(destinationIndex) >=
                    groupStart + groupCount) {
                continue;
            }
            const uint32_t localRow = static_cast<uint32_t>(
                static_cast<uint64_t>(destinationIndex) - groupStart);
            const uint64_t sourceOffset =
                (outerIndex * indexCount_ + indexPosition) *
                    inner_ +
                innerStart;
            CopyIn(
                sourceLocal,
                0,
                sourceGm_,
                sourceOffset,
                current);
            AddValues(
                accum,
                localRow * INNER_CHUNK,
                sourceLocal,
                current);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                vToMte2Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                vToMte2Event_);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
            vToMte3Event_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
            vToMte3Event_);
        for (uint32_t localRow = 0;
             localRow < groupCount;
             ++localRow) {
            const uint64_t destinationRow =
                groupStart + localRow;
            const uint64_t offset =
                (outerIndex * dimSize_ + destinationRow) *
                    inner_ +
                innerStart;
            CopyOut(
                outputGm_,
                offset,
                accum,
                localRow * INNER_CHUNK,
                current);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
            mte3ToMte2Event_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
            mte3ToMte2Event_);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sourceBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatAccumBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatSourceBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> int8MaskBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> int8WorkBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> int8OutputBuffer_;
    AscendC::GlobalTensor<T> selfGm_;
    AscendC::GlobalTensor<int32_t> indexGm_;
    AscendC::GlobalTensor<T> sourceGm_;
    AscendC::GlobalTensor<T> outputGm_;
    uint64_t totalElements_;
    uint64_t sourceElements_;
    uint64_t dimSize_;
    uint64_t inner_;
    uint64_t indexCount_;
    uint64_t taskCount_;
    uint32_t dimGroup_;
    uint32_t dimGroups_;
    uint32_t innerChunks_;
    uint32_t blockIdx_;
    uint32_t blockNum_;
    event_t mte2ToVEvent_;
    event_t mte2ToSEvent_;
    event_t vToMte2Event_;
    event_t sToMte2Event_;
    event_t vToMte3Event_;
    event_t sToMte3Event_;
    event_t mte3ToMte2Event_;
};

extern "C" __global__ __aicore__ void index_add(
    GM_ADDR self,
    GM_ADDR index,
    GM_ADDR source,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.taskCount == 0) {
        return;
    }
    if (TILING_KEY_IS(1)) {
        KernelIndexAdd<DTYPE_SELF, false> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelIndexAdd<DTYPE_SELF, true> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    }
}
