#include "kernel_operator.h"

constexpr uint32_t INNER_CHUNK = 256;
constexpr uint32_t MAX_DIM_GROUP = 64;
constexpr uint32_t MAX_INDEX_COUNT = 8000;
constexpr uint32_t ACCUM_ELEMENTS =
    INNER_CHUNK * MAX_DIM_GROUP;

template <
    typename T,
    bool USE_BATCHED_INT8,
    bool USE_HIT_REUSE,
    bool USE_CONTIGUOUS_TASKS>
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
        chunkGroups_ = tiling.chunkGroups;
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
        if constexpr (USE_HIT_REUSE) {
            pipe_.InitBuffer(
                sourceQueue_,
                2,
                INNER_CHUNK * sizeof(T));
        }
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
            hitPositionBuffer_,
            MAX_INDEX_COUNT * sizeof(uint16_t));
        pipe_.InitBuffer(
            hitRowBuffer_,
            MAX_INDEX_COUNT * sizeof(uint8_t));
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
        if constexpr (USE_CONTIGUOUS_TASKS) {
            ProcessContiguousTasks();
        } else {
            for (uint64_t task = blockIdx_;
                 task < taskCount_;
                 task += blockNum_) {
                ProcessTask(task);
            }
        }
    }

private:
    __aicore__ inline void ProcessContiguousTasks()
    {
        const uint64_t baseTasks = taskCount_ / blockNum_;
        const uint64_t extraTasks = taskCount_ % blockNum_;
        const uint64_t tasksForBlock =
            baseTasks + (blockIdx_ < extraTasks ? 1 : 0);
        const uint64_t firstTask =
            static_cast<uint64_t>(blockIdx_) * baseTasks +
            (blockIdx_ < extraTasks ? blockIdx_ : extraTasks);
        const uint64_t lastTask = firstTask + tasksForBlock;
        for (uint64_t task = firstTask;
             task < lastTask;
             ++task) {
            ProcessTask(task);
        }
    }

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

    __aicore__ inline void EnqueueSource(
        uint64_t sourceOffset,
        uint32_t count)
    {
        AscendC::LocalTensor<T> local =
            sourceQueue_.template AllocTensor<T>();
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
            local,
            sourceGm_[sourceOffset],
            copyParams,
            padParams);
        sourceQueue_.EnQue(local);
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

    __aicore__ inline bool CanBatchRows(uint32_t current) const
    {
        const uint64_t gmStrideBytes =
            (inner_ - current) * sizeof(T);
        return current * sizeof(T) % 32 == 0 &&
            gmStrideBytes <= 0xffffffffULL;
    }

    __aicore__ inline void CopyGroupIn(
        AscendC::LocalTensor<T>& local,
        const AscendC::GlobalTensor<T>& source,
        uint64_t sourceOffset,
        uint32_t groupCount,
        uint32_t current)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount =
            static_cast<uint16_t>(groupCount);
        copyParams.blockLen = current * sizeof(T);
        copyParams.srcStride = static_cast<uint32_t>(
            (inner_ - current) * sizeof(T));
        copyParams.dstStride =
            (INNER_CHUNK - current) * sizeof(T);
        AscendC::DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = static_cast<T>(0);
        AscendC::DataCopyPad(
            local,
            source[sourceOffset],
            copyParams,
            padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
            mte2ToVEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            mte2ToVEvent_);
    }

    __aicore__ inline void CopyGroupOut(
        AscendC::GlobalTensor<T>& destination,
        uint64_t destinationOffset,
        const AscendC::LocalTensor<T>& local,
        uint32_t groupCount,
        uint32_t current)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount =
            static_cast<uint16_t>(groupCount);
        copyParams.blockLen = current * sizeof(T);
        copyParams.srcStride =
            (INNER_CHUNK - current) * sizeof(T);
        copyParams.dstStride = static_cast<uint32_t>(
            (inner_ - current) * sizeof(T));
        AscendC::DataCopyPad(
            destination[destinationOffset],
            local,
            copyParams);
    }

    __aicore__ inline uint32_t BuildHitList(
        uint64_t groupStart,
        uint32_t groupCount)
    {
        AscendC::LocalTensor<int32_t> indexLocal =
            indexBuffer_.Get<int32_t>();
        AscendC::LocalTensor<uint16_t> hitPositions =
            hitPositionBuffer_.Get<uint16_t>();
        AscendC::LocalTensor<uint8_t> hitRows =
            hitRowBuffer_.Get<uint8_t>();
        uint32_t hitCount = 0;
        for (uint64_t indexPosition = 0;
             indexPosition < indexCount_;
             ++indexPosition) {
            const int32_t destinationIndex =
                indexLocal.GetValue(indexPosition);
            if (destinationIndex < 0 ||
                static_cast<uint64_t>(destinationIndex) < groupStart ||
                static_cast<uint64_t>(destinationIndex) >=
                    groupStart + groupCount) {
                continue;
            }
            hitPositions.SetValue(
                hitCount,
                static_cast<uint16_t>(indexPosition));
            hitRows.SetValue(
                hitCount,
                static_cast<uint8_t>(
                    static_cast<uint64_t>(destinationIndex) -
                    groupStart));
            ++hitCount;
        }
        return hitCount;
    }

    __aicore__ inline void ProcessInt8Chunk(
        uint64_t outerIndex,
        uint64_t groupStart,
        uint32_t groupCount,
        uint64_t innerStart,
        uint32_t current,
        uint32_t hitCount,
        const AscendC::LocalTensor<uint16_t>& hitPositions,
        const AscendC::LocalTensor<uint8_t>& hitRows)
    {
        const uint32_t aligned =
            (current + 127) / 128 * 128;

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
        }
        const uint64_t groupOffset =
            (outerIndex * dimSize_ + groupStart) *
                inner_ +
            innerStart;
        if (CanBatchRows(current)) {
            CopyGroupIn(
                outputLocal,
                selfGm_,
                groupOffset,
                groupCount,
                current);
            for (uint32_t localRow = 0;
                 localRow < groupCount;
                 ++localRow) {
                AscendC::Cast(
                    accumHalf[localRow * INNER_CHUNK],
                    outputLocal[localRow * INNER_CHUNK],
                    AscendC::RoundMode::CAST_NONE,
                    aligned);
            }
        } else {
            for (uint32_t localRow = 0;
                 localRow < groupCount;
                 ++localRow) {
                CopyIn(
                    sourceLocal,
                    0,
                    selfGm_,
                    groupOffset +
                        static_cast<uint64_t>(localRow) * inner_,
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
        }

        const uint64_t updateCount =
            USE_HIT_REUSE ? hitCount : indexCount_;
        for (uint64_t update = 0;
             update < updateCount;
             ++update) {
            uint64_t indexPosition = update;
            uint32_t localRow = 0;
            if constexpr (USE_HIT_REUSE) {
                indexPosition =
                    hitPositions.GetValue(update);
                localRow = hitRows.GetValue(update);
            } else {
                const int32_t destinationIndex =
                    indexBuffer_.Get<int32_t>().GetValue(update);
                if (destinationIndex < 0 ||
                    static_cast<uint64_t>(destinationIndex) <
                        groupStart ||
                    static_cast<uint64_t>(destinationIndex) >=
                        groupStart + groupCount) {
                    continue;
                }
                localRow = static_cast<uint32_t>(
                    static_cast<uint64_t>(destinationIndex) -
                    groupStart);
            }
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
        if (CanBatchRows(current)) {
            CopyGroupOut(
                outputGm_,
                groupOffset,
                outputLocal,
                groupCount,
                current);
        } else {
            for (uint32_t localRow = 0;
                 localRow < groupCount;
                 ++localRow) {
                CopyOut(
                    outputGm_,
                    groupOffset +
                        static_cast<uint64_t>(localRow) * inner_,
                    outputLocal,
                    localRow * INNER_CHUNK,
                    current);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
            mte3ToMte2Event_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
            mte3ToMte2Event_);
    }

    __aicore__ inline void ProcessRegularChunk(
        uint64_t outerIndex,
        uint64_t groupStart,
        uint32_t groupCount,
        uint64_t innerStart,
        uint32_t current,
        uint32_t hitCount,
        const AscendC::LocalTensor<uint16_t>& hitPositions,
        const AscendC::LocalTensor<uint8_t>& hitRows)
    {
        AscendC::LocalTensor<T> accum =
            accumBuffer_.Get<T>();
        AscendC::LocalTensor<T> sourceLocal =
            sourceBuffer_.Get<T>();

        const uint64_t groupOffset =
            (outerIndex * dimSize_ + groupStart) *
                inner_ +
            innerStart;
        if (CanBatchRows(current)) {
            CopyGroupIn(
                accum,
                selfGm_,
                groupOffset,
                groupCount,
                current);
        } else {
            for (uint32_t localRow = 0;
                 localRow < groupCount;
                 ++localRow) {
                CopyIn(
                    accum,
                    localRow * INNER_CHUNK,
                    selfGm_,
                    groupOffset +
                        static_cast<uint64_t>(localRow) * inner_,
                    current);
            }
        }

        if constexpr (USE_HIT_REUSE) {
            if (hitCount != 0) {
                const uint64_t firstPosition =
                    hitPositions.GetValue(0);
                EnqueueSource(
                    (outerIndex * indexCount_ + firstPosition) *
                        inner_ +
                        innerStart,
                    current);
            }
            for (uint32_t update = 0;
                 update < hitCount;
                 ++update) {
                if (update + 1 < hitCount) {
                    const uint64_t nextPosition =
                        hitPositions.GetValue(update + 1);
                    EnqueueSource(
                        (outerIndex * indexCount_ + nextPosition) *
                            inner_ +
                            innerStart,
                        current);
                }
                AscendC::LocalTensor<T> queuedSource =
                    sourceQueue_.template DeQue<T>();
                const uint32_t localRow =
                    hitRows.GetValue(update);
                AddValues(
                    accum,
                    localRow * INNER_CHUNK,
                    queuedSource,
                    current);
                sourceQueue_.FreeTensor(queuedSource);
            }
        } else {
            for (uint64_t indexPosition = 0;
                 indexPosition < indexCount_;
                 ++indexPosition) {
                const int32_t destinationIndex =
                    indexBuffer_.Get<int32_t>().GetValue(indexPosition);
                if (destinationIndex < 0 ||
                    static_cast<uint64_t>(destinationIndex) <
                        groupStart ||
                    static_cast<uint64_t>(destinationIndex) >=
                        groupStart + groupCount) {
                    continue;
                }
                const uint32_t localRow = static_cast<uint32_t>(
                    static_cast<uint64_t>(destinationIndex) -
                    groupStart);
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
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
            vToMte3Event_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
            vToMte3Event_);
        if (CanBatchRows(current)) {
            CopyGroupOut(
                outputGm_,
                groupOffset,
                accum,
                groupCount,
                current);
        } else {
            for (uint32_t localRow = 0;
                 localRow < groupCount;
                 ++localRow) {
                CopyOut(
                    outputGm_,
                    groupOffset +
                        static_cast<uint64_t>(localRow) * inner_,
                    accum,
                    localRow * INNER_CHUNK,
                    current);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
            mte3ToMte2Event_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
            mte3ToMte2Event_);
    }

    __aicore__ inline void ProcessTask(uint64_t task)
    {
        const uint32_t activeChunkGroups =
            USE_HIT_REUSE ? chunkGroups_ : innerChunks_;
        const uint32_t chunkGroupIndex =
            static_cast<uint32_t>(task % activeChunkGroups);
        task /= activeChunkGroups;
        const uint32_t dimGroupIndex =
            static_cast<uint32_t>(task % dimGroups_);
        const uint64_t outerIndex = task / dimGroups_;
        const uint64_t groupStart =
            static_cast<uint64_t>(dimGroupIndex) * dimGroup_;
        const uint32_t groupCount = static_cast<uint32_t>(
            dimSize_ - groupStart < dimGroup_
                ? dimSize_ - groupStart
                : dimGroup_);
        uint32_t firstChunk = chunkGroupIndex;
        uint32_t chunks = 1;
        uint32_t hitCount = 0;
        if constexpr (USE_HIT_REUSE) {
            const uint32_t baseChunks =
                innerChunks_ / chunkGroups_;
            const uint32_t extraChunkGroups =
                innerChunks_ % chunkGroups_;
            firstChunk =
                chunkGroupIndex * baseChunks +
                (chunkGroupIndex < extraChunkGroups
                    ? chunkGroupIndex
                    : extraChunkGroups);
            chunks =
                baseChunks +
                (chunkGroupIndex < extraChunkGroups ? 1 : 0);
            hitCount = BuildHitList(groupStart, groupCount);
        }
        const AscendC::LocalTensor<uint16_t> hitPositions =
            hitPositionBuffer_.Get<uint16_t>();
        const AscendC::LocalTensor<uint8_t> hitRows =
            hitRowBuffer_.Get<uint8_t>();

        for (uint32_t chunkOffset = 0;
             chunkOffset < chunks;
             ++chunkOffset) {
            const uint32_t innerChunkIndex =
                firstChunk + chunkOffset;
            const uint64_t innerStart =
                static_cast<uint64_t>(innerChunkIndex) *
                INNER_CHUNK;
            const uint32_t current = static_cast<uint32_t>(
                inner_ - innerStart < INNER_CHUNK
                    ? inner_ - innerStart
                    : INNER_CHUNK);
            if constexpr (
                AscendC::IsSameType<T, int8_t>::value &&
                USE_BATCHED_INT8) {
                ProcessInt8Chunk(
                    outerIndex,
                    groupStart,
                    groupCount,
                    innerStart,
                    current,
                    hitCount,
                    hitPositions,
                    hitRows);
            } else {
                ProcessRegularChunk(
                    outerIndex,
                    groupStart,
                    groupCount,
                    innerStart,
                    current,
                    hitCount,
                    hitPositions,
                    hitRows);
            }
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sourceBuffer_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> sourceQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatAccumBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatSourceBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hitPositionBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hitRowBuffer_;
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
    uint32_t chunkGroups_;
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
        KernelIndexAdd<DTYPE_SELF, false, false, false> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelIndexAdd<DTYPE_SELF, true, false, false> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelIndexAdd<DTYPE_SELF, false, true, true> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        KernelIndexAdd<DTYPE_SELF, true, true, true> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        KernelIndexAdd<DTYPE_SELF, false, false, true> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(6)) {
        KernelIndexAdd<DTYPE_SELF, true, false, true> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    }
}
