#include "kernel_operator.h"

constexpr uint32_t TILE_ELEMENTS = 4096;
constexpr uint32_t MAX_RANK = 6;
constexpr uint32_t MATRIX_TILE = 16;
constexpr uint32_t MATRIX_TILE_ELEMENTS = 256;
constexpr uint32_t FAST_BUFFER_NUM = 4;

class KernelTransposeFastHalf {
public:
    __aicore__ inline KernelTransposeFastHalf() {}

    __aicore__ inline void Init(
        GM_ADDR input,
        GM_ADDR output,
        const uint32_t rows,
        const uint32_t cols,
        const uint32_t tileCols,
        const uint32_t baseTilesPerBlock,
        const uint32_t extraTileBlocks)
    {
        rows_ = rows;
        cols_ = cols;
        tileCols_ = tileCols;
        inputSrcStride_ = static_cast<uint16_t>(
            cols_ / MATRIX_TILE - 1U);
        outputDstStride_ = static_cast<uint16_t>(
            rows_ / MATRIX_TILE - 1U);

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        tiles_ = baseTilesPerBlock +
            (blockIdx < extraTileBlocks ? 1U : 0U);
        const uint32_t firstTile =
            blockIdx * baseTilesPerBlock +
            (blockIdx < extraTileBlocks
                ? blockIdx
                : extraTileBlocks);
        tileRow_ = firstTile / tileCols_;
        tileCol_ = firstTile % tileCols_;
        inputOffset_ =
            tileRow_ * MATRIX_TILE * cols_ +
            tileCol_ * MATRIX_TILE;
        outputOffset_ =
            tileCol_ * MATRIX_TILE * rows_ +
            tileRow_ * MATRIX_TILE;

        inputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(input),
            rows_ * cols_);
        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(output),
            rows_ * cols_);
        pipe_.InitBuffer(
            inputQueue_,
            FAST_BUFFER_NUM,
            MATRIX_TILE_ELEMENTS * sizeof(half));
        pipe_.InitBuffer(
            outputQueue_,
            FAST_BUFFER_NUM,
            MATRIX_TILE_ELEMENTS * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t tile = 0; tile < tiles_; ++tile) {
            AscendC::LocalTensor<half> inputLocal =
                inputQueue_.AllocTensor<half>();
            AscendC::DataCopyParams inputParams{
                MATRIX_TILE,
                1,
                inputSrcStride_,
                0};
            AscendC::DataCopy(
                inputLocal,
                inputGm_[inputOffset_],
                inputParams);
            inputQueue_.EnQue(inputLocal);

            inputLocal = inputQueue_.DeQue<half>();
            AscendC::LocalTensor<half> outputLocal =
                outputQueue_.AllocTensor<half>();
            AscendC::Transpose(outputLocal, inputLocal);
            outputQueue_.EnQue(outputLocal);
            inputQueue_.FreeTensor(inputLocal);

            outputLocal = outputQueue_.DeQue<half>();
            AscendC::DataCopyParams outputParams{
                MATRIX_TILE,
                1,
                0,
                outputDstStride_};
            AscendC::DataCopy(
                outputGm_[outputOffset_],
                outputLocal,
                outputParams);
            outputQueue_.FreeTensor(outputLocal);
            AdvanceTile();
        }
    }

private:
    __aicore__ inline void AdvanceTile()
    {
        ++tileCol_;
        if (tileCol_ == tileCols_) {
            tileCol_ = 0;
            ++tileRow_;
            inputOffset_ =
                tileRow_ * MATRIX_TILE * cols_;
            outputOffset_ =
                tileRow_ * MATRIX_TILE;
        } else {
            inputOffset_ += MATRIX_TILE;
            outputOffset_ +=
                MATRIX_TILE * rows_;
        }
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, FAST_BUFFER_NUM>
        inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, FAST_BUFFER_NUM>
        outputQueue_;
    AscendC::GlobalTensor<half> inputGm_;
    AscendC::GlobalTensor<half> outputGm_;
    uint32_t inputOffset_;
    uint32_t outputOffset_;
    uint32_t rows_;
    uint32_t cols_;
    uint32_t tileCols_;
    uint32_t tileRow_;
    uint32_t tileCol_;
    uint32_t tiles_;
    uint16_t inputSrcStride_;
    uint16_t outputDstStride_;
};

class KernelTransposeFastHalfBatched {
public:
    __aicore__ inline KernelTransposeFastHalfBatched() {}

    __aicore__ inline void Init(
        GM_ADDR input,
        GM_ADDR output,
        const uint32_t rows,
        const uint32_t cols,
        const uint32_t tileCols,
        const uint32_t baseTilesPerBlock,
        const uint32_t extraTileBlocks)
    {
        rows_ = rows;
        cols_ = cols;
        tileCols_ = tileCols;
        tileRows_ = rows_ / MATRIX_TILE;
        matrixTiles_ = tileRows_ * tileCols_;
        matrixElements_ =
            static_cast<uint64_t>(rows_) * cols_;
        inputSrcStride_ = static_cast<uint16_t>(
            cols_ / MATRIX_TILE - 1U);
        outputDstStride_ = static_cast<uint16_t>(
            rows_ / MATRIX_TILE - 1U);

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        tiles_ = baseTilesPerBlock +
            (blockIdx < extraTileBlocks ? 1U : 0U);
        const uint32_t firstTile =
            blockIdx * baseTilesPerBlock +
            (blockIdx < extraTileBlocks
                ? blockIdx
                : extraTileBlocks);
        batch_ = firstTile / matrixTiles_;
        const uint32_t tileInMatrix =
            firstTile % matrixTiles_;
        tileRow_ = tileInMatrix / tileCols_;
        tileCol_ = tileInMatrix % tileCols_;
        matrixBase_ =
            static_cast<uint64_t>(batch_) * matrixElements_;
        inputOffset_ =
            matrixBase_ +
            static_cast<uint64_t>(tileRow_) *
                MATRIX_TILE * cols_ +
            tileCol_ * MATRIX_TILE;
        outputOffset_ =
            matrixBase_ +
            static_cast<uint64_t>(tileCol_) *
                MATRIX_TILE * rows_ +
            tileRow_ * MATRIX_TILE;

        const uint32_t totalTiles =
            baseTilesPerBlock * AscendC::GetBlockNum() +
            extraTileBlocks;
        const uint64_t totalElements =
            static_cast<uint64_t>(
                totalTiles / matrixTiles_) *
            matrixElements_;
        inputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(input),
            totalElements);
        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(output),
            totalElements);
        pipe_.InitBuffer(
            inputQueue_,
            FAST_BUFFER_NUM,
            MATRIX_TILE_ELEMENTS * sizeof(half));
        pipe_.InitBuffer(
            outputQueue_,
            FAST_BUFFER_NUM,
            MATRIX_TILE_ELEMENTS * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t tile = 0; tile < tiles_; ++tile) {
            AscendC::LocalTensor<half> inputLocal =
                inputQueue_.AllocTensor<half>();
            AscendC::DataCopyParams inputParams{
                MATRIX_TILE,
                1,
                inputSrcStride_,
                0};
            AscendC::DataCopy(
                inputLocal,
                inputGm_[inputOffset_],
                inputParams);
            inputQueue_.EnQue(inputLocal);

            inputLocal = inputQueue_.DeQue<half>();
            AscendC::LocalTensor<half> outputLocal =
                outputQueue_.AllocTensor<half>();
            AscendC::Transpose(outputLocal, inputLocal);
            outputQueue_.EnQue(outputLocal);
            inputQueue_.FreeTensor(inputLocal);

            outputLocal = outputQueue_.DeQue<half>();
            AscendC::DataCopyParams outputParams{
                MATRIX_TILE,
                1,
                0,
                outputDstStride_};
            AscendC::DataCopy(
                outputGm_[outputOffset_],
                outputLocal,
                outputParams);
            outputQueue_.FreeTensor(outputLocal);
            AdvanceTile();
        }
    }

private:
    __aicore__ inline void AdvanceTile()
    {
        ++tileCol_;
        if (tileCol_ == tileCols_) {
            tileCol_ = 0;
            ++tileRow_;
            if (tileRow_ == tileRows_) {
                tileRow_ = 0;
                ++batch_;
                matrixBase_ += matrixElements_;
            }
            inputOffset_ =
                matrixBase_ +
                static_cast<uint64_t>(tileRow_) *
                    MATRIX_TILE * cols_;
            outputOffset_ =
                matrixBase_ +
                tileRow_ * MATRIX_TILE;
        } else {
            inputOffset_ += MATRIX_TILE;
            outputOffset_ +=
                MATRIX_TILE * rows_;
        }
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, FAST_BUFFER_NUM>
        inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, FAST_BUFFER_NUM>
        outputQueue_;
    AscendC::GlobalTensor<half> inputGm_;
    AscendC::GlobalTensor<half> outputGm_;
    uint64_t inputOffset_;
    uint64_t outputOffset_;
    uint64_t matrixBase_;
    uint64_t matrixElements_;
    uint32_t rows_;
    uint32_t cols_;
    uint32_t tileCols_;
    uint32_t tileRows_;
    uint32_t matrixTiles_;
    uint32_t batch_;
    uint32_t tileRow_;
    uint32_t tileCol_;
    uint32_t tiles_;
    uint16_t inputSrcStride_;
    uint16_t outputDstStride_;
};

template <typename T>
class KernelTransposeRotation {
public:
    __aicore__ inline KernelTransposeRotation() {}

    __aicore__ inline void Init(
        GM_ADDR input,
        GM_ADDR output,
        const uint32_t rows,
        const uint32_t cols,
        const uint32_t tileCols,
        const uint32_t baseTilesPerBlock,
        const uint32_t extraTileBlocks)
    {
        rows_ = rows;
        cols_ = cols;
        tileCols_ = tileCols;
        tileRows_ =
            (rows_ + TILE_SIDE - 1U) / TILE_SIDE;
        matrixTiles_ = tileRows_ * tileCols_;
        matrixElements_ =
            static_cast<uint64_t>(rows_) * cols_;
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        tiles_ = baseTilesPerBlock +
            (blockIdx < extraTileBlocks ? 1U : 0U);
        const uint32_t firstTile =
            blockIdx * baseTilesPerBlock +
            (blockIdx < extraTileBlocks
                ? blockIdx
                : extraTileBlocks);
        batch_ = firstTile / matrixTiles_;
        const uint32_t tileInMatrix =
            firstTile % matrixTiles_;
        tileRow_ = tileInMatrix / tileCols_;
        tileCol_ = tileInMatrix % tileCols_;
        matrixBase_ =
            static_cast<uint64_t>(batch_) * matrixElements_;

        const uint32_t totalTiles =
            baseTilesPerBlock * AscendC::GetBlockNum() +
            extraTileBlocks;
        const uint64_t totalElements =
            static_cast<uint64_t>(
                totalTiles / matrixTiles_) *
            matrixElements_;
        inputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(input),
            totalElements);
        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(output),
            totalElements);
        pipe_.InitBuffer(
            inputBuffer_,
            TILE_ELEMENTS * sizeof(T));
        pipe_.InitBuffer(
            outputBuffer_,
            TILE_ELEMENTS * sizeof(T));
        mte2ToSEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE2_S));
        sToMte3Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::S_MTE3));
        mte2ToVEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE2_V));
        vToMte3Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::V_MTE3));
        mte3ToMte2Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<T> inputLocal =
            inputBuffer_.Get<T>();
        AscendC::LocalTensor<T> outputLocal =
            outputBuffer_.Get<T>();
        for (uint32_t tile = 0; tile < tiles_; ++tile) {
            const uint32_t rowStart = tileRow_ * TILE_SIDE;
            const uint32_t colStart = tileCol_ * TILE_SIDE;
            const uint32_t validRows =
                rows_ - rowStart < TILE_SIDE
                    ? rows_ - rowStart
                    : TILE_SIDE;
            const uint32_t validCols =
                cols_ - colStart < TILE_SIDE
                    ? cols_ - colStart
                    : TILE_SIDE;

            AscendC::DataCopyExtParams inputParams;
            inputParams.blockCount =
                static_cast<uint16_t>(validRows);
            inputParams.blockLen = validCols * sizeof(T);
            inputParams.srcStride =
                (cols_ - validCols) * sizeof(T);
            inputParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<T> padding;
            padding.isPad = false;
            padding.leftPadding = 0;
            padding.rightPadding =
                static_cast<uint8_t>(TILE_SIDE - validCols);
            const T zero = {};
            padding.paddingValue = zero;
            const uint64_t inputOffset =
                matrixBase_ +
                static_cast<uint64_t>(rowStart) * cols_ +
                colStart;
            AscendC::DataCopyPad(
                inputLocal,
                inputGm_[inputOffset],
                inputParams,
                padding);

            if constexpr (std::is_same<T, half>::value) {
                if (validRows == MATRIX_TILE &&
                    validCols == MATRIX_TILE) {
                    AscendC::SetFlag<
                        AscendC::HardEvent::MTE2_V>(
                        mte2ToVEvent_);
                    AscendC::WaitFlag<
                        AscendC::HardEvent::MTE2_V>(
                        mte2ToVEvent_);
                    AscendC::Transpose(
                        outputLocal,
                        inputLocal);
                    AscendC::SetFlag<
                        AscendC::HardEvent::V_MTE3>(
                        vToMte3Event_);
                    AscendC::WaitFlag<
                        AscendC::HardEvent::V_MTE3>(
                        vToMte3Event_);
                } else {
                    ScalarTranspose(
                        inputLocal,
                        outputLocal,
                        validRows,
                        validCols);
                }
            } else {
                ScalarTranspose(
                    inputLocal,
                    outputLocal,
                    validRows,
                    validCols);
            }

            AscendC::DataCopyExtParams outputParams;
            outputParams.blockCount =
                static_cast<uint16_t>(validCols);
            outputParams.blockLen = validRows * sizeof(T);
            outputParams.srcStride = 0;
            outputParams.dstStride =
                (rows_ - validRows) * sizeof(T);
            const uint64_t outputOffset =
                matrixBase_ +
                static_cast<uint64_t>(colStart) * rows_ +
                rowStart;
            AscendC::DataCopyPad(
                outputGm_[outputOffset],
                outputLocal,
                outputParams);
            AscendC::SetFlag<
                AscendC::HardEvent::MTE3_MTE2>(
                mte3ToMte2Event_);
            AscendC::WaitFlag<
                AscendC::HardEvent::MTE3_MTE2>(
                mte3ToMte2Event_);
            AdvanceTile();
        }
    }

private:
    __aicore__ inline void ScalarTranspose(
        const AscendC::LocalTensor<T>& inputLocal,
        AscendC::LocalTensor<T>& outputLocal,
        const uint32_t validRows,
        const uint32_t validCols)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(
            mte2ToSEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(
            mte2ToSEvent_);
        for (uint32_t row = 0; row < validRows; ++row) {
            for (uint32_t col = 0; col < validCols; ++col) {
                outputLocal.SetValue(
                    col * TILE_SIDE + row,
                    inputLocal.GetValue(row * TILE_SIDE + col));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(
            sToMte3Event_);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(
            sToMte3Event_);
    }

    __aicore__ inline void AdvanceTile()
    {
        ++tileCol_;
        if (tileCol_ == tileCols_) {
            tileCol_ = 0;
            ++tileRow_;
            if (tileRow_ == tileRows_) {
                tileRow_ = 0;
                ++batch_;
                matrixBase_ += matrixElements_;
            }
        }
    }

    static constexpr uint32_t TILE_SIDE =
        32U / sizeof(T);
    static constexpr uint32_t TILE_ELEMENTS =
        TILE_SIDE * TILE_SIDE;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuffer_;
    AscendC::GlobalTensor<T> inputGm_;
    AscendC::GlobalTensor<T> outputGm_;
    uint64_t matrixBase_;
    uint64_t matrixElements_;
    uint32_t rows_;
    uint32_t cols_;
    uint32_t tileCols_;
    uint32_t tileRows_;
    uint32_t matrixTiles_;
    uint32_t tileRow_;
    uint32_t tileCol_;
    uint32_t batch_;
    uint32_t tiles_;
    event_t mte2ToSEvent_;
    event_t sToMte3Event_;
    event_t mte2ToVEvent_;
    event_t vToMte3Event_;
    event_t mte3ToMte2Event_;
};

template <typename T>
class KernelTransposeGather {
public:
    __aicore__ inline KernelTransposeGather() {}

    __aicore__ inline void Init(
        GM_ADDR input,
        GM_ADDR output,
        const TransposeTilingData& tiling)
    {
        totalElements_ = tiling.totalElements;
        rank_ = tiling.rank;
        lastDim_ = tiling.gatherLastDim;
        inputStride_ = tiling.gatherInputStride;
        chunksPerRun_ = tiling.gatherChunksPerRun;
        for (uint32_t axis = 0; axis < MAX_RANK; ++axis) {
            outputDims_[axis] = tiling.outputDims[axis];
            inputStrides_[axis] = tiling.inputStrides[axis];
        }

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        tasks_ = tiling.baseRunsPerBlock +
            (blockIdx < tiling.extraRunBlocks ? 1U : 0U);
        firstTask_ =
            blockIdx * tiling.baseRunsPerBlock +
            (blockIdx < tiling.extraRunBlocks
                ? blockIdx
                : tiling.extraRunBlocks);

        inputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(input),
            totalElements_);
        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(output),
            totalElements_);
        pipe_.InitBuffer(
            gatherBuffer_,
            GATHER_ELEMENTS * 32U);
        pipe_.InitBuffer(
            outputBuffer_,
            GATHER_ELEMENTS * sizeof(T));
        if constexpr (sizeof(T) > 1U) {
            pipe_.InitBuffer(patternBuffer_, 32U);
        }
        mte2ToSEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE2_S));
        sToMte3Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::S_MTE3));
        mte2ToVEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE2_V));
        vToMte3Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::V_MTE3));
        mte3ToMte2Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<T> gatherLocal =
            gatherBuffer_.Get<T>();
        AscendC::LocalTensor<T> outputLocal =
            outputBuffer_.Get<T>();
        if constexpr (sizeof(T) > 1U) {
            InitializePattern();
        }
        for (uint64_t taskOffset = 0;
             taskOffset < tasks_;
             ++taskOffset) {
            const uint64_t task = firstTask_ + taskOffset;
            const uint64_t run = task / chunksPerRun_;
            const uint32_t chunk =
                static_cast<uint32_t>(task % chunksPerRun_);
            const uint32_t elementOffset =
                chunk * GATHER_ELEMENTS;
            const uint32_t current =
                lastDim_ - elementOffset < GATHER_ELEMENTS
                    ? lastDim_ - elementOffset
                    : GATHER_ELEMENTS;
            const uint64_t outputRunStart =
                run * lastDim_;
            const uint64_t inputRunStart =
                InputOffset(outputRunStart);

            AscendC::DataCopyExtParams inputParams;
            inputParams.blockCount =
                static_cast<uint16_t>(current);
            inputParams.blockLen = sizeof(T);
            inputParams.srcStride =
                (inputStride_ - 1U) * sizeof(T);
            inputParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<T> padding;
            padding.isPad = false;
            padding.leftPadding = 0;
            padding.rightPadding =
                static_cast<uint8_t>(
                    ELEMENTS_PER_BLOCK - 1U);
            const T zero = {};
            padding.paddingValue = zero;
            AscendC::DataCopyPad(
                gatherLocal,
                inputGm_[
                    inputRunStart +
                    static_cast<uint64_t>(elementOffset) *
                        inputStride_],
                inputParams,
                padding);
            if constexpr (sizeof(T) > 1U) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
                    mte2ToVEvent_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                    mte2ToVEvent_);
                VectorCompact(
                    gatherLocal,
                    outputLocal,
                    current);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                    vToMte3Event_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                    vToMte3Event_);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(
                    mte2ToSEvent_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(
                    mte2ToSEvent_);
                for (uint32_t element = 0;
                     element < current;
                     ++element) {
                    outputLocal.SetValue(
                        element,
                        gatherLocal.GetValue(
                            element * ELEMENTS_PER_BLOCK));
                }
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(
                    sToMte3Event_);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(
                    sToMte3Event_);
            }

            AscendC::DataCopyExtParams outputParams;
            outputParams.blockCount = 1;
            outputParams.blockLen = current * sizeof(T);
            outputParams.srcStride = 0;
            outputParams.dstStride = 0;
            AscendC::DataCopyPad(
                outputGm_[
                    outputRunStart + elementOffset],
                outputLocal,
                outputParams);
            AscendC::SetFlag<
                AscendC::HardEvent::MTE3_MTE2>(
                mte3ToMte2Event_);
            AscendC::WaitFlag<
                AscendC::HardEvent::MTE3_MTE2>(
                mte3ToMte2Event_);
        }
    }

private:
    __aicore__ inline void InitializePattern()
    {
        using PatternT = typename std::conditional<
            sizeof(T) == 2U,
            uint16_t,
            uint32_t>::type;
        AscendC::LocalTensor<PatternT> patternLocal =
            patternBuffer_.Get<PatternT>();
        constexpr uint32_t PATTERN_ELEMENTS =
            sizeof(T) == 2U ? 8U : 2U;
        constexpr PatternT PATTERN_VALUE =
            sizeof(T) == 2U
                ? static_cast<PatternT>(1U)
                : static_cast<PatternT>(0x01010101U);
        AscendC::Duplicate(
            patternLocal,
            PATTERN_VALUE,
            PATTERN_ELEMENTS);
    }

    __aicore__ inline void VectorCompact(
        const AscendC::LocalTensor<T>& gatherLocal,
        AscendC::LocalTensor<T>& outputLocal,
        const uint32_t current)
    {
        using PatternT = typename std::conditional<
            sizeof(T) == 2U,
            uint16_t,
            uint32_t>::type;
        AscendC::LocalTensor<PatternT> patternLocal =
            patternBuffer_.Get<PatternT>();
        AscendC::GatherMaskParams params;
        params.src0BlockStride = 1;
        params.repeatTimes = static_cast<uint16_t>(
            (current + 7U) / 8U);
        params.src0RepeatStride = 8;
        params.src1RepeatStride = 0;
        uint64_t selected = 0;
        AscendC::GatherMask(
            outputLocal,
            gatherLocal,
            patternLocal,
            false,
            0,
            params,
            selected);
    }

    __aicore__ inline uint64_t InputOffset(
        uint64_t outputIndex) const
    {
        uint64_t inputOffset = 0;
        for (int32_t axis = static_cast<int32_t>(rank_) - 1;
             axis >= 0;
             --axis) {
            const uint64_t coordinate =
                outputIndex % outputDims_[axis];
            outputIndex /= outputDims_[axis];
            inputOffset += coordinate * inputStrides_[axis];
        }
        return inputOffset;
    }

    static constexpr uint32_t GATHER_ELEMENTS = 2048;
    static constexpr uint32_t ELEMENTS_PER_BLOCK =
        32U / sizeof(T);
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> patternBuffer_;
    AscendC::GlobalTensor<T> inputGm_;
    AscendC::GlobalTensor<T> outputGm_;
    uint64_t totalElements_;
    uint64_t firstTask_;
    uint64_t tasks_;
    uint32_t rank_;
    uint32_t lastDim_;
    uint32_t inputStride_;
    uint32_t chunksPerRun_;
    uint64_t outputDims_[MAX_RANK];
    uint64_t inputStrides_[MAX_RANK];
    event_t mte2ToSEvent_;
    event_t sToMte3Event_;
    event_t mte2ToVEvent_;
    event_t vToMte3Event_;
    event_t mte3ToMte2Event_;
};

template <typename T>
class KernelTranspose {
public:
    __aicore__ inline KernelTranspose() {}

    __aicore__ inline void Init(
        GM_ADDR input,
        GM_ADDR output,
        const TransposeTilingData& tiling)
    {
        inputAddress_ = input;
        outputAddress_ = output;
        totalElements_ = tiling.totalElements;
        rank_ = tiling.rank;
        identity_ = tiling.identity;
        fast2D_ = tiling.fast2D;
        rows_ = tiling.rows;
        cols_ = tiling.cols;
        tileCols_ = tiling.tileCols;
        contiguousElements_ = tiling.contiguousElements;
        for (uint32_t axis = 0; axis < MAX_RANK; ++axis) {
            outputDims_[axis] = tiling.outputDims[axis];
            inputStrides_[axis] = tiling.inputStrides[axis];
        }

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        elements_ = tiling.baseElementsPerBlock +
            (blockIdx < tiling.extraBlocks ? 1U : 0U);
        firstElement_ =
            blockIdx * tiling.baseElementsPerBlock +
            (blockIdx < tiling.extraBlocks
                ? blockIdx
                : tiling.extraBlocks);
        tiles_ = tiling.baseTilesPerBlock +
            (blockIdx < tiling.extraTileBlocks ? 1U : 0U);
        firstTile_ =
            blockIdx * tiling.baseTilesPerBlock +
            (blockIdx < tiling.extraTileBlocks
                ? blockIdx
                : tiling.extraTileBlocks);
        runs_ = tiling.baseRunsPerBlock +
            (blockIdx < tiling.extraRunBlocks ? 1U : 0U);
        firstRun_ =
            blockIdx * tiling.baseRunsPerBlock +
            (blockIdx < tiling.extraRunBlocks
                ? blockIdx
                : tiling.extraRunBlocks);

        inputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(input),
            totalElements_);
        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(output),
            totalElements_);
        if constexpr (std::is_same<T, half>::value) {
            if (fast2D_ != 0) {
                pipe_.InitBuffer(
                    inputTileQueue_,
                    FAST_BUFFER_NUM,
                    MATRIX_TILE_ELEMENTS * sizeof(half));
                pipe_.InitBuffer(
                    outputTileQueue_,
                    FAST_BUFFER_NUM,
                    MATRIX_TILE_ELEMENTS * sizeof(half));
                return;
            }
        }
        pipe_.InitBuffer(
            outputBuffer_,
            TILE_ELEMENTS * sizeof(T));
        sToMte3Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::S_MTE3));
        mte3ToSEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE3_S));
        mte2ToMte3Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        mte3ToMte2Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void Process()
    {
        if (identity_ != 0) {
            ProcessIdentity();
            return;
        }
        if constexpr (std::is_same<T, half>::value) {
            if (fast2D_ != 0) {
                Process2DHalf();
                return;
            }
        }
        if (contiguousElements_ > 1) {
            ProcessContiguousRuns();
            return;
        }
        AscendC::LocalTensor<T> outputLocal =
            outputBuffer_.Get<T>();
        for (uint64_t offset = 0;
             offset < elements_;
             offset += TILE_ELEMENTS) {
            const uint32_t current = static_cast<uint32_t>(
                elements_ - offset < TILE_ELEMENTS
                    ? elements_ - offset
                    : TILE_ELEMENTS);
            const uint64_t outputStart = firstElement_ + offset;
            for (uint32_t element = 0;
                 element < current;
                 ++element) {
                const uint64_t inputOffset =
                    InputOffset(outputStart + element);
                outputLocal.SetValue(
                    element,
                    inputGm_.GetValue(inputOffset));
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
    __aicore__ inline void ProcessContiguousRuns()
    {
        AscendC::LocalTensor<T> local =
            outputBuffer_.Get<T>();
        for (uint64_t runOffset = 0;
             runOffset < runs_;
             ++runOffset) {
            const uint64_t run = firstRun_ + runOffset;
            const uint64_t outputStart =
                run * contiguousElements_;
            const uint64_t inputStart =
                InputOffset(outputStart);
            for (uint64_t offset = 0;
                 offset < contiguousElements_;
                 offset += TILE_ELEMENTS) {
                const uint32_t current =
                    static_cast<uint32_t>(
                        contiguousElements_ - offset <
                                TILE_ELEMENTS
                            ? contiguousElements_ - offset
                            : TILE_ELEMENTS);
                AscendC::DataCopyExtParams params;
                params.blockCount = 1;
                params.blockLen = current * sizeof(T);
                params.srcStride = 0;
                params.dstStride = 0;
                AscendC::DataCopyPadExtParams<T> padding;
                padding.isPad = false;
                padding.leftPadding = 0;
                padding.rightPadding = 0;
                const T zero = {};
                padding.paddingValue = zero;
                AscendC::DataCopyPad(
                    local,
                    inputGm_[inputStart + offset],
                    params,
                    padding);
                AscendC::SetFlag<
                    AscendC::HardEvent::MTE2_MTE3>(
                    mte2ToMte3Event_);
                AscendC::WaitFlag<
                    AscendC::HardEvent::MTE2_MTE3>(
                    mte2ToMte3Event_);
                AscendC::DataCopyPad(
                    outputGm_[outputStart + offset],
                    local,
                    params);
                AscendC::SetFlag<
                    AscendC::HardEvent::MTE3_MTE2>(
                    mte3ToMte2Event_);
                AscendC::WaitFlag<
                    AscendC::HardEvent::MTE3_MTE2>(
                    mte3ToMte2Event_);
            }
        }
    }

    __aicore__ inline void Process2DHalf()
    {
        AscendC::GlobalTensor<half> inputHalf;
        AscendC::GlobalTensor<half> outputHalf;
        inputHalf.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(inputAddress_),
            static_cast<uint64_t>(rows_) * cols_);
        outputHalf.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(outputAddress_),
            static_cast<uint64_t>(rows_) * cols_);
        const uint16_t inputStride =
            static_cast<uint16_t>(cols_ / MATRIX_TILE - 1U);
        const uint16_t outputStride =
            static_cast<uint16_t>(rows_ / MATRIX_TILE - 1U);
        uint32_t tileRow = firstTile_ / tileCols_;
        uint32_t tileCol = firstTile_ % tileCols_;
        uint64_t inputOffset =
            static_cast<uint64_t>(tileRow) *
                MATRIX_TILE * cols_ +
            tileCol * MATRIX_TILE;
        uint64_t outputOffset =
            static_cast<uint64_t>(tileCol) *
                MATRIX_TILE * rows_ +
            tileRow * MATRIX_TILE;
        for (uint32_t offset = 0;
             offset < tiles_;
             ++offset) {
            AscendC::LocalTensor<half> inputLocal =
                inputTileQueue_.AllocTensor<half>();
            AscendC::DataCopyParams inputParams{
                MATRIX_TILE,
                1,
                inputStride,
                0};
            AscendC::DataCopy(
                inputLocal,
                inputHalf[inputOffset],
                inputParams);
            inputTileQueue_.EnQue(inputLocal);
            inputLocal =
                inputTileQueue_.DeQue<half>();
            AscendC::LocalTensor<half> outputLocal =
                outputTileQueue_.AllocTensor<half>();
            AscendC::Transpose(
                outputLocal,
                inputLocal);
            outputTileQueue_.EnQue(outputLocal);
            inputTileQueue_.FreeTensor(inputLocal);
            outputLocal =
                outputTileQueue_.DeQue<half>();
            AscendC::DataCopyParams outputParams{
                MATRIX_TILE,
                1,
                0,
                outputStride};
            AscendC::DataCopy(
                outputHalf[outputOffset],
                outputLocal,
                outputParams);
            outputTileQueue_.FreeTensor(outputLocal);
            ++tileCol;
            if (tileCol == tileCols_) {
                tileCol = 0;
                ++tileRow;
                inputOffset =
                    static_cast<uint64_t>(tileRow) *
                    MATRIX_TILE * cols_;
                outputOffset =
                    static_cast<uint64_t>(tileRow) *
                    MATRIX_TILE;
            } else {
                inputOffset += MATRIX_TILE;
                outputOffset +=
                    static_cast<uint64_t>(MATRIX_TILE) * rows_;
            }
        }
    }

    __aicore__ inline void ProcessIdentity()
    {
        AscendC::LocalTensor<T> local =
            outputBuffer_.Get<T>();
        for (uint64_t offset = 0;
             offset < elements_;
             offset += TILE_ELEMENTS) {
            const uint32_t current = static_cast<uint32_t>(
                elements_ - offset < TILE_ELEMENTS
                    ? elements_ - offset
                    : TILE_ELEMENTS);
            AscendC::DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = current * sizeof(T);
            params.srcStride = 0;
            params.dstStride = 0;
            AscendC::DataCopyPadExtParams<T> padding;
            padding.isPad = false;
            padding.leftPadding = 0;
            padding.rightPadding = 0;
            const T zero = {};
            padding.paddingValue = zero;
            AscendC::DataCopyPad(
                local,
                inputGm_[firstElement_ + offset],
                params,
                padding);
            AscendC::SetFlag<
                AscendC::HardEvent::MTE2_MTE3>(
                mte2ToMte3Event_);
            AscendC::WaitFlag<
                AscendC::HardEvent::MTE2_MTE3>(
                mte2ToMte3Event_);
            AscendC::DataCopyPad(
                outputGm_[firstElement_ + offset],
                local,
                params);
            AscendC::SetFlag<
                AscendC::HardEvent::MTE3_MTE2>(
                mte3ToMte2Event_);
            AscendC::WaitFlag<
                AscendC::HardEvent::MTE3_MTE2>(
                mte3ToMte2Event_);
        }
    }

    __aicore__ inline uint64_t InputOffset(
        uint64_t outputIndex) const
    {
        uint64_t inputOffset = 0;
        for (int32_t axis = static_cast<int32_t>(rank_) - 1;
             axis >= 0;
             --axis) {
            const uint64_t coordinate =
                outputIndex % outputDims_[axis];
            outputIndex /= outputDims_[axis];
            inputOffset += coordinate * inputStrides_[axis];
        }
        return inputOffset;
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuffer_;
    AscendC::TQue<AscendC::QuePosition::VECIN, FAST_BUFFER_NUM>
        inputTileQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, FAST_BUFFER_NUM>
        outputTileQueue_;
    AscendC::GlobalTensor<T> inputGm_;
    AscendC::GlobalTensor<T> outputGm_;
    GM_ADDR inputAddress_;
    GM_ADDR outputAddress_;
    uint64_t totalElements_;
    uint64_t firstElement_;
    uint64_t elements_;
    uint32_t rank_;
    uint32_t identity_;
    uint32_t fast2D_;
    uint32_t rows_;
    uint32_t cols_;
    uint32_t tileCols_;
    uint32_t tiles_;
    uint32_t firstTile_;
    uint64_t contiguousElements_;
    uint64_t runs_;
    uint64_t firstRun_;
    uint64_t outputDims_[MAX_RANK];
    uint64_t inputStrides_[MAX_RANK];
    event_t sToMte3Event_;
    event_t mte3ToSEvent_;
    event_t mte2ToMte3Event_;
    event_t mte3ToMte2Event_;
};

extern "C" __global__ __aicore__ void transpose(
    GM_ADDR input,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA(tilingData, tiling);
        if (tilingData.totalElements == 0) {
            return;
        }
        KernelTranspose<DTYPE_INPUTS> op;
        op.Init(input, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        const __gm__ uint32_t* fastTiling =
            reinterpret_cast<const __gm__ uint32_t*>(tiling);
        KernelTransposeFastHalf op;
        op.Init(
            input,
            output,
            fastTiling[0],
            fastTiling[1],
            fastTiling[2],
            fastTiling[3],
            fastTiling[4]);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        const __gm__ uint32_t* fastTiling =
            reinterpret_cast<const __gm__ uint32_t*>(tiling);
        KernelTransposeRotation<DTYPE_INPUTS> op;
        op.Init(
            input,
            output,
            fastTiling[0],
            fastTiling[1],
            fastTiling[2],
            fastTiling[3],
            fastTiling[4]);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        GET_TILING_DATA(tilingData, tiling);
        KernelTransposeGather<DTYPE_INPUTS> op;
        op.Init(input, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        const __gm__ uint32_t* fastTiling =
            reinterpret_cast<const __gm__ uint32_t*>(tiling);
        KernelTransposeFastHalfBatched op;
        op.Init(
            input,
            output,
            fastTiling[0],
            fastTiling[1],
            fastTiling[2],
            fastTiling[3],
            fastTiling[4]);
        op.Process();
    }
}
