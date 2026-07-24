#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 4;
constexpr uint32_t TILE_SIZE = 16;
constexpr uint32_t TILE_ELEMENTS = TILE_SIZE * TILE_SIZE;

class KernelTransposeFast {
public:
    __aicore__ inline KernelTransposeFast() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t rows,
        uint32_t cols,
        uint32_t tileCols,
        uint32_t baseTilesPerBlock,
        uint32_t extraBlocks,
        uint32_t blockIdx)
    {
        rows_ = rows;
        cols_ = cols;
        tileCols_ = tileCols;
        inputSrcStride_ =
            static_cast<uint16_t>(cols_ / TILE_SIZE - 1);
        outputDstStride_ =
            static_cast<uint16_t>(rows_ / TILE_SIZE - 1);
        tilesPerBlock_ =
            baseTilesPerBlock + (blockIdx < extraBlocks ? 1 : 0);
        const uint32_t firstTile =
            blockIdx * baseTilesPerBlock +
            (blockIdx < extraBlocks ? blockIdx : extraBlocks);
        tileRow_ = firstTile / tileCols_;
        tileCol_ = firstTile % tileCols_;
        inputOffset_ =
            tileRow_ * TILE_SIZE * cols_ + tileCol_ * TILE_SIZE;
        outputOffset_ =
            tileCol_ * TILE_SIZE * rows_ + tileRow_ * TILE_SIZE;
        xGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(x), rows_ * cols_);
        yGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(y), rows_ * cols_);
        pipe_.InitBuffer(
            xQueue_, BUFFER_NUM, TILE_ELEMENTS * sizeof(half));
        pipe_.InitBuffer(
            yQueue_, BUFFER_NUM, TILE_ELEMENTS * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tilesPerBlock_; ++i) {
            ProcessTile();
            AdvanceTile();
        }
    }

private:
    __aicore__ inline void ProcessTile()
    {
        AscendC::LocalTensor<half> xLocal = xQueue_.AllocTensor<half>();
        AscendC::DataCopyParams inputCopyParams{
            TILE_SIZE,
            1,
            inputSrcStride_,
            0};
        AscendC::DataCopy(xLocal, xGm_[inputOffset_], inputCopyParams);
        xQueue_.EnQue(xLocal);

        xLocal = xQueue_.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = yQueue_.AllocTensor<half>();
        AscendC::Transpose(yLocal, xLocal);
        yQueue_.EnQue(yLocal);
        xQueue_.FreeTensor(xLocal);

        yLocal = yQueue_.DeQue<half>();
        AscendC::DataCopyParams outputCopyParams{
            TILE_SIZE,
            1,
            0,
            outputDstStride_};
        AscendC::DataCopy(yGm_[outputOffset_], yLocal, outputCopyParams);
        yQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void AdvanceTile()
    {
        ++tileCol_;
        if (tileCol_ == tileCols_) {
            tileCol_ = 0;
            ++tileRow_;
            inputOffset_ = tileRow_ * TILE_SIZE * cols_;
            outputOffset_ = tileRow_ * TILE_SIZE;
        } else {
            inputOffset_ += TILE_SIZE;
            outputOffset_ += TILE_SIZE * rows_;
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> xQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> yQueue_;
    AscendC::GlobalTensor<half> xGm_;
    AscendC::GlobalTensor<half> yGm_;
    uint32_t rows_;
    uint32_t cols_;
    uint32_t tileCols_;
    uint32_t tileRow_;
    uint32_t tileCol_;
    uint32_t inputOffset_;
    uint32_t outputOffset_;
    uint32_t tilesPerBlock_;
    uint16_t inputSrcStride_;
    uint16_t outputDstStride_;
};

extern "C" __global__ __aicore__ void transpose_fast(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelTransposeFast op;
    op.Init(
        x,
        y,
        tilingData.rows,
        tilingData.cols,
        tilingData.tileCols,
        tilingData.baseTilesPerBlock,
        tilingData.extraBlocks,
        AscendC::GetBlockIdx());
    op.Process();
}
