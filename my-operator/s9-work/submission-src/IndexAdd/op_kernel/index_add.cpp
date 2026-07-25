#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 1;

template <bool ALIGNED_EIGHT>
class KernelIndexAdd {
public:
    __aicore__ inline KernelIndexAdd() {}

    __aicore__ inline void Init(
        GM_ADDR index,
        GM_ADDR source,
        GM_ADDR out,
        uint32_t outputRows,
        uint32_t indexCount,
        uint32_t rowWidth,
        uint32_t baseRowsPerBlock,
        uint32_t extraBlocks,
        uint32_t blockIdx)
    {
        outputRows_ = outputRows;
        indexCount_ = indexCount;
        rowWidth_ = rowWidth;
        uint32_t startRow;
        if (ALIGNED_EIGHT) {
            rows_ = 8;
            startRow = blockIdx * 8U;
        } else {
            rows_ =
                baseRowsPerBlock + (blockIdx < extraBlocks ? 1U : 0U);
            startRow =
                blockIdx * baseRowsPerBlock +
                (blockIdx < extraBlocks ? blockIdx : extraBlocks);
        }
        indexGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(index) + startRow,
            rows_);
        sourceGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int8_t*>(source) + startRow * rowWidth_,
            rows_ * rowWidth_);
        outGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int8_t*>(out),
            outputRows_ * rowWidth_);

        pipe_.InitBuffer(
            indexQueue_,
            BUFFER_NUM,
            ((rows_ + 7U) / 8U * 8U) * sizeof(int32_t));
        pipe_.InitBuffer(
            sourceQueue_,
            BUFFER_NUM,
            rows_ * rowWidth_ * sizeof(int8_t));
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        AtomicAdd();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<int32_t> indexLocal =
            indexQueue_.AllocTensor<int32_t>();
        AscendC::LocalTensor<int8_t> sourceLocal =
            sourceQueue_.AllocTensor<int8_t>();
        if (ALIGNED_EIGHT) {
            AscendC::DataCopy(indexLocal, indexGm_, 8);
        } else {
            AscendC::DataCopyExtParams indexCopyParams;
            indexCopyParams.blockCount = 1;
            indexCopyParams.blockLen = rows_ * sizeof(int32_t);
            indexCopyParams.srcStride = 0;
            indexCopyParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<int32_t> indexPadParams;
            indexPadParams.isPad = true;
            indexPadParams.leftPadding = 0;
            indexPadParams.rightPadding =
                static_cast<uint8_t>((8U - rows_ % 8U) % 8U);
            indexPadParams.paddingValue = 0;
            AscendC::DataCopyPad(
                indexLocal,
                indexGm_,
                indexCopyParams,
                indexPadParams);
        }
        AscendC::DataCopy(
            sourceLocal, sourceGm_, rows_ * rowWidth_);
        indexQueue_.EnQue(indexLocal);
        sourceQueue_.EnQue(sourceLocal);
    }

    __aicore__ inline void AtomicAdd()
    {
        AscendC::LocalTensor<int32_t> indexLocal =
            indexQueue_.DeQue<int32_t>();
        AscendC::LocalTensor<int8_t> sourceLocal =
            sourceQueue_.DeQue<int8_t>();

        AscendC::SetAtomicAdd<int8_t>();
        for (uint32_t row = 0; row < rows_; ++row) {
            const int32_t outputRow = indexLocal.GetValue(row);
            if (outputRow >= 0 &&
                outputRow < static_cast<int32_t>(outputRows_)) {
                AscendC::DataCopy(
                    outGm_[static_cast<uint32_t>(outputRow) * rowWidth_],
                    sourceLocal[row * rowWidth_],
                    rowWidth_);
            }
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
        AscendC::SetAtomicNone();

        indexQueue_.FreeTensor(indexLocal);
        sourceQueue_.FreeTensor(sourceLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> indexQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> sourceQueue_;
    AscendC::GlobalTensor<int32_t> indexGm_;
    AscendC::GlobalTensor<int8_t> sourceGm_;
    AscendC::GlobalTensor<int8_t> outGm_;
    uint32_t outputRows_;
    uint32_t indexCount_;
    uint32_t rowWidth_;
    uint32_t rows_;
};

extern "C" __global__ __aicore__ void index_add(
    GM_ADDR self,
    GM_ADDR index,
    GM_ADDR source,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KernelIndexAdd<true> op;
        op.Init(
            index,
            source,
            out,
            tilingData.outputRows,
            tilingData.indexCount,
            tilingData.rowWidth,
            tilingData.baseRowsPerBlock,
            tilingData.extraBlocks,
            AscendC::GetBlockIdx());
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        KernelIndexAdd<false> op;
        op.Init(
            index,
            source,
            out,
            tilingData.outputRows,
            tilingData.indexCount,
            tilingData.rowWidth,
            tilingData.baseRowsPerBlock,
            tilingData.extraBlocks,
            AscendC::GetBlockIdx());
        op.Process();
    }
}
