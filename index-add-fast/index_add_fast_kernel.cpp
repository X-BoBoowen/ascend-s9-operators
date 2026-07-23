#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t OUTPUT_ROWS = 32;
constexpr uint32_t ROW_WIDTH = 128;

class KernelIndexAddFast {
public:
    __aicore__ inline KernelIndexAddFast() {}

    __aicore__ inline void Init(
        GM_ADDR index,
        GM_ADDR source,
        GM_ADDR out,
        uint32_t rowsPerBlock,
        uint32_t blockIdx)
    {
        rowsPerBlock_ = rowsPerBlock;
        const uint32_t startRow = blockIdx * rowsPerBlock_;
        indexGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(index) + startRow,
            rowsPerBlock_);
        sourceGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int8_t*>(source) + startRow * ROW_WIDTH,
            rowsPerBlock_ * ROW_WIDTH);
        outGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int8_t*>(out),
            OUTPUT_ROWS * ROW_WIDTH);

        const uint32_t indexBytes = rowsPerBlock_ * sizeof(int32_t);
        const uint32_t alignedIndexBytes = (indexBytes + 31) / 32 * 32;
        pipe_.InitBuffer(indexQueue_, BUFFER_NUM, alignedIndexBytes);
        pipe_.InitBuffer(
            sourceQueue_, BUFFER_NUM, rowsPerBlock_ * ROW_WIDTH * sizeof(int8_t));
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
        AscendC::DataCopyExtParams indexCopyParams{
            1,
            static_cast<uint32_t>(rowsPerBlock_ * sizeof(int32_t)),
            0,
            0,
            0};
        AscendC::DataCopyPadExtParams<int32_t> indexPadParams{
            false, 0, 0, 0};
        AscendC::DataCopyPad(
            indexLocal, indexGm_, indexCopyParams, indexPadParams);
        AscendC::DataCopy(
            sourceLocal, sourceGm_, rowsPerBlock_ * ROW_WIDTH);
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
        for (uint32_t row = 0; row < rowsPerBlock_; ++row) {
            const int32_t outputRow = indexLocal.GetValue(row);
            if (outputRow >= 0 && outputRow < static_cast<int32_t>(OUTPUT_ROWS)) {
                AscendC::DataCopy(
                    outGm_[static_cast<uint32_t>(outputRow) * ROW_WIDTH],
                    sourceLocal[row * ROW_WIDTH],
                    ROW_WIDTH);
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
    uint32_t rowsPerBlock_;
};

extern "C" __global__ __aicore__ void index_add_fast(
    GM_ADDR self,
    GM_ADDR index,
    GM_ADDR source,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelIndexAddFast op;
    op.Init(
        index,
        source,
        out,
        tilingData.rowsPerBlock,
        AscendC::GetBlockIdx());
    op.Process();
}
