#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 2;

class KernelSquareSumV1 {
public:
    __aicore__ inline KernelSquareSumV1() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t outer,
        uint32_t reduceLen,
        uint32_t paddedReduce,
        uint32_t baseRowsPerBlock,
        uint32_t extraBlocks,
        uint32_t tileRows,
        uint32_t blockIdx)
    {
        outer_ = outer;
        reduceLen_ = reduceLen;
        paddedReduce_ = paddedReduce;
        tileRows_ = tileRows == 0U ? 1U : tileRows;
        rows_ = baseRowsPerBlock + (blockIdx < extraBlocks ? 1U : 0U);
        firstRow_ =
            blockIdx * baseRowsPerBlock +
            (blockIdx < extraBlocks ? blockIdx : extraBlocks);

        // Buffers are sized by micro-tile, not the full per-core row span.
        const uint32_t inputCount = tileRows_ * paddedReduce_;
        const uint32_t outputStorage = (tileRows_ + 15U) / 16U * 16U;
        const uint32_t sumStorage = (tileRows_ + 7U) / 8U * 8U;

        xGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(x), outer_ * reduceLen_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y), outer_);

        pipe_.InitBuffer(
            xQueue_, BUFFER_NUM, inputCount * sizeof(half));
        pipe_.InitBuffer(
            yQueue_, BUFFER_NUM, outputStorage * sizeof(half));
        pipe_.InitBuffer(
            xFloatBuffer_, inputCount * sizeof(float));
        pipe_.InitBuffer(
            sumFloatBuffer_, sumStorage * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (rows_ == 0U) {
            return;
        }

        // Double-buffer pipeline:
        //   prefetch tile0
        //   for each tile: prefetch next | compute current | copy-out current
        const uint32_t firstRows =
            (rows_ < tileRows_) ? rows_ : tileRows_;
        CopyIn(0U, firstRows);

        for (uint32_t rowOffset = 0U; rowOffset < rows_; rowOffset += tileRows_) {
            const uint32_t currentRows =
                (rows_ - rowOffset < tileRows_)
                    ? (rows_ - rowOffset)
                    : tileRows_;
            const uint32_t nextOffset = rowOffset + tileRows_;
            if (nextOffset < rows_) {
                const uint32_t nextRows =
                    (rows_ - nextOffset < tileRows_)
                        ? (rows_ - nextOffset)
                        : tileRows_;
                CopyIn(nextOffset, nextRows);
            }
            Compute(currentRows);
            CopyOut(rowOffset, currentRows);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t rowOffset, uint32_t currentRows)
    {
        AscendC::LocalTensor<half> xLocal =
            xQueue_.AllocTensor<half>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(currentRows);
        copyParams.blockLen = reduceLen_ * sizeof(half);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        AscendC::DataCopyPadExtParams<half> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding =
            static_cast<uint8_t>(paddedReduce_ - reduceLen_);
        padParams.paddingValue = static_cast<half>(0.0);
        AscendC::DataCopyPad(
            xLocal,
            xGm_[(firstRow_ + rowOffset) * reduceLen_],
            copyParams,
            padParams);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t currentRows)
    {
        const uint32_t inputCount = currentRows * paddedReduce_;
        AscendC::LocalTensor<half> xHalf =
            xQueue_.DeQue<half>();
        AscendC::LocalTensor<float> xFloat =
            xFloatBuffer_.Get<float>();
        AscendC::LocalTensor<float> sumFloat =
            sumFloatBuffer_.Get<float>();
        AscendC::LocalTensor<half> yLocal =
            yQueue_.AllocTensor<half>();

        AscendC::Mul(xHalf, xHalf, xHalf, inputCount);
        AscendC::Cast(
            xFloat,
            xHalf,
            AscendC::RoundMode::CAST_NONE,
            inputCount);
        AscendC::WholeReduceSum<float>(
            sumFloat,
            xFloat,
            static_cast<int32_t>(reduceLen_),
            static_cast<int32_t>(currentRows),
            1,
            1,
            static_cast<int32_t>(paddedReduce_ / 8U));
        const AscendC::TEventID eventIdVToS =
            pipe_.FetchEventID(AscendC::HardEvent::V_S);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(
            eventIdVToS);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(
            eventIdVToS);
        for (uint32_t row = 0; row < currentRows; ++row) {
            yLocal.SetValue(
                row,
                static_cast<half>(sumFloat.GetValue(row)));
        }
        const AscendC::TEventID eventIdSToMte3 =
            pipe_.FetchEventID(AscendC::HardEvent::S_MTE3);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(
            eventIdSToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(
            eventIdSToMte3);

        yQueue_.EnQue(yLocal);
        xQueue_.FreeTensor(xHalf);
    }

    __aicore__ inline void CopyOut(uint32_t rowOffset, uint32_t currentRows)
    {
        AscendC::LocalTensor<half> yLocal =
            yQueue_.DeQue<half>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = currentRows * sizeof(half);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(
            yGm_[firstRow_ + rowOffset],
            yLocal,
            copyParams);
        yQueue_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> xQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> yQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFloatBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumFloatBuffer_;
    AscendC::GlobalTensor<half> xGm_;
    AscendC::GlobalTensor<half> yGm_;
    uint32_t outer_;
    uint32_t reduceLen_;
    uint32_t paddedReduce_;
    uint32_t tileRows_;
    uint32_t rows_;
    uint32_t firstRow_;
};

extern "C" __global__ __aicore__ void square_sum_v1(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelSquareSumV1 op;
    op.Init(
        x,
        y,
        tilingData.outer,
        tilingData.reduceLen,
        tilingData.paddedReduce,
        tilingData.baseRowsPerBlock,
        tilingData.extraBlocks,
        tilingData.tileRows,
        AscendC::GetBlockIdx());
    op.Process();
}
