#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 1;

class KernelSquareSumFast {
public:
    __aicore__ inline KernelSquareSumFast() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t outer,
        uint32_t reduceLen,
        uint32_t paddedReduce,
        uint32_t baseRowsPerBlock,
        uint32_t extraBlocks,
        uint32_t blockIdx)
    {
        outer_ = outer;
        reduceLen_ = reduceLen;
        paddedReduce_ = paddedReduce;
        rows_ = baseRowsPerBlock + (blockIdx < extraBlocks ? 1U : 0U);
        firstRow_ =
            blockIdx * baseRowsPerBlock +
            (blockIdx < extraBlocks ? blockIdx : extraBlocks);
        const uint32_t inputCount = rows_ * paddedReduce_;
        const uint32_t outputStorage = (rows_ + 15U) / 16U * 16U;
        const uint32_t sumStorage = (rows_ + 7U) / 8U * 8U;

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
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> xLocal =
            xQueue_.AllocTensor<half>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(rows_);
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
            xGm_[firstRow_ * reduceLen_],
            copyParams,
            padParams);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        const uint32_t inputCount = rows_ * paddedReduce_;
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
            static_cast<int32_t>(rows_),
            1,
            1,
            static_cast<int32_t>(paddedReduce_ / 8U));
        const AscendC::TEventID eventIdVToS =
            pipe_.FetchEventID(AscendC::HardEvent::V_S);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(
            eventIdVToS);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(
            eventIdVToS);
        for (uint32_t row = 0; row < rows_; ++row) {
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

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> yLocal =
            yQueue_.DeQue<half>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = rows_ * sizeof(half);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(
            yGm_[firstRow_],
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
    uint32_t rows_;
    uint32_t firstRow_;
};

extern "C" __global__ __aicore__ void square_sum_fast(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelSquareSumFast op;
    op.Init(
        x,
        y,
        tilingData.outer,
        tilingData.reduceLen,
        tilingData.paddedReduce,
        tilingData.baseRowsPerBlock,
        tilingData.extraBlocks,
        AscendC::GetBlockIdx());
    op.Process();
}
