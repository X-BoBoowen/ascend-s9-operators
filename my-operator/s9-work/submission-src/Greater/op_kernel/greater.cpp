#include "kernel_operator.h"


constexpr int32_t BUFFER_NUM = 1;


class KernelGreater {
public:
    __aicore__ inline KernelGreater() {}

    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR y,
        uint32_t size,
        uint32_t blockIdx)
    {
        size_ = size;
        const uint32_t offset = blockIdx * size_;
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x1) + offset, size_);
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x2) + offset, size_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y) + offset, size_);

        pipe_.InitBuffer(x1Queue_, BUFFER_NUM, size_ * sizeof(half));
        pipe_.InitBuffer(x2Queue_, BUFFER_NUM, size_ * sizeof(half));
        pipe_.InitBuffer(yQueue_, BUFFER_NUM, size_ * sizeof(int8_t));
        pipe_.InitBuffer(calcBuffer_, size_ * sizeof(half));
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
        AscendC::LocalTensor<half> x1Local = x1Queue_.AllocTensor<half>();
        AscendC::LocalTensor<half> x2Local = x2Queue_.AllocTensor<half>();
        AscendC::DataCopy(x1Local, x1Gm_, size_);
        AscendC::DataCopy(x2Local, x2Gm_, size_);
        x1Queue_.EnQue(x1Local);
        x2Queue_.EnQue(x2Local);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> x1Local = x1Queue_.DeQue<half>();
        AscendC::LocalTensor<half> x2Local = x2Queue_.DeQue<half>();
        AscendC::LocalTensor<int8_t> yLocal = yQueue_.AllocTensor<int8_t>();
        AscendC::LocalTensor<half> calc = calcBuffer_.Get<half>();

        AscendC::SubRelu(calc, x1Local, x2Local, size_);
        AscendC::Mins(calc, calc, static_cast<half>(1.0), size_);
        AscendC::Cast(yLocal, calc, AscendC::RoundMode::CAST_CEIL, size_);

        yQueue_.EnQue<int8_t>(yLocal);
        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<int8_t> yLocal = yQueue_.DeQue<int8_t>();
        AscendC::DataCopy(yGm_, yLocal, size_);
        yQueue_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x1Queue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x2Queue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> yQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuffer_;
    AscendC::GlobalTensor<half> x1Gm_;
    AscendC::GlobalTensor<half> x2Gm_;
    AscendC::GlobalTensor<int8_t> yGm_;
    uint32_t size_;
};


extern "C" __global__ __aicore__ void greater(
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelGreater op;
    op.Init(x1, x2, y, tilingData.size, AscendC::GetBlockIdx());
    op.Process();
}
