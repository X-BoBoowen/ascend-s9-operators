#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class GluGradJvpKernel_1
{
public:
    __aicore__ inline GluGradJvpKernel_1() {}
    __aicore__ inline void Init(GM_ADDR x_grad, GM_ADDR y_grad, GM_ADDR x, GM_ADDR v_y, GM_ADDR v_x, GM_ADDR jvp_out, uint32_t stride, uint32_t tileDataNum, uint32_t bigCoreNum, uint32_t bigCoreProcessNum, uint32_t smallCoreProcessNum, TPipe *pipe)
    {
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = coreNum * bigCoreProcessNum;
        this->stride = stride;
        this->tileDataNum = tileDataNum;
        uint32_t rep = 32 / sizeof(T);
        this->align_stride = (stride + rep - 1) / rep * rep;
        if(coreNum < bigCoreNum)
            this->processNum = bigCoreProcessNum;
        else
        {
            this->processNum = smallCoreProcessNum;
            globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (coreNum - bigCoreNum);
        }
        x0Gm.SetGlobalBuffer((__gm__ T *)x_grad + globalBufferIndex * stride * 2, processNum * stride * 2);
        x1Gm.SetGlobalBuffer((__gm__ T *)y_grad + globalBufferIndex * stride, processNum * stride);
        x2Gm.SetGlobalBuffer((__gm__ T *)x + globalBufferIndex * stride * 2, processNum * stride * 2);
        x3Gm.SetGlobalBuffer((__gm__ T *)v_y + globalBufferIndex * stride, processNum * stride);
        x4Gm.SetGlobalBuffer((__gm__ T *)v_x + globalBufferIndex * stride * 2, processNum * stride * 2);
        yGm.SetGlobalBuffer((__gm__ T *)jvp_out + globalBufferIndex * stride * 2, processNum * stride * 2);
        pipe->InitBuffer(Qin_yGrad, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_x1, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_x2, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_vy, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_vx1, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_vx2, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qout1, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qout2, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(B1, tileDataNum * 4);
        if constexpr (!std::is_same_v<T, float>)
        {
            pipe->InitBuffer(Qin_xGrad, 1, tileDataNum * sizeof(T));
            pipe->InitBuffer(B2, tileDataNum * 4);
            pipe->InitBuffer(B3, tileDataNum * 4);
            pipe->InitBuffer(B4, tileDataNum * 4);
            pipe->InitBuffer(B5, tileDataNum * 4);
            pipe->InitBuffer(B6, tileDataNum * 4);
            pipe->InitBuffer(B7, tileDataNum * 4);
            pipe->InitBuffer(B8, tileDataNum * 4);
            pipe->InitBuffer(B9, tileDataNum * 4);
            pipe->InitBuffer(B10, tileDataNum * 4);
        }
    }
    __aicore__ inline void Process()
    {
        this->batch = tileDataNum / align_stride;
        uint32_t loopCount = (processNum + batch - 1) / batch;
        loop_processNum = batch;
        if constexpr (std::is_same_v<T, float>)
        {
            for(uint32_t i = 0; i < loopCount; i ++)
            {
                if(i == loopCount - 1 && processNum % batch)
                    loop_processNum = processNum % batch;
                CopyIn(i);
                Compute();
                CopyOut(i);
            }
        }
        else
        {
            for(int i = 0; i < loopCount; i ++)
            {
                if(i == loopCount - 1 && processNum % batch)
                    loop_processNum = processNum % batch;
                CopyIn_with_cast(i);
                Compute_with_cast();
                CopyOut(i);
            }
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t idx)
    {
        LocalTensor<T> y_grad = Qin_yGrad.AllocTensor<T>();
        LocalTensor<T> x1 = Qin_x1.AllocTensor<T>();
        LocalTensor<T> x2 = Qin_x2.AllocTensor<T>();
        LocalTensor<T> v_y = Qin_vy.AllocTensor<T>();
        LocalTensor<T> v_x1 = Qin_vx1.AllocTensor<T>();
        LocalTensor<T> v_x2 = Qin_vx2.AllocTensor<T>();
        DataCopyExtParams copyParams1{(uint16_t)loop_processNum, (uint32_t)(stride * sizeof(T)), (uint32_t)(stride * sizeof(T)), 0, 0};
        DataCopyExtParams copyParams2{(uint16_t)loop_processNum, (uint32_t)(stride * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        uint32_t start = idx * batch * stride;
        DataCopyPad(y_grad, x1Gm[start], copyParams2, padParams);
        DataCopyPad(x1, x2Gm[start * 2], copyParams1, padParams);
        DataCopyPad(x2, x2Gm[start * 2 + stride], copyParams1, padParams);
        DataCopyPad(v_y, x3Gm[start], copyParams2, padParams);
        DataCopyPad(v_x1, x4Gm[start * 2], copyParams1, padParams);
        DataCopyPad(v_x2, x4Gm[start * 2 + stride], copyParams1, padParams);
        Qin_yGrad.EnQue(y_grad);
        Qin_x1.EnQue(x1);
        Qin_x2.EnQue(x2);
        Qin_vy.EnQue(v_y);
        Qin_vx1.EnQue(v_x1);
        Qin_vx2.EnQue(v_x2);
    }
    __aicore__ inline void CopyIn_with_cast(uint32_t idx)
    {
        LocalTensor<T> x_grad = Qin_xGrad.AllocTensor<T>();
        LocalTensor<T> y_grad = Qin_yGrad.AllocTensor<T>();
        LocalTensor<T> x1 = Qin_x1.AllocTensor<T>();
        LocalTensor<T> x2 = Qin_x2.AllocTensor<T>();
        LocalTensor<T> v_y = Qin_vy.AllocTensor<T>();
        LocalTensor<T> v_x1 = Qin_vx1.AllocTensor<T>();
        LocalTensor<T> v_x2 = Qin_vx2.AllocTensor<T>();
        DataCopyExtParams copyParams1{(uint16_t)loop_processNum, (uint32_t)(stride * sizeof(T)), (uint32_t)(stride * sizeof(T)), 0, 0};
        DataCopyExtParams copyParams2{(uint16_t)loop_processNum, (uint32_t)(stride * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        uint32_t start = idx * batch * stride;
        DataCopyPad(x_grad, x0Gm[start * 2], copyParams1, padParams);
        DataCopyPad(y_grad, x1Gm[start], copyParams2, padParams);
        DataCopyPad(x1, x2Gm[start * 2], copyParams1, padParams);
        DataCopyPad(x2, x2Gm[start * 2 + stride], copyParams1, padParams);
        DataCopyPad(v_y, x3Gm[start], copyParams2, padParams);
        DataCopyPad(v_x1, x4Gm[start * 2], copyParams1, padParams);
        DataCopyPad(v_x2, x4Gm[start * 2 + stride], copyParams1, padParams);
        Qin_xGrad.EnQue(x_grad);
        Qin_yGrad.EnQue(y_grad);
        Qin_x1.EnQue(x1);
        Qin_x2.EnQue(x2);
        Qin_vy.EnQue(v_y);
        Qin_vx1.EnQue(v_x1);
        Qin_vx2.EnQue(v_x2);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> y_grad = Qin_yGrad.DeQue<T>();
        LocalTensor<T> x1 = Qin_x1.DeQue<T>();
        LocalTensor<T> x2 = Qin_x2.DeQue<T>();
        LocalTensor<T> v_y = Qin_vy.DeQue<T>();
        LocalTensor<T> v_x1 = Qin_vx1.DeQue<T>();
        LocalTensor<T> v_x2 = Qin_vx2.DeQue<T>();
        LocalTensor<T> out1 = Qout1.AllocTensor<T>();
        LocalTensor<T> out2 = Qout2.AllocTensor<T>();
        LocalTensor<T> buf1 = B1.Get<T>();
        uint32_t calCount = loop_processNum * align_stride;
        Sigmoid(x2, x2, calCount); // sigmoid_b
        Muls(buf1, x2, (T)-1, calCount);
        Adds(buf1, buf1, (T)1, calCount); // 1 - sigmoid_b
        Mul(out1, x2, buf1, calCount);
        Mul(out1, out1, v_x2, calCount);
        MulAddDst(out1, v_y, x2, calCount); // first part
        Mul(out2, out1, x1, calCount); 
        Mul(out2, out2, buf1, calCount); // t * input_x[0] * (1 - sig[0])
        Mul(v_x1, v_x1, buf1, calCount);
        Mul(x1, x1, x2, calCount);
        Mul(x1, x1, buf1, calCount);
        Mul(x1, x1, v_x2, calCount);
        Sub(v_x1, v_x1, x1, calCount);
        MulAddDst(out2, v_x1, x2, calCount); // second part
        Qout1.EnQue(out1);
        Qout2.EnQue(out2);
        Qin_yGrad.FreeTensor(y_grad);
        Qin_x1.FreeTensor(x1);
        Qin_x2.FreeTensor(x2);
        Qin_vy.FreeTensor(v_y);
        Qin_vx1.FreeTensor(v_x1);
        Qin_vx2.FreeTensor(v_x2);
    }
    __aicore__ inline void Compute_with_cast()
    {
        LocalTensor<T> x_grad_ = Qin_xGrad.DeQue<T>();
        LocalTensor<T> y_grad_ = Qin_yGrad.DeQue<T>();
        LocalTensor<T> x1_ = Qin_x1.DeQue<T>();
        LocalTensor<T> x2_ = Qin_x2.DeQue<T>();
        LocalTensor<T> v_y_ = Qin_vy.DeQue<T>();
        LocalTensor<T> v_x1_ = Qin_vx1.DeQue<T>();
        LocalTensor<T> v_x2_ = Qin_vx2.DeQue<T>();
        LocalTensor<T> out1_ = Qout1.AllocTensor<T>();
        LocalTensor<T> out2_ = Qout2.AllocTensor<T>();
        LocalTensor<float> buf1 = B1.Get<float>();
        LocalTensor<float> y_grad = B2.Get<float>();
        LocalTensor<float> x1 = B3.Get<float>();
        LocalTensor<float> x2 = B4.Get<float>();
        LocalTensor<float> v_y = B5.Get<float>();
        LocalTensor<float> v_x1 = B6.Get<float>();
        LocalTensor<float> v_x2 = B7.Get<float>();
        LocalTensor<float> out1 = B8.Get<float>();
        LocalTensor<float> out2 = B9.Get<float>();
        LocalTensor<float> x_grad = B10.Get<float>();
        uint32_t calCount = loop_processNum * align_stride;
        Cast(x_grad, x_grad_, RoundMode::CAST_NONE, calCount);
        Cast(y_grad, y_grad_, RoundMode::CAST_NONE, calCount);
        Cast(x1, x1_, RoundMode::CAST_NONE, calCount);
        Cast(x2, x2_, RoundMode::CAST_NONE, calCount);
        Cast(v_y, v_y_, RoundMode::CAST_NONE, calCount);
        Cast(v_x1, v_x1_, RoundMode::CAST_NONE, calCount);
        Cast(v_x2, v_x2_, RoundMode::CAST_NONE, calCount);
        Sigmoid(x2, x2, calCount); // sigmoid_b
        Muls(buf1, x2, (float)-1, calCount);
        Adds(buf1, buf1, (float)1, calCount); // 1 - sigmoid_b
        Mul(out1, v_x2, buf1, calCount); // db * (1 - sigmoid_b)
        Mul(out1, out1, x_grad, calCount);
        MulAddDst(out1, v_y, x2, calCount); // first part
        Mul(out2, out1, x1, calCount); 
        Mul(out2, out2, buf1, calCount); // t * input_x[0] * (1 - sig[0])
        Mul(v_x1, v_x1, buf1, calCount);
        Mul(x1, x1, x2, calCount);
        Mul(x1, x1, buf1, calCount);
        Mul(x1, x1, v_x2, calCount);
        Sub(v_x1, v_x1, x1, calCount);
        MulAddDst(out2, v_x1, x_grad, calCount); // second part
        Cast(out1_, out1, RoundMode::CAST_ROUND, calCount);
        Cast(out2_, out2, RoundMode::CAST_ROUND, calCount);
        Qout1.EnQue(out1_);
        Qout2.EnQue(out2_);
        Qin_xGrad.FreeTensor(x_grad_);
        Qin_yGrad.FreeTensor(y_grad_);
        Qin_x1.FreeTensor(x1_);
        Qin_x2.FreeTensor(x2_);
        Qin_vy.FreeTensor(v_y_);
        Qin_vx1.FreeTensor(v_x1_);
        Qin_vx2.FreeTensor(v_x2_);
    }
    __aicore__ inline void CopyOut(uint32_t idx)
    {
        LocalTensor<T> out1 = Qout1.DeQue<T>();
        LocalTensor<T> out2 = Qout2.DeQue<T>();
        DataCopyExtParams copyParams{(uint16_t)loop_processNum, (uint32_t)(stride * sizeof(T)), 0, (uint32_t)(stride * sizeof(T)), 0};
        DataCopyPad(yGm[idx * batch * stride * 2], out1, copyParams);
        DataCopyPad(yGm[idx * batch * stride * 2 + stride], out2, copyParams);
        Qout1.FreeTensor(out1);
        Qout2.FreeTensor(out2);
    }
private:
    GlobalTensor<T> x0Gm, x1Gm, x2Gm, x3Gm, x4Gm, yGm;
    TQue<QuePosition::VECIN, 1> Qin_xGrad, Qin_yGrad, Qin_x1, Qin_x2, Qin_vy, Qin_vx1, Qin_vx2;
    TQue<QuePosition::VECOUT, 1> Qout1, Qout2;
    TBuf<QuePosition::VECCALC> B1, B2, B3, B4, B5, B6, B7, B8, B9, B10;
    uint32_t stride, align_stride, batch, tileDataNum, processNum, loop_processNum;
};

template <typename T>
class GluGradJvpKernel_2
{
public:
    __aicore__ inline GluGradJvpKernel_2() {}
    __aicore__ inline void Init(GM_ADDR x_grad, GM_ADDR y_grad, GM_ADDR x, GM_ADDR v_y, GM_ADDR v_x, GM_ADDR jvp_out, uint32_t stride, uint32_t tileDataNum, uint32_t bigCoreNum, uint32_t bigCoreProcessNum, uint32_t smallCoreProcessNum, TPipe *pipe)
    {
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = coreNum * bigCoreProcessNum;
        this->stride = stride;
        this->tileDataNum = tileDataNum;
        if(coreNum < bigCoreNum)
            this->processNum = bigCoreProcessNum;
        else
        {
            this->processNum = smallCoreProcessNum;
            globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (coreNum - bigCoreNum);
        }
        x0Gm.SetGlobalBuffer((__gm__ T *)x_grad + globalBufferIndex * stride * 2, processNum * stride * 2);
        x1Gm.SetGlobalBuffer((__gm__ T *)y_grad + globalBufferIndex * stride, processNum * stride);
        x2Gm.SetGlobalBuffer((__gm__ T *)x + globalBufferIndex * stride * 2, processNum * stride * 2);
        x3Gm.SetGlobalBuffer((__gm__ T *)v_y + globalBufferIndex * stride, processNum * stride);
        x4Gm.SetGlobalBuffer((__gm__ T *)v_x + globalBufferIndex * stride * 2, processNum * stride * 2);
        yGm.SetGlobalBuffer((__gm__ T *)jvp_out + globalBufferIndex * stride * 2, processNum * stride * 2);
        pipe->InitBuffer(Qin_yGrad, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_x1, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_x2, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_vy, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_vx1, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_vx2, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qout1, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qout2, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(B1, tileDataNum * 4);
        if constexpr (!std::is_same_v<T, float>)
        {
            pipe->InitBuffer(Qin_xGrad, 1, tileDataNum * sizeof(T));
            pipe->InitBuffer(B2, tileDataNum * 4);
            pipe->InitBuffer(B3, tileDataNum * 4);
            pipe->InitBuffer(B4, tileDataNum * 4);
            pipe->InitBuffer(B5, tileDataNum * 4);
            pipe->InitBuffer(B6, tileDataNum * 4);
            pipe->InitBuffer(B7, tileDataNum * 4);
            pipe->InitBuffer(B8, tileDataNum * 4);
            pipe->InitBuffer(B9, tileDataNum * 4);
            pipe->InitBuffer(B10, tileDataNum * 4);
        }
    }
    __aicore__ inline void Process()
    {
        uint32_t loopCount = (stride + tileDataNum - 1) / tileDataNum;
        if constexpr (std::is_same_v<T, float>)
        {
            for(uint32_t i = 0; i < processNum; i ++)
            {
                loop_processNum = tileDataNum;
                for(uint32_t j = 0; j < loopCount; j ++)
                {
                    if(j == loopCount - 1 && stride % tileDataNum)
                        loop_processNum = stride % tileDataNum;
                    CopyIn(i, j);
                    Compute();
                    CopyOut(i, j);
                }
            }
        }
        else
        {
            for(uint32_t i = 0; i < processNum; i ++)
            {
                loop_processNum = tileDataNum;
                for(uint32_t j = 0; j < loopCount; j ++)
                {
                    if(j == loopCount - 1 && stride % tileDataNum)
                        loop_processNum = stride % tileDataNum;
                    CopyIn_with_cast(i, j);
                    Compute_with_cast();
                    CopyOut(i, j);
                }
            }
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t i, uint32_t j)
    {
        LocalTensor<T> y_grad = Qin_yGrad.AllocTensor<T>();
        LocalTensor<T> x1 = Qin_x1.AllocTensor<T>();
        LocalTensor<T> x2 = Qin_x2.AllocTensor<T>();
        LocalTensor<T> v_y = Qin_vy.AllocTensor<T>();
        LocalTensor<T> v_x1 = Qin_vx1.AllocTensor<T>();
        LocalTensor<T> v_x2 = Qin_vx2.AllocTensor<T>();
        DataCopyExtParams copyParams{1, (uint32_t)(loop_processNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        uint32_t start1 = i * stride * 2 + j * tileDataNum;
        uint32_t start2 = i * stride + j * tileDataNum;
        DataCopyPad(y_grad, x1Gm[start2], copyParams, padParams);
        DataCopyPad(x1, x2Gm[start1], copyParams, padParams);
        DataCopyPad(x2, x2Gm[start1 + stride], copyParams, padParams);
        DataCopyPad(v_y, x3Gm[start2], copyParams, padParams);
        DataCopyPad(v_x1, x4Gm[start1], copyParams, padParams);
        DataCopyPad(v_x2, x4Gm[start1 + stride], copyParams, padParams);
        Qin_yGrad.EnQue(y_grad);
        Qin_x1.EnQue(x1);
        Qin_x2.EnQue(x2);
        Qin_vy.EnQue(v_y);
        Qin_vx1.EnQue(v_x1);
        Qin_vx2.EnQue(v_x2);
    }
    __aicore__ inline void CopyIn_with_cast(uint32_t i, uint32_t j)
    {
        LocalTensor<T> x_grad = Qin_xGrad.AllocTensor<T>();
        LocalTensor<T> y_grad = Qin_yGrad.AllocTensor<T>();
        LocalTensor<T> x1 = Qin_x1.AllocTensor<T>();
        LocalTensor<T> x2 = Qin_x2.AllocTensor<T>();
        LocalTensor<T> v_y = Qin_vy.AllocTensor<T>();
        LocalTensor<T> v_x1 = Qin_vx1.AllocTensor<T>();
        LocalTensor<T> v_x2 = Qin_vx2.AllocTensor<T>();
        DataCopyExtParams copyParams{1, (uint32_t)(loop_processNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        uint32_t start1 = i * stride * 2 + j * tileDataNum;
        uint32_t start2 = i * stride + j * tileDataNum;
        DataCopyPad(x_grad, x0Gm[start1], copyParams, padParams);
        DataCopyPad(y_grad, x1Gm[start2], copyParams, padParams);
        DataCopyPad(x1, x2Gm[start1], copyParams, padParams);
        DataCopyPad(x2, x2Gm[start1 + stride], copyParams, padParams);
        DataCopyPad(v_y, x3Gm[start2], copyParams, padParams);
        DataCopyPad(v_x1, x4Gm[start1], copyParams, padParams);
        DataCopyPad(v_x2, x4Gm[start1 + stride], copyParams, padParams);
        Qin_xGrad.EnQue(x_grad);
        Qin_yGrad.EnQue(y_grad);
        Qin_x1.EnQue(x1);
        Qin_x2.EnQue(x2);
        Qin_vy.EnQue(v_y);
        Qin_vx1.EnQue(v_x1);
        Qin_vx2.EnQue(v_x2);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> y_grad = Qin_yGrad.DeQue<T>();
        LocalTensor<T> x1 = Qin_x1.DeQue<T>();
        LocalTensor<T> x2 = Qin_x2.DeQue<T>();
        LocalTensor<T> v_y = Qin_vy.DeQue<T>();
        LocalTensor<T> v_x1 = Qin_vx1.DeQue<T>();
        LocalTensor<T> v_x2 = Qin_vx2.DeQue<T>();
        LocalTensor<T> out1 = Qout1.AllocTensor<T>();
        LocalTensor<T> out2 = Qout2.AllocTensor<T>();
        LocalTensor<T> buf1 = B1.Get<T>();
        Sigmoid(x2, x2, loop_processNum); // sigmoid_b
        Muls(buf1, x2, (T)-1, loop_processNum);
        Adds(buf1, buf1, (T)1, loop_processNum); // 1 - sigmoid_b
        Mul(out1, x2, buf1, loop_processNum);
        Mul(out1, out1, v_x2, loop_processNum);
        MulAddDst(out1, v_y, x2, loop_processNum); // first part
        Mul(out2, out1, x1, loop_processNum); 
        Mul(out2, out2, buf1, loop_processNum); // t * input_x[0] * (1 - sig[0])
        Mul(v_x1, v_x1, buf1, loop_processNum);
        Mul(x1, x1, x2, loop_processNum);
        Mul(x1, x1, buf1, loop_processNum);
        Mul(x1, x1, v_x2, loop_processNum);
        Sub(v_x1, v_x1, x1, loop_processNum);
        MulAddDst(out2, v_x1, x2, loop_processNum); // second part
        Qout1.EnQue(out1);
        Qout2.EnQue(out2);
        Qin_yGrad.FreeTensor(y_grad);
        Qin_x1.FreeTensor(x1);
        Qin_x2.FreeTensor(x2);
        Qin_vy.FreeTensor(v_y);
        Qin_vx1.FreeTensor(v_x1);
        Qin_vx2.FreeTensor(v_x2);
    }
    __aicore__ inline void Compute_with_cast()
    {
        LocalTensor<T> x_grad_ = Qin_xGrad.DeQue<T>();
        LocalTensor<T> y_grad_ = Qin_yGrad.DeQue<T>();
        LocalTensor<T> x1_ = Qin_x1.DeQue<T>();
        LocalTensor<T> x2_ = Qin_x2.DeQue<T>();
        LocalTensor<T> v_y_ = Qin_vy.DeQue<T>();
        LocalTensor<T> v_x1_ = Qin_vx1.DeQue<T>();
        LocalTensor<T> v_x2_ = Qin_vx2.DeQue<T>();
        LocalTensor<T> out1_ = Qout1.AllocTensor<T>();
        LocalTensor<T> out2_ = Qout2.AllocTensor<T>();
        LocalTensor<float> buf1 = B1.Get<float>();
        LocalTensor<float> y_grad = B2.Get<float>();
        LocalTensor<float> x1 = B3.Get<float>();
        LocalTensor<float> x2 = B4.Get<float>();
        LocalTensor<float> v_y = B5.Get<float>();
        LocalTensor<float> v_x1 = B6.Get<float>();
        LocalTensor<float> v_x2 = B7.Get<float>();
        LocalTensor<float> out1 = B8.Get<float>();
        LocalTensor<float> out2 = B9.Get<float>();
        LocalTensor<float> x_grad = B10.Get<float>();
        Cast(x_grad, x_grad_, RoundMode::CAST_NONE, loop_processNum);
        Cast(y_grad, y_grad_, RoundMode::CAST_NONE, loop_processNum);
        Cast(x1, x1_, RoundMode::CAST_NONE, loop_processNum);
        Cast(x2, x2_, RoundMode::CAST_NONE, loop_processNum);
        Cast(v_y, v_y_, RoundMode::CAST_NONE, loop_processNum);
        Cast(v_x1, v_x1_, RoundMode::CAST_NONE, loop_processNum);
        Cast(v_x2, v_x2_, RoundMode::CAST_NONE, loop_processNum);
        Sigmoid(x2, x2, loop_processNum); // sigmoid_b
        Muls(buf1, x2, (float)-1, loop_processNum);
        Adds(buf1, buf1, (float)1, loop_processNum); // 1 - sigmoid_b
        Mul(out1, v_x2, buf1, loop_processNum); // db * (1 - sigmoid_b)
        Mul(out1, out1, x_grad, loop_processNum);
        MulAddDst(out1, v_y, x2, loop_processNum); // first part
        Mul(out2, out1, x1, loop_processNum); 
        Mul(out2, out2, buf1, loop_processNum); // t * input_x[0] * (1 - sig[0])
        Mul(v_x1, v_x1, buf1, loop_processNum);
        Mul(x1, x1, x2, loop_processNum);
        Mul(x1, x1, buf1, loop_processNum);
        Mul(x1, x1, v_x2, loop_processNum);
        Sub(v_x1, v_x1, x1, loop_processNum);
        MulAddDst(out2, v_x1, x_grad, loop_processNum); // second part
        Cast(out1_, out1, RoundMode::CAST_ROUND, loop_processNum);
        Cast(out2_, out2, RoundMode::CAST_ROUND, loop_processNum);
        Qout1.EnQue(out1_);
        Qout2.EnQue(out2_);
        Qin_xGrad.FreeTensor(x_grad_);
        Qin_yGrad.FreeTensor(y_grad_);
        Qin_x1.FreeTensor(x1_);
        Qin_x2.FreeTensor(x2_);
        Qin_vy.FreeTensor(v_y_);
        Qin_vx1.FreeTensor(v_x1_);
        Qin_vx2.FreeTensor(v_x2_);
    }
    __aicore__ inline void CopyOut(uint32_t i, uint32_t j)
    {
        LocalTensor<T> out1 = Qout1.DeQue<T>();
        LocalTensor<T> out2 = Qout2.DeQue<T>();
        DataCopyExtParams copyParams{1, (uint32_t)(loop_processNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm[i * stride * 2 + j * tileDataNum], out1, copyParams);
        DataCopyPad(yGm[i * stride * 2 + j * tileDataNum + stride], out2, copyParams);
        Qout1.FreeTensor(out1);
        Qout2.FreeTensor(out2);
    }
private:
    GlobalTensor<T> x0Gm, x1Gm, x2Gm, x3Gm, x4Gm, yGm;
    TQue<QuePosition::VECIN, 1> Qin_xGrad, Qin_yGrad, Qin_x1, Qin_x2, Qin_vy, Qin_vx1, Qin_vx2;
    TQue<QuePosition::VECOUT, 1> Qout1, Qout2;
    TBuf<QuePosition::VECCALC> B1, B2, B3, B4, B5, B6, B7, B8, B9, B10;
    uint32_t stride, tileDataNum, processNum, loop_processNum;
};

extern "C" __global__ __aicore__ void glu_grad_jvp(GM_ADDR x_grad, GM_ADDR y_grad, GM_ADDR x, GM_ADDR v_y, GM_ADDR v_x, GM_ADDR jvp_out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if(TILING_KEY_IS(1))
    {
        GluGradJvpKernel_1<DTYPE_X> op;
        op.Init(x_grad, y_grad, x, v_y, v_x, jvp_out, tiling_data.stride, tiling_data.tileDataNum, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
    else if(TILING_KEY_IS(2))
    {
        GluGradJvpKernel_2<DTYPE_X> op;
        op.Init(x_grad, y_grad, x, v_y, v_x, jvp_out, tiling_data.stride, tiling_data.tileDataNum, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
}