#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class GluJvpKernel
{
public:
    __aicore__ inline GluJvpKernel() {}
    __aicore__ inline void Init(GM_ADDR glu_out, GM_ADDR input, GM_ADDR v, GM_ADDR jvp_out, uint32_t stride, uint32_t tileDataNum, uint32_t bigCoreNum, uint32_t bigCoreProcessNum, uint32_t smallCoreProcessNum, TPipe* pipe)
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
        // x1Gm.SetGlobalBuffer((__gm__ T *)glu_out + globalBufferIndex * stride, processNum * stride);
        x2Gm.SetGlobalBuffer((__gm__ T *)input + globalBufferIndex * stride * 2, processNum * stride * 2);
        x3Gm.SetGlobalBuffer((__gm__ T *)v + globalBufferIndex * stride * 2, processNum * stride * 2);
        yGm.SetGlobalBuffer((__gm__ T *)jvp_out + globalBufferIndex * stride, processNum * stride);
        pipe->InitBuffer(Qin_a, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_b, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_da, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_db, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qout, 1, tileDataNum * sizeof(T));
        if constexpr (!std::is_same_v<T, float>)
        {
            pipe->InitBuffer(B1, tileDataNum * 4);
            pipe->InitBuffer(B2, tileDataNum * 4);
            pipe->InitBuffer(B3, tileDataNum * 4);
            pipe->InitBuffer(B4, tileDataNum * 4);
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
            for(uint32_t i = 0; i < loopCount; i ++)
            {
                if(i == loopCount - 1 && processNum % batch)
                    loop_processNum = processNum % batch;
                CopyIn(i);
                Compute_with_cast();
                CopyOut(i);
            }
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t idx)
    {
        LocalTensor<T> a = Qin_a.AllocTensor<T>();
        LocalTensor<T> b = Qin_b.AllocTensor<T>();
        LocalTensor<T> da = Qin_da.AllocTensor<T>();
        LocalTensor<T> db = Qin_db.AllocTensor<T>();
        DataCopyExtParams copyParams{(uint16_t)loop_processNum, (uint32_t)(stride * sizeof(T)), (uint32_t)(stride * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        uint32_t start = idx * batch * stride * 2;
        DataCopyPad(a, x2Gm[start], copyParams, padParams);
        DataCopyPad(b, x2Gm[start + stride], copyParams, padParams);
        DataCopyPad(da, x3Gm[start], copyParams, padParams);
        DataCopyPad(db, x3Gm[start + stride], copyParams, padParams);
        Qin_a.EnQue(a);
        Qin_b.EnQue(b);
        Qin_da.EnQue(da);
        Qin_db.EnQue(db);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> a = Qin_a.DeQue<T>();
        LocalTensor<T> b = Qin_b.DeQue<T>();
        LocalTensor<T> da = Qin_da.DeQue<T>();
        LocalTensor<T> db = Qin_db.DeQue<T>();
        LocalTensor<T> y = Qout.AllocTensor<T>();
        uint32_t calCount = loop_processNum * align_stride;
        Sigmoid(b, b, calCount);
        Mul(a, a, b, calCount);
        Mul(da, da, b, calCount);
        Mul(db, a, db, calCount);
        Muls(b, b, (T)-1, calCount);
        Adds(b, b, (T)1, calCount);
        Mul(db, db, b, calCount);
        Add(y, da, db, calCount);
        Qout.EnQue(y);
        Qin_a.FreeTensor(a);
        Qin_b.FreeTensor(b);
        Qin_da.FreeTensor(da);
        Qin_db.FreeTensor(db);
    }
    __aicore__ inline void Compute_with_cast()
    {
        LocalTensor<T> a = Qin_a.DeQue<T>();
        LocalTensor<T> b = Qin_b.DeQue<T>();
        LocalTensor<T> da = Qin_da.DeQue<T>();
        LocalTensor<T> db = Qin_db.DeQue<T>();
        LocalTensor<T> y = Qout.AllocTensor<T>();
        uint32_t calCount = loop_processNum * align_stride;
        LocalTensor<float> buf1 = B1.Get<float>();
        LocalTensor<float> buf2 = B2.Get<float>();
        LocalTensor<float> buf3 = B3.Get<float>();
        LocalTensor<float> buf4 = B4.Get<float>();
        Cast(buf1, a, RoundMode::CAST_NONE, calCount);
        Cast(buf2, b, RoundMode::CAST_NONE, calCount);
        Cast(buf3, da, RoundMode::CAST_NONE, calCount);
        Cast(buf4, db, RoundMode::CAST_NONE, calCount);
        Sigmoid(buf2, buf2, calCount);
        Mul(buf1, buf1, buf2, calCount); // glu_result
        Mul(buf3, buf3, buf2, calCount); // da * sigmoid_b
        Mul(buf4, buf1, buf4, calCount);
        Muls(buf2, buf2, (float)-1, calCount);
        Adds(buf2, buf2, (float)1, calCount);
        Mul(buf4, buf4, buf2, calCount);
        Add(buf3, buf3, buf4, calCount);
        Cast(y, buf3, RoundMode::CAST_ROUND, calCount);
        Qout.EnQue(y);
        Qin_a.FreeTensor(a);
        Qin_b.FreeTensor(b);
        Qin_da.FreeTensor(da);
        Qin_db.FreeTensor(db);
    }
    __aicore__ inline void CopyOut(uint32_t idx)
    {
        LocalTensor<T> y = Qout.DeQue<T>();
        DataCopyExtParams copyParams{(uint16_t)loop_processNum, (uint32_t)(stride * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm[idx * batch * stride], y, copyParams);
        Qout.FreeTensor(y);
    }
private:
    GlobalTensor<T> x2Gm, x3Gm, yGm;
    TQue<QuePosition::VECIN, 1> Qin_a, Qin_b, Qin_da, Qin_db;
    TQue<QuePosition::VECOUT, 1> Qout;
    TBuf<QuePosition::VECCALC> B1, B2, B3, B4;
    uint32_t stride, batch, align_stride, tileDataNum, processNum, loop_processNum;
};

template <typename T>
class GluJvpKernelSpec
{
public:
    __aicore__ inline GluJvpKernelSpec() {}
    __aicore__ inline void Init(GM_ADDR glu_out, GM_ADDR input, GM_ADDR v, GM_ADDR jvp_out, uint32_t stride, uint32_t tileDataNum, uint32_t bigCoreNum, uint32_t bigCoreProcessNum, uint32_t smallCoreProcessNum, TPipe* pipe)
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
        x2Gm.SetGlobalBuffer((__gm__ T *)input + globalBufferIndex * stride * 2, processNum * stride * 2);
        x3Gm.SetGlobalBuffer((__gm__ T *)v + globalBufferIndex * stride * 2, processNum * stride * 2);
        yGm.SetGlobalBuffer((__gm__ T *)jvp_out + globalBufferIndex * stride, processNum * stride);
        pipe->InitBuffer(Qin_a, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_b, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_da, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qin_db, 1, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qout, 1, tileDataNum * sizeof(T));
        if constexpr (!std::is_same_v<T, float>)
        {
            pipe->InitBuffer(B1, tileDataNum * 4);
            pipe->InitBuffer(B2, tileDataNum * 4);
            pipe->InitBuffer(B3, tileDataNum * 4);
            pipe->InitBuffer(B4, tileDataNum * 4);
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
                    CopyIn(i, j);
                    Compute_with_cast();
                    CopyOut(i, j);
                }
            }
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t i, uint32_t j)
    {
        LocalTensor<T> a = Qin_a.AllocTensor<T>();
        LocalTensor<T> b = Qin_b.AllocTensor<T>();
        LocalTensor<T> da = Qin_da.AllocTensor<T>();
        LocalTensor<T> db = Qin_db.AllocTensor<T>();
        DataCopyExtParams copyParams{1, (uint32_t)(loop_processNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        uint32_t start = i * stride * 2 + j * tileDataNum;
        DataCopyPad(a, x2Gm[start], copyParams, padParams);
        DataCopyPad(b, x2Gm[start + stride], copyParams, padParams);
        DataCopyPad(da, x3Gm[start], copyParams, padParams);
        DataCopyPad(db, x3Gm[start + stride], copyParams, padParams);
        Qin_a.EnQue(a);
        Qin_b.EnQue(b);
        Qin_da.EnQue(da);
        Qin_db.EnQue(db);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> a = Qin_a.DeQue<T>();
        LocalTensor<T> b = Qin_b.DeQue<T>();
        LocalTensor<T> da = Qin_da.DeQue<T>();
        LocalTensor<T> db = Qin_db.DeQue<T>();
        LocalTensor<T> y = Qout.AllocTensor<T>();
        Sigmoid(b, b, loop_processNum);
        Mul(a, a, b, loop_processNum);
        Mul(da, da, b, loop_processNum);
        Mul(db, a, db, loop_processNum);
        Muls(b, b, (T)-1, loop_processNum);
        Adds(b, b, (T)1, loop_processNum);
        Mul(db, db, b, loop_processNum);
        Add(y, da, db, loop_processNum);
        Qout.EnQue(y);
        Qin_a.FreeTensor(a);
        Qin_b.FreeTensor(b);
        Qin_da.FreeTensor(da);
        Qin_db.FreeTensor(db);
    }
    __aicore__ inline void Compute_with_cast()
    {
        LocalTensor<T> a = Qin_a.DeQue<T>();
        LocalTensor<T> b = Qin_b.DeQue<T>();
        LocalTensor<T> da = Qin_da.DeQue<T>();
        LocalTensor<T> db = Qin_db.DeQue<T>();
        LocalTensor<T> y = Qout.AllocTensor<T>();
        LocalTensor<float> buf1 = B1.Get<float>();
        LocalTensor<float> buf2 = B2.Get<float>();
        LocalTensor<float> buf3 = B3.Get<float>();
        LocalTensor<float> buf4 = B4.Get<float>();
        Cast(buf1, a, RoundMode::CAST_NONE, loop_processNum);
        Cast(buf2, b, RoundMode::CAST_NONE, loop_processNum);
        Cast(buf3, da, RoundMode::CAST_NONE, loop_processNum);
        Cast(buf4, db, RoundMode::CAST_NONE, loop_processNum);
        Sigmoid(buf2, buf2, loop_processNum);
        Mul(buf1, buf1, buf2, loop_processNum); // glu_result
        Mul(buf3, buf3, buf2, loop_processNum); // da * sigmoid_b
        Mul(buf4, buf1, buf4, loop_processNum);
        Muls(buf2, buf2, (float)-1, loop_processNum);
        Adds(buf2, buf2, (float)1, loop_processNum);
        Mul(buf4, buf4, buf2, loop_processNum);
        Add(buf3, buf3, buf4, loop_processNum);
        Cast(y, buf3, RoundMode::CAST_ROUND, loop_processNum);
        Qout.EnQue(y);
        Qin_a.FreeTensor(a);
        Qin_b.FreeTensor(b);
        Qin_da.FreeTensor(da);
        Qin_db.FreeTensor(db);
    }
    __aicore__ inline void CopyOut(uint32_t i, uint32_t j)
    {
        LocalTensor<T> y = Qout.DeQue<T>();
        DataCopyExtParams copyParams{1, (uint32_t)(loop_processNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm[i * stride + j * tileDataNum], y, copyParams);
        Qout.FreeTensor(y);
    }
private:
    GlobalTensor<T> x2Gm, x3Gm, yGm;
    TQue<QuePosition::VECIN, 1> Qin_a, Qin_b, Qin_da, Qin_db;
    TQue<QuePosition::VECOUT, 1> Qout;
    TBuf<QuePosition::VECCALC> B1, B2, B3, B4;
    uint32_t stride, tileDataNum, processNum, loop_processNum;
};

extern "C" __global__ __aicore__ void glu_jvp(GM_ADDR glu_out, GM_ADDR input, GM_ADDR v, GM_ADDR jvp_out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if(TILING_KEY_IS(1))
    {
        GluJvpKernel<DTYPE_INPUT> op;
        op.Init(glu_out, input, v, jvp_out, tiling_data.stride, tiling_data.tileDataNum, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
    else if(TILING_KEY_IS(2))
    {
        GluJvpKernelSpec<DTYPE_INPUT> op;
        op.Init(glu_out, input, v, jvp_out, tiling_data.stride, tiling_data.tileDataNum, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
}