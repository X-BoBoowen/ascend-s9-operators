#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class BruteForce
{
public:
    __aicore__ inline BruteForce() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out, uint32_t n1[3], uint32_t n2[3], uint32_t tileDataNum, uint32_t bigCoreNum, uint32_t bigCoreProcessNum, uint32_t smallCoreProcessNum, TPipe *pipe)
    {
        uint32_t coreNum = GetBlockIdx();
        this->tileDataNum = tileDataNum;
        this->globalBufferIndex = coreNum * bigCoreProcessNum;
        if(coreNum < bigCoreNum)
            this->processNum = bigCoreProcessNum;
        else
        {
            this->processNum = smallCoreProcessNum;
            this->globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (coreNum - bigCoreNum);
        }
        uint32_t sz1 = 1, sz2 = 1;
        for(int i = 0; i < 3; i ++)
        {
            this->n1[i] = n1[i];
            this->n2[i] = n2[i];
            sz1 *= n1[i];
            sz2 *= n2[i];
        }
        x1Gm.SetGlobalBuffer((__gm__ T *)input, sz1);
        x2Gm.SetGlobalBuffer((__gm__ T *)other, sz2);
        yGm.SetGlobalBuffer((__gm__ T *)out, sz1);
        pipe->InitBuffer(Qout, 1, tileDataNum * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        uint32_t loopCount = (processNum + tileDataNum - 1) / tileDataNum;
        uint32_t loop_processNum = tileDataNum;
        for(uint32_t t = 0; t < loopCount; t ++)
        {
            if(t == loopCount - 1)
                loop_processNum = processNum % tileDataNum;
            LocalTensor<T> out_tmp = Qout.AllocTensor<T>();
            for(uint32_t i = 0; i < loop_processNum; i ++)
            {
                uint32_t k = globalBufferIndex + t * tileDataNum + i;
                uint32_t idx[3];
                for(int j = 2; j >= 0; j --)
                {
                    idx[j] = k % n1[j];
                    k /= n1[j];
                }
                uint32_t idx1 = 0, idx2 = 0;
                for(int u = 0; u < 3; u ++)
                {
                    idx1 = idx1 * n1[u] + idx[u] % n1[u];
                    idx2 = idx2 * n2[u] + idx[u] % n2[u];
                }
                T x = x1Gm.GetValue(idx1);
                T l = x2Gm.GetValue(idx2);
                T res = x << l;
                if(l >= sizeof(T) * 8) res = 0;
                out_tmp.SetValue(i, res);
            }
            Qout.EnQue(out_tmp);
            LocalTensor<T> out_local = Qout.DeQue<T>();
            DataCopyExtParams copyParams{1, (uint32_t)(loop_processNum * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm[globalBufferIndex + t * tileDataNum], out_local, copyParams);
            Qout.FreeTensor(out_local);
        }
    }
private:
    GlobalTensor<T> x1Gm, x2Gm, yGm;
    TQue<QuePosition::VECOUT, 1> Qout;
    uint32_t n1[3], n2[3], globalBufferIndex, processNum, tileDataNum;
};

template <typename T>
class BruteForceSpec
{
public:
    __aicore__ inline BruteForceSpec() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out, uint32_t n1[3], uint32_t bigCoreNum, uint32_t bigCoreProcessNum, uint32_t smallCoreProcessNum, TPipe *pipe)
    {
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = coreNum * bigCoreProcessNum;
        if(coreNum < bigCoreNum)
            this->processNum = bigCoreProcessNum;
        else
        {
            this->processNum = smallCoreProcessNum;
            globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (coreNum - bigCoreNum);
        }
        this->sz1 = n1[1] * n1[2];
        this->sz2 = n1[2];
        x1Gm.SetGlobalBuffer((__gm__ T *)input + globalBufferIndex * sz1, processNum * sz1);
        x2Gm.SetGlobalBuffer((__gm__ T *)other + globalBufferIndex * sz2, processNum * sz2);
        yGm.SetGlobalBuffer((__gm__ T *)out + globalBufferIndex * sz1, processNum * sz1);
        pipe->InitBuffer(Qin1, 1, sz2 * sizeof(T));
        pipe->InitBuffer(Qin2, 1, sz2 * sizeof(T));
        pipe->InitBuffer(Qout, 1, sz2 * sizeof(T));
        pipe->InitBuffer(B1, sz2 * sizeof(int32_t));
        pipe->InitBuffer(B2, sz2 * sizeof(int32_t));
        pipe->InitBuffer(B3, 32 * sizeof(int32_t));
        pipe->InitBuffer(B4, sz2 * sizeof(int32_t));
    }
    __aicore__ inline void Process()
    {
        if constexpr (std::is_same_v<T, int64_t>)
        {
            LocalTensor<int32_t> buf3 = B3.Get<int32_t>();
            for(int i = 0, p = 1; i < 32; i ++, p *= 2)
            {
                if(i < 31)
                    buf3.SetValue(i, p);
                else
                    buf3.SetValue(i, 0);
            }
            DataCopyExtParams copyParams{1, (uint32_t)(sz2 * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
            for(uint32_t i = 0; i < processNum; i ++)
            {
                LocalTensor<T> x2 = Qin2.AllocTensor<T>();
                DataCopyPad(x2, x2Gm[i * sz2], copyParams, padParams);
                Qin2.EnQue(x2);
                x2 = Qin2.DeQue<T>();
                LocalTensor<int32_t> buf2 = B2.Get<int32_t>();
                LocalTensor<int32_t> buf4 = B4.Get<int32_t>();
                Cast(buf2, x2, RoundMode::CAST_NONE, sz2);
                Muls(buf2, buf2, (int32_t)4, sz2);
                Gather(buf4, buf3, buf2.template ReinterpretCast<uint32_t>(), 0, sz2);
                for(int j = 0; j < (sz1 / sz2); j ++)
                {
                    LocalTensor<T> x1 = Qin1.AllocTensor<T>();
                    LocalTensor<T> y = Qout.AllocTensor<T>();
                    LocalTensor<int32_t> buf1 = B1.Get<int32_t>();
                    DataCopyPad(x1, x1Gm[i * sz1 + j * sz2], copyParams, padParams);
                    Qin1.EnQue(x1);
                    x1 = Qin1.DeQue<T>();
                    Cast(buf1, x1, RoundMode::CAST_NONE, sz2);
                    Mul(buf1, buf1, buf4, sz2);
                    Cast(y, buf1, RoundMode::CAST_NONE, sz2);
                    Qout.EnQue(y);
                    y = Qout.DeQue<T>();
                    DataCopyPad(yGm[i * sz1 + j * sz2], y, copyParams);
                    Qin1.FreeTensor(x1);
                    Qout.FreeTensor(y);
                }
                Qin2.FreeTensor(x2);
            }
        }
    }  
private:
    GlobalTensor<T> x1Gm, x2Gm, yGm;
    TQue<QuePosition::VECIN, 1> Qin1, Qin2;
    TQue<QuePosition::VECOUT,1> Qout;
    TBuf<QuePosition::VECCALC> B1, B2, B3, B4;
    uint32_t processNum, sz1, sz2;
};

template <typename T>
class BitwiseKernel
{
public:
    __aicore__ inline BitwiseKernel() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out, uint32_t tileDataNum, uint32_t bigCoreNum, uint32_t bigCoreProcessNum, uint32_t smallCoreProcessNum, TPipe *pipe)
    {
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = coreNum * bigCoreProcessNum;
        this->tileDataNum = tileDataNum;
        this->limit = sizeof(T) * 8;
        if(coreNum < bigCoreNum)
            this->processNum = bigCoreProcessNum;
        else
        {
            this->processNum = smallCoreProcessNum;
            globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (coreNum - bigCoreNum);
        }
        x1Gm.SetGlobalBuffer((__gm__ T *)input + globalBufferIndex, processNum);
        x2Gm.SetGlobalBuffer((__gm__ T *)other + globalBufferIndex, processNum);
        yGm.SetGlobalBuffer((__gm__ T *)out + globalBufferIndex, processNum);
        pipe->InitBuffer(Qout, 1, tileDataNum * sizeof(T));
        if(std::is_same_v<T, int32_t>)
        {
            pipe->InitBuffer(Qin1, 1, tileDataNum * sizeof(T));
            pipe->InitBuffer(Qin2, 1, tileDataNum * sizeof(T));
            pipe->InitBuffer(B1, limit * sizeof(T));
        }
    }
    __aicore__ inline void Process()
    {
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int64_t>)
        {
            uint32_t loopCount = (processNum + tileDataNum - 1) / tileDataNum;
            loop_processNum = tileDataNum;
            for(uint32_t t = 0; t < loopCount; t ++)
            {
                if(t == loopCount - 1)
                    loop_processNum = processNum % tileDataNum;
                LocalTensor out_tmp = Qout.AllocTensor<T>();
                for(uint32_t i = 0; i < loop_processNum; i ++)
                {
                    T x = x1Gm.GetValue(i + t * tileDataNum);
                    T l = x2Gm.GetValue(i + t * tileDataNum);
                    T res = x << l;
                    if(l >= limit) res = 0;
                    out_tmp.SetValue(i, res);
                }
                Qout.EnQue(out_tmp);
                LocalTensor<T> out_local = Qout.DeQue<T>();
                DataCopyExtParams copyParams{1, (uint32_t)(loop_processNum * sizeof(T)), 0, 0, 0};
                DataCopyPad(yGm[t * tileDataNum], out_local, copyParams);
                Qout.FreeTensor(out_local);
            }
        }
        else
        {
            buf1 = B1.Get<T>();
            for(int i = 0, p = 1; i < limit; i ++, p *= 2)
            {
                if(i < limit - 1)
                    buf1.SetValue(i, (T)p);
                else
                    buf1.SetValue(i, (T)0);
            }
            uint32_t loopCount = (processNum + tileDataNum - 1) / tileDataNum;
            loop_processNum = tileDataNum;
            for(uint32_t t = 0; t < loopCount; t ++)
            {
                if(t == loopCount - 1)
                    loop_processNum = processNum % tileDataNum;
                CopyIn(t);
                Compute(t);
                CopyOut(t);
            }
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t idx)
    {
        LocalTensor<T> x1 = Qin1.AllocTensor<T>();
        LocalTensor<T> x2 = Qin2.AllocTensor<T>();
        DataCopyExtParams copyParams{1, (uint32_t)(loop_processNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyPad(x1, x1Gm[idx * tileDataNum], copyParams, padParams);
        DataCopyPad(x2, x2Gm[idx * tileDataNum], copyParams, padParams);
        Qin1.EnQue(x1);
        Qin2.EnQue(x2);
    }
    __aicore__ inline void Compute(uint32_t idx)
    {
        LocalTensor<T> x1 = Qin1.DeQue<T>();
        LocalTensor<T> x2 = Qin2.DeQue<T>();
        LocalTensor<T> y = Qout.AllocTensor<T>();
        Muls(x2, x2, (T)sizeof(T), loop_processNum);
        Gather(y, buf1, x2.template ReinterpretCast<uint32_t>(), 0, loop_processNum);
        Mul(y, x1, y, loop_processNum);
        Qin1.FreeTensor(x1);
        Qin2.FreeTensor(x2);
        Qout.EnQue(y);
    }
    __aicore__ inline void CopyOut(uint32_t idx)
    {
        LocalTensor<T> y = Qout.DeQue<T>();
        DataCopyExtParams copyParams{1, (uint32_t)(loop_processNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm[idx * tileDataNum], y, copyParams);
        Qout.FreeTensor(y);
    }
private:
    GlobalTensor<T> x1Gm, x2Gm, yGm;
    TQue<QuePosition::VECIN, 1> Qin1, Qin2;
    TQue<QuePosition::VECOUT, 1> Qout;
    TBuf<QuePosition::VECCALC> B1;
    LocalTensor<T> buf1;
    uint32_t processNum, tileDataNum, loop_processNum, limit;
};

extern "C" __global__ __aicore__ void bitwise_left_shift(GM_ADDR input, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if(TILING_KEY_IS(1))
    {
        TPipe pipe;
        BruteForce<DTYPE_INPUT> op;
        op.Init(input, other, out, tiling_data.n1, tiling_data.n2, tiling_data.tileDataNum, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
    else if(TILING_KEY_IS(2))
    {
        TPipe pipe;
        BitwiseKernel<DTYPE_INPUT> op;
        op.Init(input, other, out, tiling_data.tileDataNum, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
    else if(TILING_KEY_IS(3))
    {
        TPipe pipe;
        BruteForceSpec<DTYPE_INPUT> op;
        op.Init(input, other, out, tiling_data.n1, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
}