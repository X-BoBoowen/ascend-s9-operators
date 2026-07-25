#include "kernel_operator.h"

using namespace AscendC;

constexpr uint64_t pre[65] = {0, 1ull,3ull,5ull,11ull,17ull,39ull,65ull,139ull,261ull,531ull,1025ull,2095ull,4097ull,8259ull,16405ull,32907ull,65537ull,131367ull,262145ull,524827ull,1048645ull,2098179ull,4194305ull,8390831ull,16777233ull,33558531ull,67109125ull,134225995ull,268435457ull,536887863ull,1073741825ull,2147516555ull,4294968325ull,8590000131ull,17179869265ull,34359871791ull,68719476737ull,137439215619ull,274877911045ull,549756338843ull,1099511627777ull,2199024312423ull,4398046511105ull,8796095120395ull,17592186061077ull,35184376283139ull,70368744177665ull,140737496778927ull,281474976710721ull,562949970199059ull,1125899906908165ull,2251799847243787ull,4503599627370497ull,9007199321981223ull,18014398509483025ull,36028797153190091ull,72057594038190085ull,144115188344291331ull,288230376151711745ull,576460752840837695ull,1152921504606846977ull,2305843010287435779ull,4611686018428436805ull,9223372039002292363ull};

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
        this->d1 = n1[2];
        this->d2 = n1[1] * n1[2];
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
        if(std::is_same_v<T, int64_t>)
        {
            for(uint32_t t = 0; t < loopCount; t ++)
            {
                if(t == loopCount - 1)
                    loop_processNum = processNum % tileDataNum;
                LocalTensor<T> y = Qout.AllocTensor<T>();
                for(uint32_t i = 0, k = globalBufferIndex + t * tileDataNum; i < loop_processNum; i ++, k ++)
                {
                    // uint32_t k = globalBufferIndex + t * tileDataNum + i;
                    int64_t a = x1Gm.GetValue(k);
                    int64_t b = x2Gm.GetValue(k / d2 * d1 + (k % d1));
                    if(a == 0 || b == 0)
                        y.SetValue(i, 0);
                    else
                    {
                        if(a < 0) a = -a;
                        if(b < 0) b = -b;
                        uint64_t ua = static_cast<uint64_t>(a);
                        uint64_t ub = static_cast<uint64_t>(b);
                        T shift = ScalarGetSFFValue<1>(a | b);
                        a >>= ScalarGetSFFValue<1>(a);
                        do {
                            b >>= ScalarGetSFFValue<1>(b);
                            if(a <= 64 && b <= 64){
                                a = 64 - ScalarCountLeadingZero(pre[a] & pre[b]);
                                break;
                            }
                            if (a > b) {
                                a ^= b ^= a ^= b;
                            }
                            b -= a;
                        } while(b);
                        uint64_t res = ua / (a << shift) * ub;
                        if((int64_t)res < 0) res = -res;
                        y.SetValue(i, res);
                    }
                }
                Qout.EnQue(y);
                y = Qout.DeQue<T>();
                DataCopyExtParams copyParams{1, (uint32_t)(loop_processNum * sizeof(T)), 0, 0, 0};
                DataCopyPad(yGm[globalBufferIndex + t * tileDataNum], y, copyParams);
                Qout.FreeTensor(y);
            }
        }
        else
        {
            for(uint32_t t = 0; t < loopCount; t ++)
            {
                if(t == loopCount - 1)
                    loop_processNum = processNum % tileDataNum;
                LocalTensor<T> y = Qout.AllocTensor<T>();
                for(uint32_t i = 0; i < loop_processNum; i ++)
                {
                    uint32_t k = globalBufferIndex + t * tileDataNum + i;
                    int64_t a = x1Gm.GetValue(k);
                    int64_t b = x2Gm.GetValue(k / d2 * d1 + (k % d1));
                    if(a == 0 || b == 0)
                        y.SetValue(i, 0);
                    else
                    {
                        if(a < 0) a = -a;
                        if(b < 0) b = -b;
                        int64_t res = a * b;
                        T shift = ScalarGetSFFValue<1>(a | b);
                        a >>= ScalarGetSFFValue<1>(a);
                        do {
                            b >>= ScalarGetSFFValue<1>(b);
                            if(a <= 64 && b <= 64){
                                a = 64 - ScalarCountLeadingZero(pre[a] & pre[b]);
                                break;
                            }
                            if (a > b) {
                                a ^= b ^= a ^= b;
                            }
                            b -= a;
                        } while(b);
                        res /= (a << shift);
                        y.SetValue(i, res);
                    }
                }
                Qout.EnQue(y);
                y = Qout.DeQue<T>();
                DataCopyExtParams copyParams{1, (uint32_t)(loop_processNum * sizeof(T)), 0, 0, 0};
                DataCopyPad(yGm[globalBufferIndex + t * tileDataNum], y, copyParams);
                Qout.FreeTensor(y);
            }
        }
    }
private:
    GlobalTensor<T> x1Gm, x2Gm, yGm;
    TQue<QuePosition::VECOUT, 1> Qout;
    uint32_t d1, d2, globalBufferIndex, processNum, tileDataNum;
};

template <typename T>
class LcmKernel
{
public:
    __aicore__ inline LcmKernel() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out, uint32_t tileDataNum, uint32_t bigCoreNum, uint32_t bigCoreProcessNum, uint32_t smallCoreProcessNum, TPipe *pipe)
    {
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = coreNum * bigCoreProcessNum;
        this->tileDataNum = tileDataNum;
        this->repeatNum = 256 / sizeof(T);
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
        if (std::is_same_v<T, int32_t>)
        {
            pipe->InitBuffer(Qin1, 1, tileDataNum * sizeof(T));
            pipe->InitBuffer(Qin2, 1, tileDataNum * sizeof(T));
            pipe->InitBuffer(B1, tileDataNum * sizeof(float));
            pipe->InitBuffer(B2, tileDataNum * sizeof(float));
            pipe->InitBuffer(B3, tileDataNum * sizeof(float));
            pipe->InitBuffer(B4, tileDataNum * sizeof(float));
            pipe->InitBuffer(B5, tileDataNum * sizeof(float));
            pipe->InitBuffer(B6, tileDataNum / 8);
            pipe->InitBuffer(B7, tileDataNum / 4);
            pipe->InitBuffer(B8, tileDataNum / 4);
            pipe->InitBuffer(B9, tileDataNum / 4);
        }
    }
    __aicore__ inline void Process()
    {
        loopCount = (processNum + tileDataNum - 1) / tileDataNum;
        loop_processNum = tileDataNum;
        if constexpr (std::is_same_v<T, int32_t>)
        {
            for(uint32_t i = 0; i < loopCount; i ++)
            {
                if(i == loopCount - 1)
                    loop_processNum = ((processNum % tileDataNum) + repeatNum - 1) / repeatNum * repeatNum;
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }
        }
        else
        {
            for(uint32_t t = 0; t < loopCount; t ++)
            {
                if(t == loopCount - 1)
                    loop_processNum = processNum % tileDataNum;
                LocalTensor<T> y = Qout.AllocTensor<T>();
                for(uint32_t i = 0; i < loop_processNum; i ++)
                {
                    int64_t a = x1Gm.GetValue(i + t * tileDataNum);
                    int64_t b = x2Gm.GetValue(i + t * tileDataNum);
                    if(a == 0 || b == 0)
                        y.SetValue(i, 0);
                    else
                    {
                        if(a < 0) a = -a;
                        if(b < 0) b = -b;
                        int64_t res = a * b;
                        T shift = ScalarGetSFFValue<1>(a | b);
                        a >>= ScalarGetSFFValue<1>(a);
                        do {
                            b >>= ScalarGetSFFValue<1>(b);
                            if(a <= 64 && b <= 64){
                                a = 64 - ScalarCountLeadingZero(pre[a] & pre[b]);
                                break;
                            }
                            if (a > b) {
                                a ^= b ^= a ^= b;
                            }
                            b -= a;
                        } while(b);
                        res /= (a << shift);
                        y.SetValue(i, res);
                    }
                }
                Qout.EnQue(y);
                y = Qout.DeQue<T>();
                DataCopyExtParams copyParams{1, (uint32_t)(loop_processNum * sizeof(T)), 0, 0, 0};
                DataCopyPad(yGm[t * tileDataNum], y, copyParams);
                Qout.FreeTensor(y);
            }
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t idx)
    {
        LocalTensor<T> x1 = Qin1.AllocTensor<T>();
        LocalTensor<T> x2 = Qin2.AllocTensor<T>();
        DataCopy(x1, x1Gm[idx * tileDataNum], loop_processNum);
        DataCopy(x2, x2Gm[idx * tileDataNum], loop_processNum);
        Qin1.EnQue(x1);
        Qin2.EnQue(x2);
    }
    __aicore__ inline void Compute(uint32_t idx)
    {
        LocalTensor<T> x1 = Qin1.DeQue<T>();
        LocalTensor<T> x2 = Qin2.DeQue<T>();
        LocalTensor<T> y = Qout.AllocTensor<T>();
        LocalTensor<float> buf1 = B1.Get<float>();
        LocalTensor<float> buf2 = B2.Get<float>();
        LocalTensor<float> buf3 = B3.Get<float>();
        LocalTensor<float> buf4 = B4.Get<float>();
        LocalTensor<float> buf5 = B5.Get<float>();
        LocalTensor<uint8_t> msk = B6.Get<uint8_t>();
        LocalTensor<half> buf7 = B7.Get<half>();
        LocalTensor<int16_t> buf8 = B8.Get<int16_t>();
        LocalTensor<int16_t> buf9 = B9.Get<int16_t>();
        uint32_t count = 0;
        Cast(buf1, x1, RoundMode::CAST_NONE, loop_processNum);
        Cast(buf2, x2, RoundMode::CAST_NONE, loop_processNum);
        Abs(buf1, buf1, loop_processNum);
        Abs(buf2, buf2, loop_processNum);
        Max(buf3, buf1, buf2, loop_processNum);
        Min(buf2, buf1, buf2, loop_processNum);
        Mul(buf1, buf2, buf3, loop_processNum);
        CompareScalar(msk, buf2, (float)0, CMPMODE::EQ, loop_processNum);
        Select(buf4, msk, buf3, (float)0, SELMODE::VSEL_TENSOR_SCALAR_MODE, loop_processNum);
        while(count < 12)
        {
            Fmod(buf5, buf3, buf2, loop_processNum);
            CompareScalar(msk, buf5, (float)0, CMPMODE::EQ, loop_processNum);
            Select(buf4, msk, buf2, buf4, SELMODE::VSEL_TENSOR_TENSOR_MODE, loop_processNum);
            Fmod(buf3, buf2, buf5, loop_processNum);
            CompareScalar(msk, buf3, (float)0, CMPMODE::EQ, loop_processNum);
            Select(buf4, msk, buf5, buf4, SELMODE::VSEL_TENSOR_TENSOR_MODE, loop_processNum);
            Fmod(buf2, buf5, buf3, loop_processNum);
            CompareScalar(msk, buf2, (float)0, CMPMODE::EQ, loop_processNum);
            Select(buf4, msk, buf3, buf4, SELMODE::VSEL_TENSOR_TENSOR_MODE, loop_processNum);
            count += 3;
        }
        Div(buf1, buf1, buf4, loop_processNum);
        Cast(y, buf1, RoundMode::CAST_RINT, loop_processNum);
        Qout.EnQue(y);
        Qin1.FreeTensor(x1);
        Qin2.FreeTensor(x2);
    }
    __aicore__ inline void CopyOut(uint32_t idx)
    {
        LocalTensor<T> y = Qout.DeQue<T>();
        uint32_t copysz = tileDataNum * sizeof(T);
        if(idx == loopCount - 1)
            copysz = (processNum % tileDataNum) * sizeof(T);
        DataCopyExtParams copyParams{1, copysz, 0, 0, 0};
        DataCopyPad(yGm[idx * tileDataNum], y, copyParams);
        Qout.FreeTensor(y);
    }
    __aicore__ inline uint32_t get1(LocalTensor<uint8_t> &x)
    {
        uint32_t res = 0;
        for(uint32_t i = 0; i < loop_processNum / 8; i ++)
            res += ScalarGetCountOfValue<1>(x.GetValue(i));
        return res;
    }

private:
    GlobalTensor<T> x1Gm, x2Gm, yGm;
    TQue<QuePosition::VECIN, 1> Qin1, Qin2;
    TQue<QuePosition::VECOUT, 1> Qout;
    TBuf<QuePosition::VECCALC> B1, B2, B3, B4, B5, B6, B7, B8, B9;
    uint32_t tileDataNum, processNum, loopCount, loop_processNum, repeatNum;
};

extern "C" __global__ __aicore__ void lcm(GM_ADDR input, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
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
        // BruteForce<DTYPE_INPUT> op;
        // op.Init(input, other, out, tiling_data.n1, tiling_data.n2, tiling_data.tileDataNum, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        LcmKernel<DTYPE_INPUT> op;
        op.Init(input, other, out, tiling_data.tileDataNum, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
}