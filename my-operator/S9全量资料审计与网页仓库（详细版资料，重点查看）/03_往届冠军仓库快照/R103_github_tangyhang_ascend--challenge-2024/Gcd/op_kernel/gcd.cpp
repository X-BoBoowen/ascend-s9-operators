#include "kernel_operator.h"
using namespace AscendC;

template <typename T>
class BruteForce_1
{
public:
    __aicore__ inline BruteForce_1() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t bigCoreNum, uint32_t bigCoreProcessNum, uint32_t smallCoreProcessNum, TPipe *pipe)
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
        x1Gm.SetGlobalBuffer((__gm__ T *)x1 + globalBufferIndex, processNum);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2 + globalBufferIndex, processNum);
        yGm.SetGlobalBuffer((__gm__ T *)y + globalBufferIndex, processNum);
        pipe->InitBuffer(Qout, 1, sizeof(T));
    }
    __aicore__ inline void Process()
    {
        for(uint32_t i = 0; i < processNum; i++)
        {
            T a = x1Gm.GetValue(i);
            T b = x2Gm.GetValue(i);
            if(a < 0)
                a *= -1;
            if(b < 0)
                b *= -1;
            LocalTensor<T> out_tmp = Qout.AllocTensor<T>();
            out_tmp.SetValue(0, gcd(a, b));
            Qout.EnQue(out_tmp);
            LocalTensor<T> out_local = Qout.DeQue<T>();
            DataCopyExtParams copyparams{1, (uint32_t)sizeof(T), 0, 0, 0};
            DataCopyPad(yGm[i], out_local, copyparams);
            Qout.FreeTensor(out_local);
        }
    }
    __aicore__ inline T gcd(T a, T b)
    {
        while(b != 0)
        {
            T tmp = a % b;
            a = b;
            b = tmp;
        }
        return a;
    }
private:
    GlobalTensor<T> x1Gm, x2Gm, yGm;
    TQue<QuePosition::VECOUT, 1> Qout;
    uint32_t processNum;
};

template <typename T>
class BruteForce_2
{
public:
    __aicore__ inline BruteForce_2() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t x1dimNum, uint32_t x2dimNum, uint32_t bigCoreNum, uint32_t bigCoreProcessNum, uint32_t smallCoreProcessNum, TPipe *pipe)
    {
        uint32_t coreNum = GetBlockIdx();
        this->globalBufferIndex = coreNum * bigCoreProcessNum;
        this->broadcastNum = x2dimNum;
        if(coreNum < bigCoreNum)
            this->processNum = bigCoreProcessNum;
        else
        {
            this->processNum = smallCoreProcessNum;
            this->globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (coreNum - bigCoreNum);
        }
        x1Gm.SetGlobalBuffer((__gm__ T *)x1, x1dimNum);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2, x2dimNum);
        yGm.SetGlobalBuffer((__gm__ T *)y, x1dimNum);
        pipe->InitBuffer(Qout, 1, sizeof(T));
    }
    __aicore__ inline void Process()
    {
        for(uint32_t i = 0; i < processNum; i++)
        {
            T a = x1Gm.GetValue(i + globalBufferIndex);
            T b = x2Gm.GetValue((i + globalBufferIndex) % broadcastNum);
            if(a < 0)
                a *= -1;
            if(b < 0)
                b *= -1;
            LocalTensor<T> out_tmp = Qout.AllocTensor<T>();
            out_tmp.SetValue(0, gcd(a, b));
            Qout.EnQue(out_tmp);
            LocalTensor<T> out_local = Qout.DeQue<T>();
            DataCopyExtParams copyparams{1, (uint32_t)sizeof(T), 0, 0, 0};
            DataCopyPad(yGm[i + globalBufferIndex], out_local, copyparams);
            Qout.FreeTensor(out_local);
        }
    }
    __aicore__ inline T gcd(T a, T b)
    {
        while(b != 0)
        {
            T tmp = a % b;
            a = b;
            b = tmp;
        }
        return a;
    }
private:
    GlobalTensor<T> x1Gm, x2Gm, yGm;
    TQue<QuePosition::VECOUT, 1> Qout;
    uint32_t globalBufferIndex, broadcastNum, processNum;
};

template <typename T>
class BruteForce_3
{
public:
    __aicore__ inline BruteForce_3() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t x1dimNum, uint32_t x2dimNum, uint32_t x1dimCnt, uint32_t x2dimCnt, uint32_t x1Arr[], uint32_t x2Arr[], TPipe *pipe)
    {
        this->x1dimNum = x1dimNum;
        this->x2dimNum = x2dimNum;
        this->x1dimCnt = x1dimCnt;
        this->x2dimCnt = x2dimCnt;
        this->x1Arr = x1Arr;
        this->x2Arr = x2Arr;
        x1Gm.SetGlobalBuffer((__gm__ T *)x1, x1dimNum);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2, x2dimNum);
        yGm.SetGlobalBuffer((__gm__ T *)y, x1dimNum);
        pipe->InitBuffer(Qout, 1, sizeof(T));
    }
    __aicore__ inline void Process()
    {
        uint32_t idx[5];
        for(int i = 0; i < x1dimNum; i++)
        {
            int sz = 1;
            int offset = 0;
            for(int j = 0; j < x1dimCnt; j++)
            {
                idx[j] = (i % (sz * x1Arr[x1dimCnt - j - 1])) / sz;
                sz *= x1Arr[x1dimCnt - j - 1];
            }
            // for(int j = 0; j < x1dimCnt; j++)
            //     printf("%d ", idx[j]);
            // printf("\n");
            int sz2 = 1;
            for(int j = 0; j < x2dimCnt; j++)
            {
                if(x2Arr[x2dimCnt - j - 1] != 1)
                    offset += idx[j] * sz2;
                sz2 *= x2Arr[x2dimCnt - j - 1];
            }
            T a = x1Gm.GetValue(i);
            T b = x2Gm.GetValue(offset);
            if(a < 0)
                a *= -1;
            if(b < 0)
                b *= -1;
            LocalTensor<T> out_tmp = Qout.AllocTensor<T>();
            out_tmp.SetValue(0, gcd(a, b));
            Qout.EnQue(out_tmp);
            LocalTensor<T> out_local = Qout.DeQue<T>();
            DataCopyExtParams copyparams{1, (uint32_t)sizeof(T), 0, 0, 0};
            DataCopyPad(yGm[i], out_local, copyparams);
            Qout.FreeTensor(out_local);
        }
    }
    __aicore__ inline T gcd(T a, T b)
    {
        while(b != 0)
        {
            T tmp = a % b;
            a = b;
            b = tmp;
        }
        return a;
    }
private:
    GlobalTensor<T> x1Gm, x2Gm, yGm;
    TQue<QuePosition::VECOUT, 1> Qout;
    uint32_t x1dimNum, x2dimNum, x1dimCnt, x2dimCnt;
    uint32_t *x1Arr, *x2Arr;
};

extern "C" __global__ __aicore__ void gcd(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if(TILING_KEY_IS(1))
    {
        BruteForce_1<DTYPE_X1> op;
        op.Init(x1, x2, y, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
    else if(TILING_KEY_IS(2))
    {
        // BruteForce_2<DTYPE_X1> op;
        // op.Init(x1, x2, y, tiling_data.x1dimNum, tiling_data.x2dimNum, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        // op.Process();
        BruteForce_3<DTYPE_X1> op;
        op.Init(x1, x2, y, tiling_data.x1dimNum, tiling_data.x2dimNum, tiling_data.x1dimCnt, tiling_data.x2dimCnt, tiling_data.x1Arr, tiling_data.x2Arr, &pipe);
        op.Process();
    }
}