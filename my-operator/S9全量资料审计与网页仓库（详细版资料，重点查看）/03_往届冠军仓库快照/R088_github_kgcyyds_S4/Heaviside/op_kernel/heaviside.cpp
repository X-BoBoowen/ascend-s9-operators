#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class KernelHeaviside_0
{
public:
    __aicore__ inline KernelHeaviside_0() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR values, GM_ADDR out, uint32_t smallCoreProcessNum, uint32_t valuesNum)
    {
        this->processNum = smallCoreProcessNum;
        this->valuesNum = valuesNum;
        inputGm.SetGlobalBuffer((__gm__ T *)input, processNum);
        valuesGm.SetGlobalBuffer((__gm__ T *)values, processNum);
        outGm.SetGlobalBuffer((__gm__ T *)out, processNum);
    }
    __aicore__ inline void Process()
    {
        for(int i = 0; i < processNum; i++)
        {
            float input = static_cast<float>(inputGm.GetValue(i));
            if(input < 0)
                outGm.SetValue(i, (T)0);
            else if(input == 0)
                outGm.SetValue(i, valuesGm.GetValue(i % valuesNum));
            else
                outGm.SetValue(i, (T)1);
        }
    }

private:
    GlobalTensor<T> inputGm, valuesGm, outGm;
    uint32_t processNum, valuesNum;
};

template <typename T>
class KernelHeaviside_1
{
public:
    __aicore__ inline KernelHeaviside_1() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR values, GM_ADDR out, uint32_t bigCoreNum, uint32_t bigCoreProcessNum,
    uint32_t smallCoreProcessNum, uint32_t tileDataNum, TPipe *pipeIn)
    {
        pipe = pipeIn;
        this->tileDataNum = tileDataNum;
        this->repeatNum = 256 / sizeof(T);
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = coreNum * bigCoreProcessNum;
        if(coreNum < bigCoreNum)
            this->processNum = bigCoreProcessNum;
        else
        {
            this->processNum = smallCoreProcessNum;
            globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (coreNum - bigCoreNum);
        }
        inputGm.SetGlobalBuffer((__gm__ T *)input + globalBufferIndex, processNum);
        valuesGm.SetGlobalBuffer((__gm__ T *)values, 1);
        outGm.SetGlobalBuffer((__gm__ T *)out + globalBufferIndex, processNum);
        pipe->InitBuffer(Qin, BUFFER_NUM, tileDataNum * sizeof(T));
        pipe->InitBuffer(Qout, BUFFER_NUM, tileDataNum * sizeof(T));
        pipe->InitBuffer(B1, tileDataNum / 8);
    }
    __aicore__ inline void Process()
    {
        this->val = valuesGm.GetValue(0);
        int loopCount = (processNum + tileDataNum - 1) / tileDataNum;
        this->loop_processNum = tileDataNum;
        for(int i = 0; i < loopCount; i ++)
        {
            if(i == loopCount - 1)
                this->loop_processNum = ((processNum % tileDataNum) + repeatNum - 1) / repeatNum * repeatNum;
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
private:
    __aicore__ inline void CopyIn(int idx)
    {
        LocalTensor<T> in_local = Qin.AllocTensor<T>();
        DataCopy(in_local, inputGm[idx * tileDataNum], loop_processNum);
        Qin.EnQue(in_local);
    }
    __aicore__ inline void Compute(int idx)
    {
        LocalTensor<T> in_local = Qin.DeQue<T>();
        LocalTensor<T> out_local = Qout.AllocTensor<T>();
        LocalTensor<uint8_t> tmp1 = B1.Get<uint8_t>();
        Duplicate(out_local, (T)0, loop_processNum);
        CompareScalar(tmp1, in_local, (T)0, CMPMODE::NE, loop_processNum);
        Select(out_local, tmp1, out_local, val, SELMODE::VSEL_TENSOR_SCALAR_MODE, loop_processNum);
        CompareScalar(tmp1, in_local, (T)0, CMPMODE::LE, loop_processNum);
        Select(out_local, tmp1, out_local, (T)1, SELMODE::VSEL_TENSOR_SCALAR_MODE, loop_processNum);
        Qin.FreeTensor(in_local);
        Qout.EnQue(out_local);
    }
    __aicore__ inline void CopyOut(int idx)
    {
        LocalTensor<T> out_local = Qout.DeQue<T>();
        if(idx < (processNum + tileDataNum - 1) / tileDataNum - 1)
            DataCopy(outGm[idx * tileDataNum], out_local, tileDataNum);
        else
        {
            DataCopyExtParams copyparams{1, (uint32_t)((processNum % tileDataNum) * sizeof(T)), 0, 0, 0};
            DataCopyPad(outGm[idx * tileDataNum], out_local, copyparams);
        }
        Qout.FreeTensor(out_local);
    }
private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Qin;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Qout;
    TBuf<QuePosition::VECCALC> B1;
    GlobalTensor<T> inputGm, valuesGm, outGm;
    T val;
    uint32_t processNum, loop_processNum, tileDataNum, repeatNum;
};

extern "C" __global__ __aicore__ void heaviside(GM_ADDR input, GM_ADDR values, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    if(TILING_KEY_IS(0))
    {
        KernelHeaviside_0<DTYPE_INPUT> op;
        op.Init(input, values, out, tiling_data.smallCoreProcessNum, tiling_data.valuesNum);
        op.Process();
    }
    else if(TILING_KEY_IS(1))
    {
        TPipe pipe;
        KernelHeaviside_1<DTYPE_INPUT> op;
        op.Init(input, values, out, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, tiling_data.tileDataNum, &pipe);
        op.Process();
    }
}