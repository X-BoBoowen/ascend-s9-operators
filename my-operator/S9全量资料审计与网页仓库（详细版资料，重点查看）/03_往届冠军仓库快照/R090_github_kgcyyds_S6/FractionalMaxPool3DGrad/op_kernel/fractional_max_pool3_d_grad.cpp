#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class BruteForce
{
public:
    __aicore__ inline BruteForce() {}
    __aicore__ inline void Init(GM_ADDR grad_output, GM_ADDR input, GM_ADDR indices, GM_ADDR out, GM_ADDR workspace, uint32_t inputSize[3], 
                                uint32_t outputSize[3], uint32_t n, uint32_t c, uint32_t bigCoreNum, uint32_t bigCoreProcessNum, uint32_t smallCoreProcessNum, TPipe *pipe)
    {
        uint32_t coreNum = GetBlockIdx();
        this->n = n;
        this->c = c;
        this->globalBufferIndex = coreNum * bigCoreProcessNum;
        for(int i = 0; i < 3; i ++) this->outputSize[i] = outputSize[i];
        if(coreNum < bigCoreNum)
            this->processNum = bigCoreProcessNum;
        else
        {
            this->processNum = smallCoreProcessNum;
            this->globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (coreNum - bigCoreNum);
        }
        s1 = (uint32_t)inputSize[0] * inputSize[1] * inputSize[2], s2 = (uint32_t)outputSize[0] * outputSize[1] * outputSize[2];
        x1Gm.SetGlobalBuffer((__gm__ T *)grad_output, n * c * s2);
        x2Gm.SetGlobalBuffer((__gm__ T *)input, n * c * s1);
        x3Gm.SetGlobalBuffer((__gm__ int32_t *)indices, n * c * s2);
        yGm.SetGlobalBuffer((__gm__ T *)out, n * c * s1);
        workGm.SetGlobalBuffer((__gm__ int32_t *)workspace, 320);
        if(coreNum == 0)
            InitGlobalMemory(yGm, n * c * s1, (T)0);
        InitGlobalMemory(workGm, 320, 0);
        pipe->InitBuffer(Qout, 1, 64);
        pipe->InitBuffer(Qwork, 1, 32 * 40);
    }
    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> work = Qwork.AllocTensor<int32_t>();
        SyncAll(workGm, work);
        Qwork.FreeTensor(work);
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>)
        {
            for(uint32_t i = 0; i < processNum; i ++)
            {
                uint32_t idx = i + globalBufferIndex;
                idx /= s2;
                uint32_t plane = idx % c;
                idx /= c;
                uint32_t batch = idx % n;
                int32_t indice = x3Gm.GetValue(i + globalBufferIndex);
                idx = batch * c * s1 + plane * s1 + indice;
                LocalTensor<T> out = Qout.AllocTensor<T>();
                out.SetValue(0, x1Gm.GetValue(i + globalBufferIndex));
                Qout.EnQue(out);
                out = Qout.DeQue<T>();
                DataCopyExtParams copyParams{1, (uint32_t)(sizeof(T)), 0, 0, 0};
                SetAtomicAdd<T>();
                DataCopyPad(yGm[idx], out, copyParams);
                SetAtomicNone();
                Qout.FreeTensor(out);
            }
        }
        else
        {
            for(uint32_t i = 0; i < n * c * s2; i ++)
            {
                int32_t indice = x3Gm.GetValue(i);
                if(indice >= globalBufferIndex && indice < globalBufferIndex + processNum)
                {
                    uint32_t idx = i;
                    idx /= s2;
                    uint32_t plane = idx % c;
                    idx /= c;
                    uint32_t batch = idx % n;
                    idx = batch * c * s1 + plane * s1 + indice;
                    LocalTensor<T> out = Qout.AllocTensor<T>();
                    out.SetValue(0, x1Gm.GetValue(i));
                    Qout.EnQue(out);
                    out = Qout.DeQue<T>();
                    DataCopyExtParams copyParams{1, (uint32_t)(sizeof(T)), 0, 0, 0};
                    SetAtomicAdd<T>();
                    DataCopyPad(yGm[idx], out, copyParams);
                    SetAtomicNone();
                    Qout.FreeTensor(out);
                }
            }
        }
    }
private:
    GlobalTensor<T> x1Gm, x2Gm, yGm;
    GlobalTensor<int32_t> x3Gm, workGm;
    TQue<QuePosition::VECIN, 1> Qwork;
    TQue<QuePosition::VECOUT, 1> Qout;
    uint32_t n, c, s1, s2, outputSize[3];
    uint32_t processNum, globalBufferIndex;
};

extern "C" __global__ __aicore__ void fractional_max_pool3_d_grad(GM_ADDR grad_output, GM_ADDR input, GM_ADDR indices, GM_ADDR random_sample, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    BruteForce<DTYPE_INPUT> op;
    op.Init(grad_output, input, indices, out, GetUserWorkspace(workspace), tiling_data.inputSize, tiling_data.outputSize, tiling_data.n,
            tiling_data.c, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
    op.Process();
}