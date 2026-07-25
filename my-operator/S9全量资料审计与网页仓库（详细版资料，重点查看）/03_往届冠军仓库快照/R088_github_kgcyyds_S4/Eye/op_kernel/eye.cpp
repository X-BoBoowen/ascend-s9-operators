#include "kernel_operator.h"
using namespace AscendC;

template <typename T>
class KernelEye
{
public:
    __aicore__ inline KernelEye() {}
    __aicore__ inline void Init(GM_ADDR y, uint32_t row, uint32_t col, uint32_t bigCoreNum, 
    uint32_t bigCoreProcessNum, uint32_t smallCoreProcessNum, TPipe *pipeIn)
    {
        pipe = pipeIn;
        this->row = row;
        this->col = col;
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = coreNum * bigCoreProcessNum;
        if(coreNum < bigCoreNum)
            this->processNum = bigCoreProcessNum;
        else
        {
            this->processNum = smallCoreProcessNum;
            globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (coreNum - bigCoreNum);
        }
        yGm.SetGlobalBuffer((__gm__ T *)y + globalBufferIndex * row * col, processNum * row * col);
        pipe->InitBuffer(QY, processNum * 32);
        // InitGlobalMemory(yGm, size, (T)0);
    }
    __aicore__ inline void Process()
    {
        if(processNum <= 4095)
        {
            LocalTensor<T> y_local = QY.Get<T>();
            DataCopyExtParams copyparams{(uint16_t)processNum, (uint32_t)(sizeof(T)), 0, (uint32_t)(row * col * sizeof(T) - sizeof(T)), 0};
            for(int i = 0; i < processNum; i++)
                y_local.SetValue(i * (32 / sizeof(T)), (T)1);
            for(int i = 0; i < row && i < col; i ++)
            {
                DataCopyPad(yGm[i * col + i], y_local, copyparams);
            }
        }
        else
        {
            int offset = 0;
            for(int i = 0; i < processNum; i++)
            {
                int count = 0;
                while(count < row && count < col)
                {
                    yGm.SetValue(offset + count * col + count, (T)1);
                    count ++;
                }
                offset += col * row;
            }
        }
    }

private:
    TPipe *pipe;
    TBuf<QuePosition::VECCALC> QY;
    GlobalTensor<T> yGm;
    uint32_t row, col;
    uint32_t processNum;
};

extern "C" __global__ __aicore__ void eye(GM_ADDR y, GM_ADDR y_ref, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    int dtype = tiling_data.dtype;
    if(dtype == 0)
    {
        KernelEye<float> op;
        op.Init(y_ref, tiling_data.row, tiling_data.col, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
    else if(dtype == 1)
    {
        KernelEye<half> op;
        op.Init(y_ref, tiling_data.row, tiling_data.col, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
    else if(dtype == 2)
    {
        KernelEye<double> op;
        op.Init(y_ref, tiling_data.row, tiling_data.col, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
    else
    {
        KernelEye<int32_t> op;
        op.Init(y_ref, tiling_data.row, tiling_data.col, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
}