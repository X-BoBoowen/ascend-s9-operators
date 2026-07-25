#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
// constexpr int32_t SIZE = 3968;

class KernelReduceLogSumExp {
public:
    __aicore__ inline KernelReduceLogSumExp() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR axes, GM_ADDR y, uint32_t dimension, uint32_t dataSize, uint32_t dataType)
    {
        this->dimension = dimension;
        this->dataType = dataType;
        this->SIZE = 3968;
        axesGm.SetGlobalBuffer((__gm__ DTYPE_AXES *)axes, 1);
        this->axes = axesGm(0);
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, dataSize);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, dataSize);


        pipe.InitBuffer(accumulateQueueFloat, BUFFER_NUM, this->SIZE * sizeof(float));
        pipe.InitBuffer(tempQueueFloat, BUFFER_NUM, this->SIZE * sizeof(float));
        pipe.InitBuffer(maxQueueFloat, BUFFER_NUM, this->SIZE * sizeof(float));

        pipe.InitBuffer(accumulateQueueHalf, BUFFER_NUM, this->SIZE * sizeof(half));
        pipe.InitBuffer(tempQueueHalf, BUFFER_NUM, this->SIZE * sizeof(half));


    }
    __aicore__ inline void Process(){
        if (this->dataType == 1 && this->dimension == 3 && this->axes == -2){
            // while(true){
            //     AscendC::printf("terst\n");
            // }
            // return;
        }
    }



private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> accumulateQueueFloat, tempQueueFloat, accumulateQueueHalf, tempQueueHalf;
    TQue<QuePosition::VECCALC, BUFFER_NUM> maxQueueFloat;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_AXES> axesGm;
    GlobalTensor<DTYPE_Y> yGm;

    uint32_t dataType; // 运行时数据类型
    int32_t axes;
    uint32_t dimension;
    int32_t SIZE;
};

extern "C" __global__ __aicore__ void reduce_log_sum_exp(GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelReduceLogSumExp op;
    op.Init(x, axes, y, tiling_data.dimension, tiling_data.dataSize, tiling_data.dataType);
    op.Process();
}