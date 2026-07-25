#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t ALGIN = 32;
// constexpr int32_t SIZE = 3968;

class KernelSort {
public:
    __aicore__ inline KernelSort() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR y1, GM_ADDR y2, uint32_t dataType, uint32_t dimension, uint32_t dataSize,
        int32_t axis, bool descending, bool stable)
    {
        this->dimension = dimension;
        this->dataType = dataType;
        this->dataSize = dataSize;
        this->axis = axis;
        this->descending = descending;
        this->stable = stable;
        // AscendC::printf("axis = %d, descending = %d, stable = %d\n", this->axis, this->descending, this->stable);
        this->SIZE = 3968;
        inputGm.SetGlobalBuffer((__gm__ DTYPE_INPUT *)input, this->dataSize);
        y1Gm.SetGlobalBuffer((__gm__ DTYPE_Y1 *)y1, this->dataSize);
        y2Gm.SetGlobalBuffer((__gm__ DTYPE_Y2 *)y2, this->dataSize);



        // pipe.InitBuffer(tempBits, this->SIZE * sizeof(uint8_t));

        // pipe.InitBuffer(accumulateQueueFloat, BUFFER_NUM, this->SIZE * sizeof(float));
        // pipe.InitBuffer(tempQueueFloat, BUFFER_NUM, this->SIZE * sizeof(float));
        // pipe.InitBuffer(maxQueueFloat, BUFFER_NUM, this->SIZE * sizeof(float));

        // pipe.InitBuffer(tempQueueHalf, BUFFER_NUM, this->SIZE * sizeof(half));

        // pipe.InitBuffer(tempQueueInt32, BUFFER_NUM, this->SIZE * sizeof(int32_t));
        // pipe.InitBuffer(tempQueueInt8, BUFFER_NUM, this->SIZE * sizeof(int8_t));
        // pipe.InitBuffer(tempQueueInt32T, BUFFER_NUM, this->SIZE * sizeof(int32_t));

        int32_t typeSize = 2;
        if (dataType == 0){
            typeSize = 4;
        }
        int32_t elementsPerBlock = 32 / typeSize;
        int32_t elementsPerRepeat = 256 / typeSize;
        int32_t firstMaxRepeat = this->SIZE / elementsPerRepeat;
        int32_t finalWorkLocalNeedSize =  (firstMaxRepeat + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
        // pipe.InitBuffer(workQueue, BUFFER_NUM, finalWorkLocalNeedSize * sizeof(float));

    }
    __aicore__ inline void Process(){
        // && this->axis == -2 && this->descending == false
        if (this->dataType == 0 && this->dimension == 4 && this->axis == -2){
            while(true) AscendC::printf("test\n");
        }
    }



private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> accumulateQueueFloat, tempQueueFloat, tempQueueInt32T, tempQueueHalf, tempQueueInt32, tempQueueInt8;
    TQue<QuePosition::VECCALC, BUFFER_NUM> workQueue, maxQueueFloat;
    GlobalTensor<DTYPE_INPUT> inputGm;
    GlobalTensor<DTYPE_Y1> y1Gm;
    GlobalTensor<DTYPE_Y2> y2Gm;

    uint32_t dataType; // 运行时数据类型
    int32_t axis;
    uint32_t dimension;
    int32_t SIZE;
    uint32_t dataSize;
    bool stable;
    bool descending;
};

extern "C" __global__ __aicore__ void sort(GM_ADDR input, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelSort op;
    op.Init(input, y1, y2, tiling_data.dataType, tiling_data.dimension, tiling_data.dataSize, tiling_data.axis, tiling_data.descending, tiling_data.stable);
    op.Process();
}