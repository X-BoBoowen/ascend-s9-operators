#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template<typename T>
class KernelSelectV2 {
public:
    __aicore__ inline KernelSelectV2() {}
    __aicore__ inline void Init(GM_ADDR condition, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t tileDataNum, uint32_t smallCoreDataNum, uint32_t smallCoreCarryNum, uint32_t smallCoreFinallDealNum, uint32_t dataType, uint32_t bigCoreDataNum, uint32_t bigCoreCarryNum, uint32_t bigCoreFinallDealNum, uint32_t bigCoreNum, uint32_t isBroadCast, uint32_t rows, uint32_t interval)
    {
        // 测试代码
        this->isBroadCast = isBroadCast;
        this->rows = rows;
        this->interval = interval;
        //考生补充初始化代码
        this->dataType = dataType;
        uint32_t aicoreIndex = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * aicoreIndex;
        this->tileDataNum = tileDataNum;
        if (aicoreIndex < bigCoreNum){
            this->coreDataNum = bigCoreDataNum;
            this->coreCarryTimes = bigCoreCarryNum;
            this->coreFinallDataNum = bigCoreFinallDealNum;
        }else{
            // AscendC::printf("smallCore\n");
            this->coreDataNum = smallCoreDataNum;
            this->coreCarryTimes = smallCoreCarryNum;
            this->coreFinallDataNum = smallCoreFinallDealNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (aicoreIndex - bigCoreNum);
        }
        this->globalBufferIndex = globalBufferIndex;
        conditionGm.SetGlobalBuffer((__gm__ int8_t *)condition + globalBufferIndex + this->rows * this->interval, this->coreDataNum);
        x1Gm.SetGlobalBuffer((__gm__ T *)x1 + globalBufferIndex + this->rows * this->interval, this->coreDataNum);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2 + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ T *)y + globalBufferIndex + this->rows * this->interval, this->coreDataNum);

        pipe.InitBuffer(inQueueCondition, BUFFER_NUM, this->tileDataNum * sizeof(int8_t));
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));

        pipe.InitBuffer(inQueueHalf, BUFFER_NUM, this->tileDataNum * sizeof(half));

        pipe.InitBuffer(QueueHalf, this->tileDataNum * sizeof(half));
        pipe.InitBuffer(QueueFloat, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(QueueInt32, this->tileDataNum * sizeof(int32_t));

    }
    __aicore__ inline void Process()
    {
        //考生补充对“loopCount”的定义，注意对Tiling的处理
        uint32_t loopCount = this->coreCarryTimes;
        this->processDataNum = this->tileDataNum;

        for (int32_t i = 0; i < loopCount; i++) {
        // for (int32_t i = 0; i < 1; i++) {
            if ( i == this->coreCarryTimes - 1){
                this->processDataNum = this->coreFinallDataNum;
            }
            if (this->processDataNum % 32 != 0) {
                this->processDataNum = (this->processDataNum / 32 + 1) * 32;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }

    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        //考生补充算子代码
        AscendC::LocalTensor<int8_t> conditionLocal = inQueueCondition.AllocTensor<int8_t>();
        AscendC::LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
        AscendC::LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
        AscendC::DataCopy(conditionLocal, conditionGm[progress * this->tileDataNum], this->processDataNum);
        AscendC::DataCopy(x1Local, x1Gm[progress * this->tileDataNum], this->processDataNum);
        AscendC::DataCopy(x2Local, x2Gm[progress * this->tileDataNum], this->processDataNum);

        inQueueCondition.EnQue(conditionLocal);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        //考生补充算子计算代码
        if constexpr (std::is_same_v<T, half>){
            // half
            // while(true){
            //     AscendC::printf("123\n");
            // }
            LocalTensor<int8_t> conditionLocal = inQueueCondition.DeQue<int8_t>();
            LocalTensor<half> x1Local = inQueueX1.DeQue<half>();
            LocalTensor<half> x2Local = inQueueX2.DeQue<half>();
            LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();

            auto halfLocal = QueueHalf.Get<half>();


            // 计算
            AscendC::Cast(halfLocal, conditionLocal, RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Mul(x1Local, x1Local, halfLocal, this->processDataNum);
            half scalar = -1;
            AscendC::Adds(halfLocal, halfLocal, scalar, this->processDataNum);
            AscendC::Abs(halfLocal, halfLocal, this->processDataNum);
            AscendC::Mul(x2Local, x2Local, halfLocal, this->processDataNum);
            AscendC::Add(yLocal, x1Local, x2Local, this->processDataNum);

            outQueueY.EnQue(yLocal);

            inQueueCondition.FreeTensor(conditionLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }else if constexpr (std::is_same_v<T, float>){
            // float
            // while(true){
            //     AscendC::printf("123\n");
            // }
            LocalTensor<int8_t> conditionLocal = inQueueCondition.DeQue<int8_t>();
            LocalTensor<float> x1Local = inQueueX1.DeQue<float>();
            LocalTensor<float> x2Local = inQueueX2.DeQue<float>();
            LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

            auto halfLocal = QueueHalf.Get<half>();
            auto floatLocal = QueueFloat.Get<float>();

            // 计算
            AscendC::Cast(halfLocal, conditionLocal, RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(floatLocal, halfLocal, RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Mul(x1Local, x1Local, floatLocal, this->processDataNum);
            float scalar = -1;
            AscendC::Adds(floatLocal, floatLocal, scalar, this->processDataNum);
            AscendC::Abs(floatLocal, floatLocal, this->processDataNum);
            // AscendC::Mul(x2Local, x2Local, floatLocal, this->processDataNum);
            AscendC::Add(yLocal, x1Local, x2Local, this->processDataNum);

            outQueueY.EnQue(yLocal);

            inQueueCondition.FreeTensor(conditionLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }else if constexpr (std::is_same_v<T, int32_t>){
            // int32
            // if (this->flag == 1){
            //     while(true){
            //         AscendC::printf("123\n");
            //     }
            // }
            LocalTensor<int8_t> conditionLocal = inQueueCondition.DeQue<int8_t>();
            LocalTensor<int32_t> x1Local = inQueueX1.DeQue<int32_t>();
            LocalTensor<int32_t> x2Local = inQueueX2.DeQue<int32_t>();
            LocalTensor<int32_t> yLocal = outQueueY.AllocTensor<int32_t>();

            auto halfLocal = QueueHalf.Get<half>();
            auto int32Local = QueueInt32.Get<int32_t>();


            // 计算
            AscendC::Cast(halfLocal, conditionLocal, RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(int32Local, halfLocal, RoundMode::CAST_RINT, this->processDataNum);
            AscendC::Mul(x1Local, x1Local, int32Local, this->processDataNum);
            int32_t scalar = -1;
            AscendC::Muls(int32Local, int32Local, scalar, this->processDataNum);
            scalar = 1;
            AscendC::Adds(int32Local, int32Local, scalar, this->processDataNum);
            AscendC::Mul(x2Local, x2Local, int32Local, this->processDataNum);
            AscendC::Add(yLocal, x1Local, x2Local, this->processDataNum);

            outQueueY.EnQue(yLocal);

            inQueueCondition.FreeTensor(conditionLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }else if constexpr (std::is_same_v<T, int8_t>){
            // int8
            // AscendC::printf("int8\n");
            // while(true){
            //     AscendC::printf("123\n");
            // }
            LocalTensor<int8_t> conditionLocal = inQueueCondition.DeQue<int8_t>();
            LocalTensor<int8_t> x1Local = inQueueX1.DeQue<int8_t>();
            LocalTensor<int8_t> x2Local = inQueueX2.DeQue<int8_t>();
            LocalTensor<int8_t> yLocal = outQueueY.AllocTensor<int8_t>();

            auto halfLocal = QueueHalf.Get<half>();

            LocalTensor<half> halfX1Local = inQueueHalf.AllocTensor<half>();
            LocalTensor<half> halfX2Local = inQueueHalf.AllocTensor<half>();

            // 计算
            AscendC::Cast(halfLocal, conditionLocal, RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(halfX1Local, x1Local, RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(halfX2Local, x2Local, RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Mul(halfX1Local, halfX1Local, halfLocal, this->processDataNum);
            half scalar = -1;
            AscendC::Adds(halfLocal, halfLocal, scalar, this->processDataNum);
            AscendC::Abs(halfLocal, halfLocal, this->processDataNum);
            AscendC::Mul(halfX2Local, halfX2Local, halfLocal, this->processDataNum);
            AscendC::Add(halfX2Local, halfX1Local, halfX2Local, this->processDataNum);
            AscendC::Cast(yLocal, halfX2Local, RoundMode::CAST_NONE, this->processDataNum);

            outQueueY.EnQue(yLocal);

            inQueueCondition.FreeTensor(conditionLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
            inQueueHalf.FreeTensor(halfX1Local);
            inQueueHalf.FreeTensor(halfX2Local);
        }
        
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        //考生补充算子代码
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueCondition, inQueueX1, inQueueX2, inQueueHalf;
    TBuf<QuePosition::VECCALC> QueueHalf, QueueFloat, QueueInt32;
    // TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_half, inQueueX_float, inQueueX_base_half;
    //create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<int8_t> conditionGm;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> yGm;

    //考生补充自定义成员变量
    uint32_t globalBufferIndex;
    uint32_t tileDataNum;
    uint32_t coreDataNum; // 每个核要处理的数据量
    uint32_t coreCarryTimes; // 每个核循环计算的次数
    uint32_t coreFinallDataNum; // 每个核最后处理的数据量
    uint32_t processDataNum; // 每个核每次要处理的数据量
    uint32_t dataType; // 运行时数据类型
    uint32_t isBroadCast;
    uint32_t rows;
    uint32_t interval;
    
};


extern "C" __global__ __aicore__ void select_v2(GM_ADDR condition, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    int32_t rows = tiling_data.rows;
    if (tiling_data.dataType != 3) rows = 1;
    for (int i = 0; i < rows; ++i){
        KernelSelectV2<DTYPE_X1> op;
        op.Init(condition, x1, x2, y, tiling_data.tileDataNum,tiling_data.smallCoreDataNum, tiling_data.smallCoreCarryNum, tiling_data.smallCoreFinallDealNum, tiling_data.dataType, tiling_data.bigCoreDataNum, tiling_data.bigCoreCarryNum, tiling_data.bigCoreFinallDealNum, tiling_data.bigCoreNum, tiling_data.isBroadCast, i, tiling_data.interval);
        op.Process();
    }
}