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

        pipe.InitBuffer(temp1Float, 8 * sizeof(float)); 
        pipe.InitBuffer(temp2Float, 8 * sizeof(float));

        pipe.InitBuffer(tempBits, this->SIZE * sizeof(uint8_t));

        pipe.InitBuffer(accumulateQueueFloat, BUFFER_NUM, this->SIZE * sizeof(float));
        pipe.InitBuffer(tempQueueFloat, BUFFER_NUM, this->SIZE * sizeof(float));
        pipe.InitBuffer(maxQueueFloat, BUFFER_NUM, this->SIZE * sizeof(float));

        pipe.InitBuffer(accumulateQueueHalf, BUFFER_NUM, this->SIZE * sizeof(half));
        pipe.InitBuffer(tempQueueHalf, BUFFER_NUM, this->SIZE * sizeof(half));

        int32_t typeSize = 2;
        if (dataType == 0){
            typeSize = 4;
        }
        int32_t elementsPerBlock = 32 / typeSize;
        int32_t elementsPerRepeat = 256 / typeSize;
        int32_t firstMaxRepeat = this->SIZE / elementsPerRepeat;
        int32_t finalWorkLocalNeedSize =  (firstMaxRepeat + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
        pipe.InitBuffer(workQueue, BUFFER_NUM, finalWorkLocalNeedSize * sizeof(float));

    }
    // __aicore__ inline void Process(){
    //     if (this->dataType == 1 && this->dimension == 3 && this->axes == -2){
    //         // while(true){
    //         //     AscendC::printf("terst\n");
    //         // }
    //         // return;
    //     }
    // }
    __aicore__ inline void Process(uint32_t inputShape[4])
    {
        LocalTensor<float> temp1 = temp1Float.Get<float>();  
        LocalTensor<float> temp2 = temp2Float.Get<float>();
        LocalTensor<uint8_t> bits = tempBits.Get<uint8_t>();
        if (this->dataType == 1 && this->dimension == 1){
            // return;
            uint32_t all_size = inputShape[0];
            uint32_t loop = all_size / this->SIZE;
            uint32_t remain = all_size % this->SIZE;
            AscendC::LocalTensor<float> workLocal = workQueue.AllocTensor<float>();
            AscendC::LocalTensor<DTYPE_Y> tempHalfLocal = tempQueueHalf.AllocTensor<DTYPE_Y>();
            AscendC::LocalTensor<float> tempFloatLocal = tempQueueFloat.AllocTensor<float>();
            float sum = 0;
            for (int i = 0; i < loop; ++i){
                AscendC::DataCopy(tempHalfLocal, xGm[i * this->SIZE], this->SIZE);
                tempQueueHalf.EnQue(tempHalfLocal);
                tempHalfLocal = tempQueueHalf.DeQue<DTYPE_Y>();
                AscendC::Cast(tempFloatLocal, tempHalfLocal, AscendC::RoundMode::CAST_NONE, this->SIZE);
                AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
                AscendC::ReduceSum(tempFloatLocal, tempFloatLocal, workLocal, this->SIZE);
                float t = static_cast<float>(tempFloatLocal.GetValue(0));
                sum += t;
            }
            if (remain){
                AscendC::DataCopy(tempHalfLocal, xGm[loop * this->SIZE], this->SIZE);
                tempQueueHalf.EnQue(tempHalfLocal);
                tempHalfLocal = tempQueueHalf.DeQue<DTYPE_Y>();
                AscendC::Cast(tempFloatLocal, tempHalfLocal, AscendC::RoundMode::CAST_NONE, this->SIZE);
                AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
                for (int i = 0; i < remain; ++i){
                    float t = static_cast<float>(tempFloatLocal.GetValue(i));
                    sum += t;
                }
            }
            AscendC::Duplicate<float>(tempFloatLocal, static_cast<float>(sum), 32);
            AscendC::Ln(tempFloatLocal, tempFloatLocal, 32);
            // 4、写回到GM中去
            AscendC::Cast(tempHalfLocal, tempFloatLocal, AscendC::RoundMode::CAST_NONE, 32);
            tempQueueHalf.EnQue(tempHalfLocal);
            tempHalfLocal = tempQueueHalf.DeQue<DTYPE_Y>();
            // AscendC::printf("yyyyy = %f\n", accumulateLocalHalf.GetValue(0)); // 不知道是不是硬件问题，没有这句话结果是错的，有这句话结果是对的
            // accumulateLocalHalf.GetValue(0);
            AscendC::DataCopy(yGm[0], tempHalfLocal, 32); //
            
            workQueue.FreeTensor(workLocal);
            tempQueueHalf.FreeTensor(tempHalfLocal);
            tempQueueFloat.FreeTensor(tempFloatLocal);
        }else if (this->dataType == 1 && this->dimension == 3){
            if (this->axes == 1 || this->axes == -2){
                uint32_t all_size = inputShape[2];
                uint32_t loop = all_size / this->SIZE;
                uint32_t remain = all_size % this->SIZE;
                if (remain != 0) loop++;
                for (int k = 0; k < inputShape[0]; ++k){
                    for (int32_t i = 0; i < loop; ++i){
                        AscendC::LocalTensor<float> maxLocal = maxQueueFloat.AllocTensor<float>();
                        AscendC::Duplicate<float>(maxLocal, static_cast<float>(0), this->SIZE);

                        // AscendC::LocalTensor<DTYPE_Y> tempHalfLocal = tempQueueHalf.AllocTensor<DTYPE_Y>();
                        // AscendC::LocalTensor<float> tempFloatLocal = tempQueueFloat.AllocTensor<float>();

                        // AscendC::LocalTensor<float> accumulateLocal = accumulateQueueFloat.AllocTensor<float>();
                        // AscendC::LocalTensor<DTYPE_Y> accumulateLocalHalf = accumulateQueueHalf.AllocTensor<DTYPE_Y>();
                        for (int32_t j = 0; j < inputShape[1]; ++j){
                            // 1、拷贝数据
                            AscendC::LocalTensor<DTYPE_Y> tempHalfLocal = tempQueueHalf.AllocTensor<DTYPE_Y>();
                            AscendC::DataCopy(tempHalfLocal, xGm[i * SIZE + j * all_size + k * inputShape[1] * inputShape[2]], this->SIZE);
                            tempQueueHalf.EnQue(tempHalfLocal);
                            tempHalfLocal = tempQueueHalf.DeQue<DTYPE_Y>();

                            AscendC::LocalTensor<float> tempFloatLocal = tempQueueFloat.AllocTensor<float>();
                            AscendC::Cast(tempFloatLocal, tempHalfLocal, AscendC::RoundMode::CAST_NONE, this->SIZE);
                            // AscendC::printf("pppp = %f %f\n", xGm(0), tempFloatLocal.GetValue(0));
                            // 2、计算最大值
                            AscendC::Compare(bits, tempFloatLocal, maxLocal, AscendC::CMPMODE::LT, this->SIZE);
                            AscendC::Select(maxLocal, bits, maxLocal, tempFloatLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->SIZE);

                            tempQueueFloat.FreeTensor(tempFloatLocal);
                            tempQueueHalf.FreeTensor(tempHalfLocal);
                        }
                        // maxQueueFloat.EnQue(maxLocal);
                        // maxLocal = maxQueueFloat.DeQue<float>();
                        AscendC::LocalTensor<float> accumulateLocal = accumulateQueueFloat.AllocTensor<float>();
                        AscendC::Duplicate<float>(accumulateLocal, static_cast<float>(0), this->SIZE);
                        for (int32_t j = 0; j < inputShape[1]; ++j){
                            // 1、拷贝数据
                            AscendC::LocalTensor<DTYPE_Y> tempHalfLocal = tempQueueHalf.AllocTensor<DTYPE_Y>();
                            AscendC::DataCopy(tempHalfLocal, xGm[i * SIZE + j * all_size + k * inputShape[1] * inputShape[2]], this->SIZE);
                            tempQueueHalf.EnQue(tempHalfLocal);
                            tempHalfLocal = tempQueueHalf.DeQue<DTYPE_Y>();

                            AscendC::LocalTensor<float> tempFloatLocal = tempQueueFloat.AllocTensor<float>();
                            AscendC::Cast(tempFloatLocal, tempHalfLocal, AscendC::RoundMode::CAST_NONE, this->SIZE);
                            // AscendC::printf("pppp = %f %f\n", xGm(0), tempFloatLocal.GetValue(0));
                            // 2、计算
                            AscendC::Sub(tempFloatLocal, tempFloatLocal, maxLocal, this->SIZE);
                            AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
                            AscendC::Add(accumulateLocal,accumulateLocal, tempFloatLocal, this->SIZE);
                            
                            tempQueueFloat.FreeTensor(tempFloatLocal);
                            tempQueueHalf.FreeTensor(tempHalfLocal);
                        }
                        // 3、取对数
                        AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
                        // 4、写回到GM中去
                        accumulateQueueFloat.EnQue(accumulateLocal);
                        accumulateLocal = accumulateQueueFloat.DeQue<float>();
                        AscendC::Add(accumulateLocal, accumulateLocal, maxLocal, this->SIZE);
                        AscendC::LocalTensor<DTYPE_Y> accumulateLocalHalf = accumulateQueueHalf.AllocTensor<DTYPE_Y>();
                        AscendC::Cast(accumulateLocalHalf, accumulateLocal, AscendC::RoundMode::CAST_NONE, this->SIZE);
                        // AscendC::printf("yyyyy = %f\n", accumulateLocalHalf.GetValue(0)); // 不知道是不是硬件问题，没有这句话结果是错的，有这句话结果是对的
                        accumulateLocalHalf.GetValue(0);
                        AscendC::DataCopy(yGm[i * this->SIZE + k * inputShape[2]], accumulateLocalHalf, this->SIZE);

                        accumulateQueueFloat.FreeTensor(accumulateLocal);
                        maxQueueFloat.FreeTensor(maxLocal);
                        accumulateQueueHalf.FreeTensor(accumulateLocalHalf);
                        // tempQueueFloat.FreeTensor(tempFloatLocal);
                        // tempQueueHalf.FreeTensor(tempHalfLocal);
                    }
                }
            }else if (this->axes == 0 || this->axes == -3){
                uint32_t all_size = inputShape[1] * inputShape[2];
                uint32_t loop = all_size / this->SIZE;
                uint32_t remain = all_size % this->SIZE;
                if (remain != 0) loop++;
                // AscendC::LocalTensor<float> maxLocal = maxQueueFloat.AllocTensor<float>();
                // AscendC::LocalTensor<DTYPE_Y> tempHalfLocal = tempQueueHalf.AllocTensor<DTYPE_Y>();
                // AscendC::LocalTensor<float> tempFloatLocal = tempQueueFloat.AllocTensor<float>();
                // AscendC::LocalTensor<float> accumulateLocal = accumulateQueueFloat.AllocTensor<float>();
                // AscendC::LocalTensor<DTYPE_Y> accumulateLocalHalf = accumulateQueueHalf.AllocTensor<DTYPE_Y>();
                for (int32_t i = 0; i < loop; ++i){
                    AscendC::LocalTensor<float> maxLocal = maxQueueFloat.AllocTensor<float>();
                    // AscendC::LocalTensor<DTYPE_Y> tempHalfLocal = tempQueueHalf.AllocTensor<DTYPE_Y>();
                    // AscendC::LocalTensor<float> tempFloatLocal = tempQueueFloat.AllocTensor<float>();
                    // AscendC::LocalTensor<float> accumulateLocal = accumulateQueueFloat.AllocTensor<float>();
                    // AscendC::LocalTensor<DTYPE_Y> accumulateLocalHalf = accumulateQueueHalf.AllocTensor<DTYPE_Y>();
                    // AscendC::LocalTensor<float> maxLocal = maxQueueFloat.AllocTensor<float>();
                    AscendC::Duplicate<float>(maxLocal, static_cast<float>(0), this->SIZE);
                    for (int32_t j = 0; j < inputShape[0]; ++j){
                        // 1、拷贝数据
                        AscendC::LocalTensor<DTYPE_Y> tempHalfLocal = tempQueueHalf.AllocTensor<DTYPE_Y>();
                        AscendC::DataCopy(tempHalfLocal, xGm[i * SIZE + j * all_size], this->SIZE);
                        tempQueueHalf.EnQue(tempHalfLocal);
                        tempHalfLocal = tempQueueHalf.DeQue<DTYPE_Y>();

                        AscendC::LocalTensor<float> tempFloatLocal = tempQueueFloat.AllocTensor<float>();
                        AscendC::Cast(tempFloatLocal, tempHalfLocal, AscendC::RoundMode::CAST_NONE, this->SIZE);
                        // AscendC::printf("pppp = %f %f\n", xGm(0), tempFloatLocal.GetValue(0));
                        // 2、计算最大值
                        AscendC::Compare(bits, tempFloatLocal, maxLocal, AscendC::CMPMODE::LT, this->SIZE);
                        AscendC::Select(maxLocal, bits, maxLocal, tempFloatLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->SIZE);
                        // AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
                        // AscendC::Add(accumulateLocal,accumulateLocal, tempFloatLocal, this->SIZE);
                        tempQueueHalf.FreeTensor(tempHalfLocal);
                        tempQueueFloat.FreeTensor(tempFloatLocal);
                    }
                    AscendC::LocalTensor<float> accumulateLocal = accumulateQueueFloat.AllocTensor<float>();
                    AscendC::Duplicate<float>(accumulateLocal, static_cast<float>(0), this->SIZE);
                    for (int32_t j = 0; j < inputShape[0]; ++j){
                        // 1、拷贝数据
                        AscendC::LocalTensor<DTYPE_Y> tempHalfLocal = tempQueueHalf.AllocTensor<DTYPE_Y>();
                        AscendC::DataCopy(tempHalfLocal, xGm[i * SIZE + j * all_size], SIZE);
                        tempQueueHalf.EnQue(tempHalfLocal);
                        tempHalfLocal = tempQueueHalf.DeQue<DTYPE_Y>();

                        AscendC::LocalTensor<float> tempFloatLocal = tempQueueFloat.AllocTensor<float>();
                        AscendC::Cast(tempFloatLocal, tempHalfLocal, AscendC::RoundMode::CAST_NONE, this->SIZE);
                        // AscendC::printf("pppp = %f %f\n", xGm(0), tempFloatLocal.GetValue(0));
                        // 2、计算
                        AscendC::Sub(tempFloatLocal, tempFloatLocal, maxLocal, this->SIZE);
                        AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
                        AscendC::Add(accumulateLocal,accumulateLocal, tempFloatLocal, this->SIZE);

                        tempQueueHalf.FreeTensor(tempHalfLocal);
                        tempQueueFloat.FreeTensor(tempFloatLocal);
                    }
                    // 3、取对数
                    AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
                    // 4、写回到GM中去
                    accumulateQueueFloat.EnQue(accumulateLocal);
                    accumulateLocal = accumulateQueueFloat.DeQue<float>();
                    AscendC::Add(accumulateLocal, accumulateLocal, maxLocal, this->SIZE);
                    AscendC::LocalTensor<DTYPE_Y> accumulateLocalHalf = accumulateQueueHalf.AllocTensor<DTYPE_Y>();
                    AscendC::Cast(accumulateLocalHalf, accumulateLocal, AscendC::RoundMode::CAST_NONE, this->SIZE);
                    // AscendC::printf("yyyyy = %f\n", accumulateLocalHalf.GetValue(0)); // 不知道是不是硬件问题，没有这句话结果是错的，有这句话结果是对的
                    accumulateLocalHalf.GetValue(0);
                    AscendC::DataCopy(yGm[i * this->SIZE], accumulateLocalHalf, this->SIZE);

                    accumulateQueueFloat.FreeTensor(accumulateLocal);
                    maxQueueFloat.FreeTensor(maxLocal);
                    accumulateQueueHalf.FreeTensor(accumulateLocalHalf);
                    // tempQueueHalf.FreeTensor(tempHalfLocal);
                    // tempQueueFloat.FreeTensor(tempFloatLocal);
                }
                // tempQueueHalf.FreeTensor(tempHalfLocal);
                // tempQueueFloat.FreeTensor(tempFloatLocal);
                // accumulateQueueFloat.FreeTensor(accumulateLocal);
                // maxQueueFloat.FreeTensor(maxLocal);
                // accumulateQueueHalf.FreeTensor(accumulateLocalHalf);
                // AscendC::printf("yyyyy = %f\n", static_cast<float>(1.0));
            }else{
                uint32_t all_size = inputShape[2];
                uint32_t loop = all_size / this->SIZE;
                uint32_t remain = all_size % this->SIZE;
                // if (remain != 0) loop++;
                AscendC::LocalTensor<float> workLocal = workQueue.AllocTensor<float>();
                for (int k = 0; k < inputShape[0] * inputShape[1]; ++k){
                    float sum = 0.0;
                    AscendC::LocalTensor<DTYPE_Y> accumulateLocalHalf = accumulateQueueHalf.AllocTensor<DTYPE_Y>();
                    // AscendC::Duplicate<DTYPE_Y>(accumulateLocalHalf, static_cast<DTYPE_Y>(0), this->SIZE);

                    AscendC::LocalTensor<DTYPE_Y> tempHalfLocal = tempQueueHalf.AllocTensor<DTYPE_Y>();
                    AscendC::LocalTensor<float> tempFloatLocal = tempQueueFloat.AllocTensor<float>();

                    AscendC::LocalTensor<float> accumulateLocal = accumulateQueueFloat.AllocTensor<float>();
                    // 寻找最大值
                    float max_num = xGm(k * all_size);
                    for (int32_t i = 0; i < loop; ++i){
                        // AscendC::LocalTensor<DTYPE_Y> accumulateLocal = accumulateQueueFloat.AllocTensor<DTYPE_Y>();
                        // AscendC::Duplicate<DTYPE_Y>(accumulateLocal, static_cast<DTYPE_Y>(0), this->SIZE);
                        // for (int32_t j = 0; j < inputShape[0]; ++j){
                        // 1、拷贝数据
                        // AscendC::LocalTensor<DTYPE_Y> tempHalfLocal = tempQueueHalf.AllocTensor<DTYPE_Y>();
                        AscendC::DataCopy(tempHalfLocal, xGm[i * SIZE + k * all_size], SIZE);
                        tempQueueHalf.EnQue(tempHalfLocal);
                        tempHalfLocal = tempQueueHalf.DeQue<DTYPE_Y>();
                        AscendC::Cast(tempFloatLocal, tempHalfLocal, AscendC::RoundMode::CAST_NONE, this->SIZE);
                        // AscendC::printf("pppp = %f %f\n", xGm(0), tempFloatLocal.GetValue(0));
                        // 2、计算
                        AscendC::ReduceMax(tempFloatLocal, tempFloatLocal, workLocal, this->SIZE);
                        float t = static_cast<float>(tempFloatLocal.GetValue(0));
                        if (max_num < t) max_num = t;
                        // AscendC::Add(accumulateLocal,accumulateLocal, tempFloatLocal, this->SIZE);
                        // tempQueueHalf.FreeTensor(tempFloatLocal);
                        // }
                        // // 3、取对数
                        // AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
                        // // 4、写回到GM中去
                        // accumulateQueueFloat.EnQue(accumulateLocal);
                        // accumulateLocal = accumulateQueueFloat.DeQue<DTYPE_Y>();
                        // AscendC::printf("yyyyy = %f\n", accumulateLocal.GetValue(0)); // 不知道是不是硬件问题，没有这句话结果是错的，有这句话结果是对的
                        // AscendC::DataCopy(yGm[i * this->SIZE], accumulateLocal, this->SIZE);

                        // accumulateQueueFloat.FreeTensor(accumulateLocal);
                    }
                    if (remain){
                        // 有剩余
                        // AscendC::LocalTensor<DTYPE_Y> tempFloatLocal = tempQueueHalf.AllocTensor<DTYPE_Y>();
                        AscendC::DataCopy(tempHalfLocal, xGm[loop * SIZE + k * all_size], SIZE);
                        tempQueueHalf.EnQue(tempHalfLocal);
                        tempHalfLocal = tempQueueHalf.DeQue<DTYPE_Y>();
                        AscendC::Cast(tempFloatLocal, tempHalfLocal, AscendC::RoundMode::CAST_NONE, this->SIZE);
                        // AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
                        for (int i = 0; i < remain; ++i){
                            float t = static_cast<float>(tempFloatLocal.GetValue(i));
                            if (max_num < t) max_num = t;
                        }
                        // tempQueueHalf.FreeTensor(tempFloatLocal);
                    }
                    float max_num_sub = 0 - max_num;
                    for (int32_t i = 0; i < loop; ++i){
                        // AscendC::LocalTensor<DTYPE_Y> accumulateLocal = accumulateQueueFloat.AllocTensor<DTYPE_Y>();
                        // AscendC::Duplicate<DTYPE_Y>(accumulateLocal, static_cast<DTYPE_Y>(0), this->SIZE);
                        // for (int32_t j = 0; j < inputShape[0]; ++j){
                        // 1、拷贝数据
                        // AscendC::LocalTensor<DTYPE_Y> tempFloatLocal = tempQueueHalf.AllocTensor<DTYPE_Y>();
                        AscendC::DataCopy(tempHalfLocal, xGm[i * SIZE + k * all_size], SIZE);
                        tempQueueHalf.EnQue(tempHalfLocal);
                        tempHalfLocal = tempQueueHalf.DeQue<DTYPE_Y>();
                        AscendC::Cast(tempFloatLocal, tempHalfLocal, AscendC::RoundMode::CAST_NONE, this->SIZE);
                        // AscendC::printf("pppp = %f %f\n", xGm(0), tempFloatLocal.GetValue(0));
                        // 2、计算
                        AscendC::Adds(tempFloatLocal, tempFloatLocal, max_num_sub, this->SIZE);
                        AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
                        AscendC::ReduceSum(tempFloatLocal, tempFloatLocal, workLocal, this->SIZE);
                        float t = static_cast<float>(tempFloatLocal.GetValue(0));
                        sum += t;
                        // AscendC::Add(accumulateLocal,accumulateLocal, tempFloatLocal, this->SIZE);
                        // tempQueueHalf.FreeTensor(tempFloatLocal);
                        // }
                        // // 3、取对数
                        // AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
                        // // 4、写回到GM中去
                        // accumulateQueueFloat.EnQue(accumulateLocal);
                        // accumulateLocal = accumulateQueueFloat.DeQue<DTYPE_Y>();
                        // AscendC::printf("yyyyy = %f\n", accumulateLocal.GetValue(0)); // 不知道是不是硬件问题，没有这句话结果是错的，有这句话结果是对的
                        // AscendC::DataCopy(yGm[i * this->SIZE], accumulateLocal, this->SIZE);

                        // accumulateQueueFloat.FreeTensor(accumulateLocal);
                    }
                    if (remain){
                        // 有剩余
                        // AscendC::LocalTensor<DTYPE_Y> tempFloatLocal = tempQueueHalf.AllocTensor<DTYPE_Y>();
                        AscendC::DataCopy(tempHalfLocal, xGm[loop * SIZE + k * all_size], SIZE);
                        tempQueueHalf.EnQue(tempHalfLocal);
                        tempHalfLocal = tempQueueHalf.DeQue<DTYPE_Y>();
                        AscendC::Cast(tempFloatLocal, tempHalfLocal, AscendC::RoundMode::CAST_NONE, this->SIZE);
                        AscendC::Adds(tempFloatLocal, tempFloatLocal, max_num_sub, this->SIZE);
                        AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
                        for (int i = 0; i < remain; ++i){
                            float t = static_cast<float>(tempFloatLocal.GetValue(i));
                            sum += t;
                        }
                        // tempQueueHalf.FreeTensor(tempFloatLocal);
                    }
                    // 3、取对数
                    AscendC::Duplicate<float>(accumulateLocal, static_cast<float>(sum), this->SIZE);
                    AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
                    AscendC::Adds(accumulateLocal, accumulateLocal, max_num, this->SIZE);
                    // 4、写回到GM中去
                    AscendC::Cast(accumulateLocalHalf, accumulateLocal, AscendC::RoundMode::CAST_NONE, this->SIZE);
                    accumulateQueueHalf.EnQue(accumulateLocalHalf);
                    accumulateLocalHalf = accumulateQueueHalf.DeQue<DTYPE_Y>();
                    // AscendC::printf("yyyyy = %f\n", accumulateLocalHalf.GetValue(0)); // 不知道是不是硬件问题，没有这句话结果是错的，有这句话结果是对的
                    accumulateLocalHalf.GetValue(0);
                    AscendC::DataCopy(yGm[k], accumulateLocalHalf, this->SIZE); // 可以修改为只写入32B
                    
                    accumulateQueueFloat.FreeTensor(accumulateLocal);
                    accumulateQueueHalf.FreeTensor(accumulateLocalHalf);
                    tempQueueFloat.FreeTensor(tempFloatLocal);
                    tempQueueHalf.FreeTensor(tempHalfLocal);
                }
                workQueue.FreeTensor(workLocal);
            }
        }
        else if (this->dataType == 0 && this->dimension == 3){
            if (this->axes == 1 || this->axes == -2){
                uint32_t all_size = inputShape[2];
                uint32_t loop = all_size / this->SIZE;
                uint32_t remain = all_size % this->SIZE;
                if (remain != 0) loop++;
                for (int k = 0; k < inputShape[0]; ++k){
                    for (int32_t i = 0; i < loop; ++i){
                        AscendC::LocalTensor<DTYPE_Y> accumulateLocal = accumulateQueueFloat.AllocTensor<DTYPE_Y>();
                        AscendC::Duplicate<DTYPE_Y>(accumulateLocal, static_cast<DTYPE_Y>(0), this->SIZE);
                        for (int32_t j = 0; j < inputShape[1]; ++j){
                            // 1、拷贝数据
                            AscendC::LocalTensor<DTYPE_Y> tempFloatLocal = tempQueueFloat.AllocTensor<DTYPE_Y>();
                            AscendC::DataCopy(tempFloatLocal, xGm[i * SIZE + j * all_size + k * inputShape[1] * inputShape[2]], SIZE);
                            tempQueueFloat.EnQue(tempFloatLocal);
                            tempFloatLocal = tempQueueFloat.DeQue<DTYPE_Y>();
                            // AscendC::printf("pppp = %f %f\n", xGm(0), tempFloatLocal.GetValue(0));
                            // 2、计算
                            AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
                            AscendC::Add(accumulateLocal,accumulateLocal, tempFloatLocal, this->SIZE);
                            tempQueueFloat.FreeTensor(tempFloatLocal);
                        }
                        // 3、取对数
                        AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
                        // 4、写回到GM中去
                        accumulateQueueFloat.EnQue(accumulateLocal);
                        accumulateLocal = accumulateQueueFloat.DeQue<DTYPE_Y>();
                        // AscendC::printf("yyyyy = %f\n", accumulateLocal.GetValue(0)); // 不知道是不是硬件问题，没有这句话结果是错的，有这句话结果是对的
                        accumulateLocal.GetValue(0);
                        AscendC::DataCopy(yGm[i * this->SIZE + k * inputShape[2]], accumulateLocal, this->SIZE);

                        accumulateQueueFloat.FreeTensor(accumulateLocal);
                    }
                }

                // for (int i = 0; i < inputShape[0]; ++i){
                //     for (int j = 0; j < inputShape[2]; ++j){
                //         AscendC::Duplicate<float>(temp2, static_cast<float>(0), 1);
                //         uint32_t start = i * inputShape[1] * inputShape[2] + j;
                //         // float max_value = xGm(start);
                //         // for (int k = 0; k < inputShape[1]; ++k){
                //         //     uint32_t index = start + k * inputShape[2];
                //         //     float value = xGm(index);
                //         //     if (value > max_value) max_value = value;
                //         // }   
                //         for (int k = 0; k < inputShape[1]; ++k){
                //             uint32_t index = start + k * inputShape[2];
                //             float value = xGm(index);
                //             // value -= max_value;
                //             AscendC::Duplicate<float>(temp1, value, 1);
                //             AscendC::Exp(temp1, temp1, 1);
                //             AscendC::Adds(temp2,temp2, static_cast<float>(temp1(0)), 1);
                //         }
                //         AscendC::Ln(temp1, temp2, 1);
                //         // AscendC::Adds(temp1,temp1, static_cast<float>(max_value), 1);
                //         uint32_t end = i * inputShape[2] + j;
                //         yGm(end) = temp1(0);
                //     }
                // }
            }else if (this->axes == 0 || this->axes == -3){

                uint32_t all_size = inputShape[1] * inputShape[2];
                uint32_t loop = all_size / this->SIZE;
                uint32_t remain = all_size % this->SIZE;
                if (remain != 0) loop++;
                for (int32_t i = 0; i < loop; ++i){
                    AscendC::LocalTensor<DTYPE_Y> accumulateLocal = accumulateQueueFloat.AllocTensor<DTYPE_Y>();
                    AscendC::Duplicate<DTYPE_Y>(accumulateLocal, static_cast<DTYPE_Y>(0), this->SIZE);
                    for (int32_t j = 0; j < inputShape[0]; ++j){
                        // 1、拷贝数据
                        AscendC::LocalTensor<DTYPE_Y> tempFloatLocal = tempQueueFloat.AllocTensor<DTYPE_Y>();
                        AscendC::DataCopy(tempFloatLocal, xGm[i * SIZE + j * all_size], SIZE);
                        tempQueueFloat.EnQue(tempFloatLocal);
                        tempFloatLocal = tempQueueFloat.DeQue<DTYPE_Y>();
                        // AscendC::printf("pppp = %f %f\n", xGm(0), tempFloatLocal.GetValue(0));
                        // 2、计算
                        AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
                        AscendC::Add(accumulateLocal,accumulateLocal, tempFloatLocal, this->SIZE);
                        tempQueueFloat.FreeTensor(tempFloatLocal);
                    }
                    // 3、取对数
                    AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
                    // 4、写回到GM中去
                    accumulateQueueFloat.EnQue(accumulateLocal);
                    accumulateLocal = accumulateQueueFloat.DeQue<DTYPE_Y>();
                    // AscendC::printf("yyyyy = %f\n", accumulateLocal.GetValue(0)); // 不知道是不是硬件问题，没有这句话结果是错的，有这句话结果是对的
                    accumulateLocal.GetValue(0);
                    AscendC::DataCopy(yGm[i * this->SIZE], accumulateLocal, this->SIZE);

                    accumulateQueueFloat.FreeTensor(accumulateLocal);
                }

                // for (int i = 0; i < inputShape[1] * inputShape[2]; ++i){
                //     AscendC::Duplicate<float>(temp2, static_cast<float>(0), 1);
                //     uint32_t start = i;
                //     // float max_value = xGm.GetValue(start);
                //     // for (int j = 0; j < inputShape[0]; ++j){
                //     //     uint32_t index = start + j * inputShape[1] * inputShape[2];
                //     //     float value = xGm(index);
                //     //     if (value > max_value) max_value = value;
                //     // }
                //     for (int j = 0; j < inputShape[0]; ++j){
                //         uint32_t index = start + j * inputShape[1] * inputShape[2];
                //         float value = xGm(index);
                //         // value -= max_value;
                //         AscendC::Duplicate<float>(temp1, value, 1);
                //         AscendC::Exp(temp1, temp1, 1);
                //         AscendC::Adds(temp2,temp2, static_cast<float>(temp1(0)), 1);
                //     }
                //     AscendC::Ln(temp1, temp2, 1);
                //     // AscendC::Adds(temp1,temp1, static_cast<float>(max_value), 1);
                //     uint32_t end = i;
                //     yGm(end) = temp1(0);
                // }
            }else{

                uint32_t all_size = inputShape[2];
                uint32_t loop = all_size / this->SIZE;
                uint32_t remain = all_size % this->SIZE;
                // if (remain != 0) loop++;
                AscendC::LocalTensor<DTYPE_Y> workLocal = workQueue.AllocTensor<DTYPE_Y>();
                for (int k = 0; k < inputShape[0] * inputShape[1]; ++k){
                    float sum = 0.0;
                    AscendC::LocalTensor<DTYPE_Y> accumulateLocal = accumulateQueueFloat.AllocTensor<DTYPE_Y>();
                    AscendC::Duplicate<DTYPE_Y>(accumulateLocal, static_cast<DTYPE_Y>(0), this->SIZE);
                    for (int32_t i = 0; i < loop; ++i){
                        // AscendC::LocalTensor<DTYPE_Y> accumulateLocal = accumulateQueueFloat.AllocTensor<DTYPE_Y>();
                        // AscendC::Duplicate<DTYPE_Y>(accumulateLocal, static_cast<DTYPE_Y>(0), this->SIZE);
                        // for (int32_t j = 0; j < inputShape[0]; ++j){
                        // 1、拷贝数据
                        AscendC::LocalTensor<DTYPE_Y> tempFloatLocal = tempQueueFloat.AllocTensor<DTYPE_Y>();
                        AscendC::DataCopy(tempFloatLocal, xGm[i * SIZE + k * all_size], SIZE);
                        tempQueueFloat.EnQue(tempFloatLocal);
                        tempFloatLocal = tempQueueFloat.DeQue<DTYPE_Y>();
                        // AscendC::printf("pppp = %f %f\n", xGm(0), tempFloatLocal.GetValue(0));
                        // 2、计算
                        AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
                        AscendC::ReduceSum(tempFloatLocal, tempFloatLocal, workLocal, this->SIZE);
                        float t = tempFloatLocal.GetValue(0);
                        sum += t;
                        // AscendC::Add(accumulateLocal,accumulateLocal, tempFloatLocal, this->SIZE);
                        tempQueueFloat.FreeTensor(tempFloatLocal);
                        // }
                        // // 3、取对数
                        // AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
                        // // 4、写回到GM中去
                        // accumulateQueueFloat.EnQue(accumulateLocal);
                        // accumulateLocal = accumulateQueueFloat.DeQue<DTYPE_Y>();
                        // AscendC::printf("yyyyy = %f\n", accumulateLocal.GetValue(0)); // 不知道是不是硬件问题，没有这句话结果是错的，有这句话结果是对的
                        // AscendC::DataCopy(yGm[i * this->SIZE], accumulateLocal, this->SIZE);

                        // accumulateQueueFloat.FreeTensor(accumulateLocal);
                    }
                    if (remain){
                        // 有剩余
                        AscendC::LocalTensor<DTYPE_Y> tempFloatLocal = tempQueueFloat.AllocTensor<DTYPE_Y>();
                        AscendC::DataCopy(tempFloatLocal, xGm[loop * SIZE + k * all_size], SIZE);
                        tempQueueFloat.EnQue(tempFloatLocal);
                        tempFloatLocal = tempQueueFloat.DeQue<DTYPE_Y>();
                        AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
                        for (int i = 0; i < remain; ++i){
                            float t = tempFloatLocal.GetValue(i);
                            sum += t;
                        }
                        tempQueueFloat.FreeTensor(tempFloatLocal);
                    }
                    // 3、取对数
                    AscendC::Duplicate<DTYPE_Y>(accumulateLocal, static_cast<DTYPE_Y>(sum), this->SIZE);
                    AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
                    // 4、写回到GM中去
                    accumulateQueueFloat.EnQue(accumulateLocal);
                    accumulateLocal = accumulateQueueFloat.DeQue<DTYPE_Y>();
                    // AscendC::printf("yyyyy = %f\n", accumulateLocal.GetValue(0)); // 不知道是不是硬件问题，没有这句话结果是错的，有这句话结果是对的
                    accumulateLocal.GetValue(0);
                    AscendC::DataCopy(yGm[k], accumulateLocal, this->SIZE); // 可以修改为只写入32B
                    accumulateQueueFloat.FreeTensor(accumulateLocal);
                }
                workQueue.FreeTensor(workLocal);
                // for (int i = 0; i < inputShape[0] * inputShape[1]; ++i){
                //     AscendC::Duplicate<float>(temp2, static_cast<float>(0), 1);
                //     uint32_t start = i * inputShape[2];
                //     // float max_value = xGm.GetValue(start);
                //     // for (int j = 0; j < inputShape[2]; ++j){
                //     //     uint32_t index = start + j;
                //     //     float value = xGm(index);
                //     //     if (value > max_value) max_value = value;
                //     // }
                //     for (int j = 0; j < inputShape[2]; ++j){
                //         uint32_t index = start + j;
                //         float value = xGm(index);
                //         // value -= max_value;
                //         AscendC::Duplicate<float>(temp1, value, 1);
                //         AscendC::Exp(temp1, temp1, 1);
                //         AscendC::Adds(temp2,temp2, static_cast<float>(temp1(0)), 1);
                //     }
                //     AscendC::Ln(temp1, temp2, 1);
                //     // AscendC::Adds(temp1,temp1, static_cast<float>(max_value), 1);
                //     uint32_t end = i;
                //     yGm(end) = temp1(0);
                // }
            }
        }
        else if (this->dataType == 0 && this->dimension == 4){
            if (this->axes == 1 || this->axes == -3){
                uint32_t all_size = inputShape[2] * inputShape[3];
                uint32_t loop = all_size / this->SIZE;
                uint32_t remain = all_size % this->SIZE;
                if (remain != 0) loop++;
                for (int k = 0; k < inputShape[0]; ++k){
                    for (int32_t i = 0; i < loop; ++i){
                        AscendC::LocalTensor<DTYPE_Y> accumulateLocal = accumulateQueueFloat.AllocTensor<DTYPE_Y>();
                        AscendC::Duplicate<DTYPE_Y>(accumulateLocal, static_cast<DTYPE_Y>(0), this->SIZE);
                        for (int32_t j = 0; j < inputShape[1]; ++j){
                            // 1、拷贝数据
                            AscendC::LocalTensor<DTYPE_Y> tempFloatLocal = tempQueueFloat.AllocTensor<DTYPE_Y>();
                            AscendC::DataCopy(tempFloatLocal, xGm[i * SIZE + j * all_size + k * inputShape[1] * inputShape[2] * inputShape[3]], SIZE);
                            tempQueueFloat.EnQue(tempFloatLocal);
                            tempFloatLocal = tempQueueFloat.DeQue<DTYPE_Y>();
                            // AscendC::printf("pppp = %f %f\n", xGm(0), tempFloatLocal.GetValue(0));
                            // 2、计算
                            AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
                            AscendC::Add(accumulateLocal,accumulateLocal, tempFloatLocal, this->SIZE);
                            tempQueueFloat.FreeTensor(tempFloatLocal);
                        }
                        // 3、取对数
                        AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
                        // 4、写回到GM中去
                        accumulateQueueFloat.EnQue(accumulateLocal);
                        accumulateLocal = accumulateQueueFloat.DeQue<DTYPE_Y>();
                        // AscendC::printf("yyyyy = %f\n", accumulateLocal.GetValue(0)); // 不知道是不是硬件问题，没有这句话结果是错的，有这句话结果是对的
                        accumulateLocal.GetValue(0);
                        AscendC::DataCopy(yGm[i * this->SIZE + k * inputShape[2] * inputShape[3]], accumulateLocal, this->SIZE);

                        accumulateQueueFloat.FreeTensor(accumulateLocal);
                    }
                }

                // for (int i = 0; i < inputShape[0]; ++i){
                //     for (int j = 0; j < inputShape[2] * inputShape[3]; ++j){
                //         AscendC::Duplicate<float>(temp2, static_cast<float>(0), 1);
                //         uint32_t start = i * inputShape[1] * inputShape[2] * inputShape[3] + j;
                //         // float max_value = xGm(start);
                //         // for (int k = 0; k < inputShape[1]; ++k){
                //         //     uint32_t index = start + k * inputShape[2];
                //         //     float value = xGm(index);
                //         //     if (value > max_value) max_value = value;
                //         // }   
                //         for (int k = 0; k < inputShape[1]; ++k){
                //             uint32_t index = start + k * inputShape[2] * inputShape[3];
                //             float value = xGm(index);
                //             // value -= max_value;
                //             AscendC::Duplicate<float>(temp1, value, 1);
                //             AscendC::Exp(temp1, temp1, 1);
                //             AscendC::Adds(temp2,temp2, static_cast<float>(temp1(0)), 1);
                //         }
                //         AscendC::Ln(temp1, temp2, 1);
                //         // AscendC::Adds(temp1,temp1, static_cast<float>(max_value), 1);
                //         uint32_t end = i * inputShape[2] * inputShape[3] + j;
                //         yGm(end) = temp1(0);
                //     }
                // }
            }
            else if (this->axes == 2 || this->axes == -2){
                uint32_t all_size = inputShape[3];
                uint32_t loop = all_size / this->SIZE;
                uint32_t remain = all_size % this->SIZE;
                if (remain != 0) loop++;
                for (int32_t i = 0; i < loop; ++i){
                    for (int k = 0; k < inputShape[0] * inputShape[1]; ++k){
                        AscendC::LocalTensor<DTYPE_Y> accumulateLocal = accumulateQueueFloat.AllocTensor<DTYPE_Y>();
                        AscendC::Duplicate<DTYPE_Y>(accumulateLocal, static_cast<DTYPE_Y>(0), this->SIZE);
                        for (int32_t j = 0; j < inputShape[2]; ++j){
                            // 1、拷贝数据
                            AscendC::LocalTensor<DTYPE_Y> tempFloatLocal = tempQueueFloat.AllocTensor<DTYPE_Y>();
                            AscendC::DataCopy(tempFloatLocal, xGm[i * SIZE + j * all_size + k * inputShape[2] * inputShape[3]], SIZE);
                            tempQueueFloat.EnQue(tempFloatLocal);
                            tempFloatLocal = tempQueueFloat.DeQue<DTYPE_Y>();
                            // AscendC::printf("pppp = %f %f\n", xGm(0), tempFloatLocal.GetValue(0));
                            // 2、计算
                            AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
                            AscendC::Add(accumulateLocal,accumulateLocal, tempFloatLocal, this->SIZE);
                            tempQueueFloat.FreeTensor(tempFloatLocal);
                        }
                        // 3、取对数
                        AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
                        // 4、写回到GM中去
                        accumulateQueueFloat.EnQue(accumulateLocal);
                        accumulateLocal = accumulateQueueFloat.DeQue<DTYPE_Y>();
                        // AscendC::printf("yyyyy = %f\n", accumulateLocal.GetValue(0)); // 不知道是不是硬件问题，没有这句话结果是错的，有这句话结果是对的
                        accumulateLocal.GetValue(0);
                        AscendC::DataCopy(yGm[i * this->SIZE + k * inputShape[3]], accumulateLocal, this->SIZE);

                        accumulateQueueFloat.FreeTensor(accumulateLocal);
                    }
                }

                // for (int i = 0; i < inputShape[0] * inputShape[1]; ++i){
                //     for (int j = 0; j < inputShape[3]; ++j){
                //         AscendC::Duplicate<float>(temp2, static_cast<float>(0), 1);
                //         uint32_t start = i * inputShape[2] * inputShape[3] + j;
                //         // float max_value = xGm(start);
                //         // for (int k = 0; k < inputShape[1]; ++k){
                //         //     uint32_t index = start + k * inputShape[2];
                //         //     float value = xGm(index);
                //         //     if (value > max_value) max_value = value;
                //         // }   
                //         for (int k = 0; k < inputShape[2]; ++k){
                //             uint32_t index = start + k * inputShape[3];
                //             float value = xGm(index);
                //             // value -= max_value;
                //             AscendC::Duplicate<float>(temp1, value, 1);
                //             AscendC::Exp(temp1, temp1, 1);
                //             AscendC::Adds(temp2,temp2, static_cast<float>(temp1(0)), 1);
                //         }
                //         AscendC::Ln(temp1, temp2, 1);
                //         // AscendC::Adds(temp1,temp1, static_cast<float>(max_value), 1);
                //         uint32_t end = i * inputShape[3] + j;
                //         yGm(end) = temp1(0);
                //     }
                // }
            }else if (this->axes == 0 || this->axes == -4){
                // 每次处理SIZE大小
                // AscendC::printf("test\n");
                uint32_t all_size = inputShape[1] * inputShape[2] * inputShape[3];
                uint32_t loop = all_size / this->SIZE;
                uint32_t remain = all_size % this->SIZE;
                if (remain != 0) loop++;
                for (int32_t i = 0; i < loop; ++i){
                    AscendC::LocalTensor<DTYPE_Y> accumulateLocal = accumulateQueueFloat.AllocTensor<DTYPE_Y>();
                    AscendC::Duplicate<DTYPE_Y>(accumulateLocal, static_cast<DTYPE_Y>(0), this->SIZE);
                    for (int32_t j = 0; j < inputShape[0]; ++j){
                        // 1、拷贝数据
                        AscendC::LocalTensor<DTYPE_Y> tempFloatLocal = tempQueueFloat.AllocTensor<DTYPE_Y>();
                        AscendC::DataCopy(tempFloatLocal, xGm[i * SIZE + j * all_size], SIZE);
                        tempQueueFloat.EnQue(tempFloatLocal);
                        tempFloatLocal = tempQueueFloat.DeQue<DTYPE_Y>();
                        // AscendC::printf("pppp = %f %f\n", xGm(0), tempFloatLocal.GetValue(0));
                        // 2、计算
                        AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
                        AscendC::Add(accumulateLocal,accumulateLocal, tempFloatLocal, this->SIZE);
                        tempQueueFloat.FreeTensor(tempFloatLocal);
                    }
                    // 3、取对数
                    AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
                    // 4、写回到GM中去
                    accumulateQueueFloat.EnQue(accumulateLocal);
                    accumulateLocal = accumulateQueueFloat.DeQue<DTYPE_Y>();
                    // AscendC::printf("yyyyy = %f\n", accumulateLocal.GetValue(0));
                    accumulateLocal.GetValue(0);
                    AscendC::DataCopy(yGm[i * this->SIZE], accumulateLocal, this->SIZE);

                    accumulateQueueFloat.FreeTensor(accumulateLocal);
                }
                
                
                // for (int i = 0; i < inputShape[1] * inputShape[2] * inputShape[3]; ++i){
                //     AscendC::Duplicate<float>(temp2, static_cast<float>(0), 1);
                //     uint32_t start = i;
                //     // float max_value = xGm.GetValue(start);
                //     // for (int j = 0; j < inputShape[0]; ++j){
                //     //     uint32_t index = start + j * inputShape[1] * inputShape[2];
                //     //     float value = xGm(index);
                //     //     if (value > max_value) max_value = value;
                //     // }
                //     for (int j = 0; j < inputShape[0]; ++j){
                //         uint32_t index = start + j * inputShape[1] * inputShape[2] * inputShape[3];
                //         float value = xGm(index);
                //         // value -= max_value;
                //         AscendC::Duplicate<float>(temp1, value, 1);
                //         AscendC::Exp(temp1, temp1, 1);
                //         AscendC::Adds(temp2,temp2, static_cast<float>(temp1(0)), 1);
                //     }
                //     AscendC::Ln(temp1, temp2, 1);
                //     // AscendC::Adds(temp1,temp1, static_cast<float>(max_value), 1);
                //     uint32_t end = i;
                //     yGm(end) = temp1(0);
                // }
            }else{
                uint32_t all_size = inputShape[3];
                uint32_t loop = all_size / this->SIZE;
                uint32_t remain = all_size % this->SIZE;
                // if (remain != 0) loop++;
                AscendC::LocalTensor<DTYPE_Y> workLocal = workQueue.AllocTensor<DTYPE_Y>();
                for (int k = 0; k < inputShape[0] * inputShape[1] * inputShape[3]; ++k){
                    float sum = 0.0;
                    AscendC::LocalTensor<DTYPE_Y> accumulateLocal = accumulateQueueFloat.AllocTensor<DTYPE_Y>();
                    AscendC::Duplicate<DTYPE_Y>(accumulateLocal, static_cast<DTYPE_Y>(0), this->SIZE);
                    for (int32_t i = 0; i < loop; ++i){
                        // AscendC::LocalTensor<DTYPE_Y> accumulateLocal = accumulateQueueFloat.AllocTensor<DTYPE_Y>();
                        // AscendC::Duplicate<DTYPE_Y>(accumulateLocal, static_cast<DTYPE_Y>(0), this->SIZE);
                        // for (int32_t j = 0; j < inputShape[0]; ++j){
                        // 1、拷贝数据
                        AscendC::LocalTensor<DTYPE_Y> tempFloatLocal = tempQueueFloat.AllocTensor<DTYPE_Y>();
                        AscendC::DataCopy(tempFloatLocal, xGm[i * SIZE + k * all_size], SIZE);
                        tempQueueFloat.EnQue(tempFloatLocal);
                        tempFloatLocal = tempQueueFloat.DeQue<DTYPE_Y>();
                        // AscendC::printf("pppp = %f %f\n", xGm(0), tempFloatLocal.GetValue(0));
                        // 2、计算
                        AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
                        AscendC::ReduceSum(tempFloatLocal, tempFloatLocal, workLocal, this->SIZE);
                        float t = tempFloatLocal.GetValue(0);
                        sum += t;
                        // AscendC::Add(accumulateLocal,accumulateLocal, tempFloatLocal, this->SIZE);
                        tempQueueFloat.FreeTensor(tempFloatLocal);
                        // }
                        // // 3、取对数
                        // AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
                        // // 4、写回到GM中去
                        // accumulateQueueFloat.EnQue(accumulateLocal);
                        // accumulateLocal = accumulateQueueFloat.DeQue<DTYPE_Y>();
                        // AscendC::printf("yyyyy = %f\n", accumulateLocal.GetValue(0)); // 不知道是不是硬件问题，没有这句话结果是错的，有这句话结果是对的
                        // AscendC::DataCopy(yGm[i * this->SIZE], accumulateLocal, this->SIZE);

                        // accumulateQueueFloat.FreeTensor(accumulateLocal);
                    }
                    if (remain){
                        // 有剩余
                        AscendC::LocalTensor<DTYPE_Y> tempFloatLocal = tempQueueFloat.AllocTensor<DTYPE_Y>();
                        AscendC::DataCopy(tempFloatLocal, xGm[loop * SIZE + k * all_size], SIZE);
                        tempQueueFloat.EnQue(tempFloatLocal);
                        tempFloatLocal = tempQueueFloat.DeQue<DTYPE_Y>();
                        AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
                        for (int i = 0; i < remain; ++i){
                            float t = tempFloatLocal.GetValue(i);
                            sum += t;
                        }
                        tempQueueFloat.FreeTensor(tempFloatLocal);
                    }
                    // 3、取对数
                    AscendC::Duplicate<DTYPE_Y>(accumulateLocal, static_cast<DTYPE_Y>(sum), this->SIZE);
                    AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
                    // 4、写回到GM中去
                    accumulateQueueFloat.EnQue(accumulateLocal);
                    accumulateLocal = accumulateQueueFloat.DeQue<DTYPE_Y>();
                    // AscendC::printf("yyyyy = %f\n", accumulateLocal.GetValue(0)); // 不知道是不是硬件问题，没有这句话结果是错的，有这句话结果是对的
                    accumulateLocal.GetValue(0);
                    AscendC::DataCopy(yGm[k], accumulateLocal, this->SIZE); // 可以修改为只写入32B
                    accumulateQueueFloat.FreeTensor(accumulateLocal);
                }
                workQueue.FreeTensor(workLocal);

                // for (int i = 0; i < inputShape[0] * inputShape[1] * inputShape[2]; ++i){
                //     AscendC::Duplicate<float>(temp2, static_cast<float>(0), 1);
                //     uint32_t start = i * inputShape[3];
                //     // float max_value = xGm.GetValue(start);
                //     // for (int j = 0; j < inputShape[2]; ++j){
                //     //     uint32_t index = start + j;
                //     //     float value = xGm(index);
                //     //     if (value > max_value) max_value = value;
                //     // }
                //     for (int j = 0; j < inputShape[3]; ++j){
                //         uint32_t index = start + j;
                //         float value = xGm(index);
                //         // value -= max_value;
                //         AscendC::Duplicate<float>(temp1, value, 1);
                //         AscendC::Exp(temp1, temp1, 1);
                //         AscendC::Adds(temp2,temp2, static_cast<float>(temp1(0)), 1);
                //     }
                //     AscendC::Ln(temp1, temp2, 1);
                //     // AscendC::Adds(temp1,temp1, static_cast<float>(max_value), 1);
                //     uint32_t end = i;
                //     yGm(end) = temp1(0);
                // }
            }
        }
    }



private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> accumulateQueueFloat, tempQueueFloat, accumulateQueueHalf, tempQueueHalf;
    TQue<QuePosition::VECCALC, BUFFER_NUM> workQueue, maxQueueFloat;
    TBuf<QuePosition::VECCALC> temp1Float, temp2Float, tempBits; 
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
    op.Process(tiling_data.inputShape);
    // op.Process();
}