#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelLogSumExpReduce {
 public:
  __aicore__ inline KernelLogSumExpReduce() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR axes, GM_ADDR y,
                              uint32_t dimension, uint32_t dataSize,
                              uint32_t dataType) {
    this->dimension = dimension;
    this->dataType = dataType;
    this->SIZE = 3968;
    axesGm.SetGlobalBuffer((__gm__ DTYPE_AXES *)axes, 1);
    this->axes = axesGm(0);
    AscendC::printf("dimension: %d, axes: %d", this->dimension, this->axes);
    xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, dataSize);
    yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, dataSize);

    pipe.InitBuffer(temp1Float, 8 * sizeof(float));
    pipe.InitBuffer(temp2Float, 8 * sizeof(float));

    pipe.InitBuffer(tempBits, this->SIZE * sizeof(uint8_t));

    pipe.InitBuffer(accumulateQueueFloat, BUFFER_NUM,
                    this->SIZE * sizeof(float));
    pipe.InitBuffer(tempQueueFloat, BUFFER_NUM, this->SIZE * sizeof(float));
    pipe.InitBuffer(maxQueueFloat, BUFFER_NUM, this->SIZE * sizeof(float));

    pipe.InitBuffer(accumulateQueueHalf, BUFFER_NUM, this->SIZE * sizeof(half));
    pipe.InitBuffer(tempQueueHalf, BUFFER_NUM, this->SIZE * sizeof(half));

    int32_t typeSize = 2;
    if (dataType == 0) {
      typeSize = 4;
    }
    int32_t elementsPerBlock = 32 / typeSize;
    int32_t elementsPerRepeat = 256 / typeSize;
    int32_t firstMaxRepeat = this->SIZE / elementsPerRepeat;
    int32_t finalWorkLocalNeedSize = (firstMaxRepeat + elementsPerBlock - 1) /
                                     elementsPerBlock * elementsPerBlock;
    pipe.InitBuffer(workQueue, BUFFER_NUM,
                    finalWorkLocalNeedSize * sizeof(float));
  }

  __aicore__ inline void Process(uint32_t inputShape[4]) {
    LocalTensor<float> temp1 = temp1Float.Get<float>();
    LocalTensor<float> temp2 = temp2Float.Get<float>();
    LocalTensor<uint8_t> bits = tempBits.Get<uint8_t>();
    if (this->dataType == 1 && this->dimension == 1) {
      // uint32_t all_size = 1;
      // uint32_t loop = all_size / this->SIZE;
      // uint32_t remain = all_size % this->SIZE;
      // if (remain != 0) loop++;
      AscendC::LocalTensor<float> accumulateLocal =
          accumulateQueueFloat.AllocTensor<float>();
      AscendC::LocalTensor<DTYPE_Y> accumulateHalfLocal =
          accumulateQueueHalf.AllocTensor<DTYPE_Y>();
      AscendC::Duplicate<float>(accumulateLocal, static_cast<float>(0),
                                this->SIZE);
      for (int32_t j = 0; j < inputShape[0]; ++j) {
        AscendC::LocalTensor<DTYPE_X> tempHalfLocal =
            tempQueueHalf.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<float> tempFloatLocal =
            tempQueueFloat.AllocTensor<float>();
        // 1、拷贝数据
        AscendC::DataCopy(tempHalfLocal, xGm[j], SIZE);
        tempQueueHalf.EnQue(tempHalfLocal);
        tempHalfLocal = tempQueueHalf.DeQue<DTYPE_X>();
        AscendC::Cast(tempFloatLocal, tempHalfLocal,
                      AscendC::RoundMode::CAST_NONE, SIZE);
        // AscendC::printf("pppp = %f %fIn",xGm(e), tempFloatLocal.GetValue(e));
        // 2、计算
        AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
        AscendC::Add(accumulateLocal, accumulateLocal, tempFloatLocal,
                     this->SIZE);
        tempQueueHalf.FreeTensor(tempHalfLocal);
        tempQueueFloat.FreeTensor(tempFloatLocal);
      }
      // 3、取对数
      AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
      AscendC::Cast(accumulateHalfLocal, accumulateLocal,
                    AscendC::RoundMode::CAST_NONE, SIZE);
      // 4. 写回
      accumulateQueueHalf.EnQue(accumulateHalfLocal);
      accumulateHalfLocal = accumulateQueueHalf.DeQue<DTYPE_Y>();
      accumulateHalfLocal.GetValue(0);
      AscendC::DataCopy(yGm[0], accumulateHalfLocal, this->SIZE);
      // AscendC::DataCopy(yGm[k2 * inputShape[1]], accumulateHalfLocal,
      // this->SIZE);
      accumulateQueueFloat.FreeTensor(accumulateLocal);
      accumulateQueueHalf.FreeTensor(accumulateHalfLocal);
    } else if (this->dataType == 1 && this->dimension == 3) {
      if (this->axes == 1 || this->axes == -2) {
        uint32_t all_size = inputShape[2];
        uint32_t loop = all_size / this->SIZE;
        uint32_t remain = all_size % this->SIZE;
        if (remain != 0) loop++;
        for (int k = 0; k < inputShape[0]; ++k) {
          for (int32_t i = 0; i < loop; ++i) {
            AscendC::LocalTensor<float> accumulateLocal =
                accumulateQueueFloat.AllocTensor<float>();
            AscendC::LocalTensor<DTYPE_Y> accumulateHalfLocal =
                accumulateQueueHalf.AllocTensor<DTYPE_Y>();
            AscendC::Duplicate<float>(accumulateLocal, static_cast<float>(0),
                                      this->SIZE);
            for (int32_t j = 0; j < inputShape[1]; ++j) {
              AscendC::LocalTensor<DTYPE_X> tempHalfLocal =
                  tempQueueHalf.AllocTensor<DTYPE_X>();
              AscendC::LocalTensor<float> tempFloatLocal =
                  tempQueueFloat.AllocTensor<float>();
              // 1、拷贝数据
              //  AscendC::LocalTensor<DTYPE_Y> tempHalfLocal =
              //  tempQueueHalf.AllocTensor<DTYPE_Y>();
              AscendC::DataCopy(tempHalfLocal,
                                xGm[i * SIZE + j * all_size +
                                    k * inputShape[1] * inputShape[2]],
                                SIZE);
              tempQueueHalf.EnQue(tempHalfLocal);
              tempHalfLocal = tempQueueHalf.DeQue<DTYPE_X>();
              AscendC::Cast(tempFloatLocal, tempHalfLocal,
                            AscendC::RoundMode::CAST_NONE, SIZE);
              // 2、计算
              AscendC::Exp(tempFloatLocal, tempFloatLocal, SIZE);
              // AscendC::printf("pppp = %f %fIn",xGm(e),
              // tempHalfLocal.GetValue(e));
              AscendC::Add(accumulateLocal, accumulateLocal, tempFloatLocal,
                           this->SIZE);
              tempQueueHalf.FreeTensor(tempHalfLocal);
              tempQueueFloat.FreeTensor(tempFloatLocal);
            }
            // 3、取对数
            AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
            AscendC::Cast(accumulateHalfLocal, accumulateLocal,
                          AscendC::RoundMode::CAST_NONE, SIZE);
            // 4. 写回
            accumulateQueueHalf.EnQue(accumulateHalfLocal);
            accumulateHalfLocal = accumulateQueueHalf.DeQue<DTYPE_Y>();
            accumulateHalfLocal.GetValue(0);
            AscendC::DataCopy(yGm[i * SIZE + k * inputShape[2]],
                              accumulateHalfLocal, this->SIZE);
            accumulateQueueFloat.FreeTensor(accumulateLocal);
            accumulateQueueHalf.FreeTensor(accumulateHalfLocal);
          }
        }
      } else if (this->axes == 0 || this->axes == -3) {
        uint32_t all_size = inputShape[1] * inputShape[2];
        uint32_t loop = all_size / this->SIZE;
        uint32_t remain = all_size % this->SIZE;
        if (remain != 0) loop++;
        for (int32_t i = 0; i < loop; ++i) {
          AscendC::LocalTensor<float> accumulateLocal =
              accumulateQueueFloat.AllocTensor<float>();
          AscendC::LocalTensor<DTYPE_Y> accumulateHalfLocal =
              accumulateQueueHalf.AllocTensor<DTYPE_Y>();
          AscendC::Duplicate<float>(accumulateLocal, static_cast<float>(0),
                                    this->SIZE);
          for (int32_t j = 0; j < inputShape[0]; ++j) {
            AscendC::LocalTensor<DTYPE_X> tempHalfLocal =
                tempQueueHalf.AllocTensor<DTYPE_X>();
            AscendC::LocalTensor<float> tempFloatLocal =
                tempQueueFloat.AllocTensor<float>();
            // 1、拷贝数据
            AscendC::DataCopy(tempHalfLocal, xGm[i * SIZE + j * all_size],
                              SIZE);
            tempQueueHalf.EnQue(tempHalfLocal);
            tempHalfLocal = tempQueueHalf.DeQue<DTYPE_X>();
            AscendC::Cast(tempFloatLocal, tempHalfLocal,
                          AscendC::RoundMode::CAST_NONE, SIZE);
            // AscendC::printf("pppp = %f %fIn",xGm(e),
            // tempFloatLocal.GetValue(e));
            // 2、计算
            AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
            AscendC::Add(accumulateLocal, accumulateLocal, tempFloatLocal,
                         this->SIZE);
            tempQueueHalf.FreeTensor(tempHalfLocal);
            tempQueueFloat.FreeTensor(tempFloatLocal);
          }
          // 3、取对数
          AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
          AscendC::Cast(accumulateHalfLocal, accumulateLocal,
                        AscendC::RoundMode::CAST_NONE, SIZE);
          // 4. 写回
          accumulateQueueHalf.EnQue(accumulateHalfLocal);
          accumulateHalfLocal = accumulateQueueHalf.DeQue<DTYPE_Y>();
          accumulateHalfLocal.GetValue(0);
          AscendC::DataCopy(yGm[i * SIZE], accumulateHalfLocal, this->SIZE);
          accumulateQueueFloat.FreeTensor(accumulateLocal);
          accumulateQueueHalf.FreeTensor(accumulateHalfLocal);
        }
      } else if (this->axes == 2 || this->axes == -1) {
        // uint32_t all_size = 1;
        // uint32_t loop = all_size / this->SIZE;
        // uint32_t remain = all_size % this->SIZE;
        // if (remain != 0) loop++;
        for (int k2 = 0; k2 < inputShape[0]; ++k2) {
          for (int k1 = 0; k1 < inputShape[1]; ++k1) {
            AscendC::LocalTensor<float> accumulateLocal =
                accumulateQueueFloat.AllocTensor<float>();
            AscendC::LocalTensor<DTYPE_Y> accumulateHalfLocal =
                accumulateQueueHalf.AllocTensor<DTYPE_Y>();
            AscendC::LocalTensor<float> workLocal =
                workQueue.AllocTensor<float>();
            // AscendC::Duplicate<float>(accumulateLocal, static_cast<float>(0),
            // this->SIZE); for (int32_t j = 0; j < inputShape[2]; ++j) {
            AscendC::LocalTensor<DTYPE_X> tempHalfLocal =
                tempQueueHalf.AllocTensor<DTYPE_X>();
            AscendC::LocalTensor<float> tempFloatLocal =
                tempQueueFloat.AllocTensor<float>();
            // 1、拷贝数据
            AscendC::DataCopy(
                tempHalfLocal,
                xGm[k2 * inputShape[1] * inputShape[2] + k1 * inputShape[2]],
                SIZE);
            tempQueueHalf.EnQue(tempHalfLocal);
            tempHalfLocal = tempQueueHalf.DeQue<DTYPE_X>();
            AscendC::Cast(tempFloatLocal, tempHalfLocal,
                          AscendC::RoundMode::CAST_NONE, SIZE);
            // AscendC::printf("pppp = %f %fIn",xGm(e),
            // tempFloatLocal.GetValue(e));
            // 2、计算
            AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
            AscendC::ReduceSum(accumulateLocal, tempFloatLocal, workLocal,
                               inputShape[2]);
            // AscendC::Add(accumulateLocal, accumulateLocal, tempFloatLocal,
            // this->SIZE);
            tempQueueHalf.FreeTensor(tempHalfLocal);
            workQueue.FreeTensor(workLocal);
            tempQueueFloat.FreeTensor(tempFloatLocal);
            // }
            // 3、取对数
            AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
            AscendC::Cast(accumulateHalfLocal, accumulateLocal,
                          AscendC::RoundMode::CAST_NONE, SIZE);
            // 4. 写回
            accumulateQueueHalf.EnQue(accumulateHalfLocal);
            accumulateHalfLocal = accumulateQueueHalf.DeQue<DTYPE_Y>();
            accumulateHalfLocal.GetValue(0);
            AscendC::DataCopy(yGm[k2 * inputShape[1] + k1], accumulateHalfLocal,
                              this->SIZE);
            // AscendC::DataCopy(yGm[k2 * inputShape[1]], accumulateHalfLocal,
            // this->SIZE);
            accumulateQueueFloat.FreeTensor(accumulateLocal);
            accumulateQueueHalf.FreeTensor(accumulateHalfLocal);
          }
        }
      }
    } else if (this->dataType == 0 && this->dimension == 3) {
      if (this->axes == 1 || this->axes == -2) {
        uint32_t all_size = inputShape[2];
        uint32_t loop = all_size / this->SIZE;
        uint32_t remain = all_size % this->SIZE;
        if (remain != 0) loop++;
        for (int k = 0; k < inputShape[0]; ++k) {
          for (int32_t i = 0; i < loop; ++i) {
            AscendC::LocalTensor<DTYPE_Y> accumulateLocal =
                accumulateQueueFloat.AllocTensor<DTYPE_Y>();
            AscendC::Duplicate<DTYPE_Y>(accumulateLocal,
                                        static_cast<DTYPE_Y>(0), this->SIZE);
            for (int32_t j = 0; j < inputShape[1]; ++j) {
              // 1、拷贝数据
              AscendC::LocalTensor<DTYPE_Y> tempFloatLocal =
                  tempQueueFloat.AllocTensor<DTYPE_Y>();
              AscendC::DataCopy(tempFloatLocal,
                                xGm[i * SIZE + j * all_size +
                                    k * inputShape[1] * inputShape[2]],
                                SIZE);
              tempQueueFloat.EnQue(tempFloatLocal);
              tempFloatLocal = tempQueueFloat.DeQue<DTYPE_Y>();
              // AscendC::printf("pppp = %f %fIn",xGm(e),
              // tempFloatLocal.GetValue(e));
              // 2、计算
              AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
              AscendC::Add(accumulateLocal, accumulateLocal, tempFloatLocal,
                           this->SIZE);
              tempQueueFloat.FreeTensor(tempFloatLocal);
            }
            // 3、取对数
            AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
            // 4. 写回
            accumulateQueueFloat.EnQue(accumulateLocal);
            accumulateLocal = accumulateQueueFloat.DeQue<DTYPE_Y>();
            accumulateLocal.GetValue(0);
            AscendC::DataCopy(yGm[i * SIZE + k * inputShape[2]],
                              accumulateLocal, this->SIZE);
            accumulateQueueFloat.FreeTensor(accumulateLocal);
          }
        }
      } else if (this->axes == 0 || this->axes == -3) {
        uint32_t all_size = inputShape[1] * inputShape[2];
        uint32_t loop = all_size / this->SIZE;
        uint32_t remain = all_size % this->SIZE;
        if (remain != 0) loop++;
        for (int32_t i = 0; i < loop; ++i) {
          AscendC::LocalTensor<DTYPE_Y> accumulateLocal =
              accumulateQueueFloat.AllocTensor<DTYPE_Y>();
          AscendC::Duplicate<DTYPE_Y>(accumulateLocal, static_cast<DTYPE_Y>(0),
                                      this->SIZE);
          for (int32_t j = 0; j < inputShape[0]; ++j) {
            // 1、拷贝数据
            AscendC::LocalTensor<DTYPE_Y> tempFloatLocal =
                tempQueueFloat.AllocTensor<DTYPE_Y>();
            AscendC::DataCopy(tempFloatLocal, xGm[i * SIZE + j * all_size],
                              SIZE);
            tempQueueFloat.EnQue(tempFloatLocal);
            tempFloatLocal = tempQueueFloat.DeQue<DTYPE_Y>();
            // AscendC::printf("pppp = %f %fIn",xGm(e),
            // tempFloatLocal.GetValue(e));
            // 2、计算
            AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
            AscendC::Add(accumulateLocal, accumulateLocal, tempFloatLocal,
                         this->SIZE);
            tempQueueFloat.FreeTensor(tempFloatLocal);
          }
          // 3、取对数
          AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
          // 4. 写回
          accumulateQueueFloat.EnQue(accumulateLocal);
          accumulateLocal = accumulateQueueFloat.DeQue<DTYPE_Y>();
          accumulateLocal.GetValue(0);
          AscendC::DataCopy(yGm[i * SIZE], accumulateLocal, this->SIZE);
          accumulateQueueFloat.FreeTensor(accumulateLocal);
        }
      } else if (this->axes == 2 || this->axes == -1) {
        uint32_t all_size = 1;
        uint32_t loop = all_size / this->SIZE;
        uint32_t remain = all_size % this->SIZE;
        if (remain != 0) loop++;
        for (int k2 = 0; k2 < inputShape[0]; ++k2) {
          for (int k1 = 0; k1 < inputShape[1]; ++k1) {
            AscendC::LocalTensor<DTYPE_Y> accumulateLocal =
                accumulateQueueFloat.AllocTensor<DTYPE_Y>();
            // AscendC::Duplicate<DTYPE_Y>(accumulateLocal,
            // static_cast<DTYPE_Y>(0), this->SIZE);
            AscendC::LocalTensor<DTYPE_X> workLocal =
                workQueue.AllocTensor<DTYPE_X>();
            AscendC::LocalTensor<DTYPE_X> tempFloatLocal =
                tempQueueFloat.AllocTensor<DTYPE_X>();
            // for (int32_t j = 0; j < inputShape[2]; ++j) {
            // 1、拷贝数据
            // AscendC::LocalTensor<DTYPE_Y> tempFloatLocal =
            // tempQueueFloat.AllocTensor<DTYPE_Y>();
            // AscendC::DataCopy(tempFloatLocal, xGm[i * SIZE + j * all_size],
            // SIZE);
            AscendC::DataCopy(
                tempFloatLocal,
                xGm[k2 * inputShape[1] * inputShape[2] + k1 * inputShape[2]],
                SIZE);
            tempQueueFloat.EnQue(tempFloatLocal);
            tempFloatLocal = tempQueueFloat.DeQue<DTYPE_X>();
            // AscendC::printf("pppp = %f %fIn",xGm(e),
            // tempFloatLocal.GetValue(e));
            // 2、计算
            AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
            AscendC::ReduceSum(accumulateLocal, tempFloatLocal, workLocal,
                               inputShape[2]);
            // float t = static_cast<float>(accumulateLocal.GetValue(0));
            // sum += t;
            // AscendC::Add(accumulateLocal, accumulateLocal, tempFloatLocal,
            // this->SIZE); tempQueueFloat.FreeTensor(tempFloatLocal);

            // }
            // 3、取对数
            AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
            // 4. 写回
            accumulateQueueFloat.EnQue(accumulateLocal);
            accumulateLocal = accumulateQueueFloat.DeQue<DTYPE_Y>();
            accumulateLocal.GetValue(0);
            AscendC::DataCopy(yGm[k2 * inputShape[1] + k1], accumulateLocal,
                              this->SIZE);
            // AscendC::DataCopy(yGm[i * SIZE + k1 * inputShape[1] *
            // inputShape[3] + k * inputShape[3]], accumulateLocal, this->SIZE);
            tempQueueFloat.FreeTensor(tempFloatLocal);
            workQueue.FreeTensor(workLocal);
            accumulateQueueFloat.FreeTensor(accumulateLocal);
          }
        }
      }
    } else if (this->dataType == 0 && this->dimension == 4) {
      if (this->axes == 1 || this->axes == -3) {
        uint32_t all_size = inputShape[2] * inputShape[3];
        uint32_t loop = all_size / this->SIZE;
        uint32_t remain = all_size % this->SIZE;
        // if (remain != 0) loop++;
        for (int k = 0; k < inputShape[0]; ++k) {
          for (int32_t i = 0; i < loop; ++i) {
            AscendC::LocalTensor<DTYPE_Y> accumulateLocal =
                accumulateQueueFloat.AllocTensor<DTYPE_Y>();
            AscendC::Duplicate<DTYPE_Y>(accumulateLocal,
                                        static_cast<DTYPE_Y>(0), this->SIZE);
            for (int32_t j = 0; j < inputShape[1]; ++j) {
              // 1、拷贝数据
              AscendC::LocalTensor<DTYPE_Y> tempFloatLocal =
                  tempQueueFloat.AllocTensor<DTYPE_Y>();
              AscendC::DataCopy(
                  tempFloatLocal,
                  xGm[i * SIZE + j * all_size +
                      k * inputShape[1] * inputShape[2] * inputShape[3]],
                  SIZE);
              tempQueueFloat.EnQue(tempFloatLocal);
              tempFloatLocal = tempQueueFloat.DeQue<DTYPE_Y>();
              // AscendC::printf("pppp = %f %fIn",xGm(e),
              // tempFloatLocal.GetValue(e));
              // 2、计算
              AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
              AscendC::Add(accumulateLocal, accumulateLocal, tempFloatLocal,
                           this->SIZE);
              tempQueueFloat.FreeTensor(tempFloatLocal);
            }
            // 3、取对数
            AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
            // 4. 写回
            accumulateQueueFloat.EnQue(accumulateLocal);
            accumulateLocal = accumulateQueueFloat.DeQue<DTYPE_Y>();
            accumulateLocal.GetValue(0);
            AscendC::DataCopy(yGm[i * SIZE + k * inputShape[2] * inputShape[3]],
                              accumulateLocal, this->SIZE);
            accumulateQueueFloat.FreeTensor(accumulateLocal);
          }
          if (remain) {
            AscendC::LocalTensor<DTYPE_Y> accumulateLocal =
                accumulateQueueFloat.AllocTensor<DTYPE_Y>();
            AscendC::Duplicate<DTYPE_Y>(accumulateLocal,
                                        static_cast<DTYPE_Y>(0), this->SIZE);
            for (int32_t j = 0; j < inputShape[1]; ++j) {
              // 1、拷贝数据
              AscendC::LocalTensor<DTYPE_Y> tempFloatLocal =
                  tempQueueFloat.AllocTensor<DTYPE_Y>();
              AscendC::DataCopy(
                  tempFloatLocal,
                  xGm[loop * SIZE + j * all_size +
                      k * inputShape[1] * inputShape[2] * inputShape[3]],
                  SIZE);
              tempQueueFloat.EnQue(tempFloatLocal);
              tempFloatLocal = tempQueueFloat.DeQue<DTYPE_Y>();
              // 2、计算
              AscendC::Exp(tempFloatLocal, tempFloatLocal, remain);
              AscendC::Add(accumulateLocal, accumulateLocal, tempFloatLocal,
                           remain);
              tempQueueFloat.FreeTensor(tempFloatLocal);
            }
            // 3、取对数
            AscendC::Ln(accumulateLocal, accumulateLocal, remain);
            // 4. 写回
            accumulateQueueFloat.EnQue(accumulateLocal);
            accumulateLocal = accumulateQueueFloat.DeQue<DTYPE_Y>();
            // AscendC::DataCopy(yGm[i * SIZE + k * inputShape[2] *
            // inputShape[3]], accumulateLocal, this->SIZE);
            accumulateLocal.GetValue(0);
            AscendC::DataCopy(
                yGm[loop * SIZE + k * inputShape[2] * inputShape[3]],
                accumulateLocal, remain);
            accumulateQueueFloat.FreeTensor(accumulateLocal);
          }
        }
      } else if (this->axes == 0 || this->axes == -4) {
        uint32_t all_size = inputShape[1] * inputShape[2] * inputShape[3];
        uint32_t loop = all_size / this->SIZE;
        uint32_t remain = all_size % this->SIZE;
        if (remain != 0) loop++;
        // for (int k = 0; k < inputShape[0]; ++k) {
        for (int32_t i = 0; i < loop; ++i) {
          AscendC::LocalTensor<DTYPE_Y> accumulateLocal =
              accumulateQueueFloat.AllocTensor<DTYPE_Y>();
          AscendC::Duplicate<DTYPE_Y>(accumulateLocal, static_cast<DTYPE_Y>(0),
                                      this->SIZE);
          for (int32_t j = 0; j < inputShape[0]; ++j) {
            // 1、拷贝数据
            AscendC::LocalTensor<DTYPE_Y> tempFloatLocal =
                tempQueueFloat.AllocTensor<DTYPE_Y>();
            AscendC::DataCopy(tempFloatLocal, xGm[i * SIZE + j * all_size],
                              SIZE);
            tempQueueFloat.EnQue(tempFloatLocal);
            tempFloatLocal = tempQueueFloat.DeQue<DTYPE_Y>();
            // AscendC::printf("pppp = %f %fIn",xGm(e),
            // tempFloatLocal.GetValue(e));
            // 2、计算
            AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
            AscendC::Add(accumulateLocal, accumulateLocal, tempFloatLocal,
                         this->SIZE);
            tempQueueFloat.FreeTensor(tempFloatLocal);
          }
          // 3、取对数
          AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
          // 4. 写回
          accumulateQueueFloat.EnQue(accumulateLocal);
          accumulateLocal = accumulateQueueFloat.DeQue<DTYPE_Y>();
          accumulateLocal.GetValue(0);
          AscendC::DataCopy(yGm[i * SIZE], accumulateLocal, this->SIZE);
          accumulateQueueFloat.FreeTensor(accumulateLocal);
        }
      } else if (this->axes == 2 || this->axes == -2) {
        uint32_t all_size = inputShape[3];
        uint32_t loop = all_size / this->SIZE;
        uint32_t remain = all_size % this->SIZE;
        if (remain != 0) loop++;
        for (int k1 = 0; k1 < inputShape[0]; ++k1) {
          for (int k = 0; k < inputShape[1]; ++k) {
            for (int32_t i = 0; i < loop; ++i) {
              AscendC::LocalTensor<DTYPE_Y> accumulateLocal =
                  accumulateQueueFloat.AllocTensor<DTYPE_Y>();
              AscendC::Duplicate<DTYPE_Y>(accumulateLocal,
                                          static_cast<DTYPE_Y>(0), this->SIZE);
              for (int32_t j = 0; j < inputShape[2]; ++j) {
                // 1、拷贝数据
                AscendC::LocalTensor<DTYPE_Y> tempFloatLocal =
                    tempQueueFloat.AllocTensor<DTYPE_Y>();
                // AscendC::DataCopy(tempFloatLocal, xGm[i * SIZE + j *
                // all_size], SIZE);
                AscendC::DataCopy(
                    tempFloatLocal,
                    xGm[i * SIZE + j * all_size +
                        k1 * inputShape[1] * inputShape[2] * inputShape[3] +
                        k * inputShape[2] * inputShape[3]],
                    SIZE);
                tempQueueFloat.EnQue(tempFloatLocal);
                tempFloatLocal = tempQueueFloat.DeQue<DTYPE_Y>();
                // AscendC::printf("pppp = %f %fIn",xGm(e),
                // tempFloatLocal.GetValue(e));
                // 2、计算
                AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
                AscendC::Add(accumulateLocal, accumulateLocal, tempFloatLocal,
                             this->SIZE);
                tempQueueFloat.FreeTensor(tempFloatLocal);
              }
              // 3、取对数
              AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
              // 4. 写回
              accumulateQueueFloat.EnQue(accumulateLocal);
              accumulateLocal = accumulateQueueFloat.DeQue<DTYPE_Y>();
              accumulateLocal.GetValue(0);
              AscendC::DataCopy(
                  yGm[i * SIZE + k1 * inputShape[1] * inputShape[3] +
                      k * inputShape[3]],
                  accumulateLocal, this->SIZE);
              // AscendC::DataCopy(yGm[i * SIZE], accumulateLocal, this->SIZE);
              accumulateQueueFloat.FreeTensor(accumulateLocal);
            }
          }
        }
      } else if (this->axes == 3 || this->axes == -1) {
        for (int k2 = 0; k2 < inputShape[0]; ++k2) {
          for (int k1 = 0; k1 < inputShape[1]; ++k1) {
            for (int k = 0; k < inputShape[2]; ++k) {
              AscendC::LocalTensor<DTYPE_Y> accumulateLocal =
                  accumulateQueueFloat.AllocTensor<DTYPE_Y>();
              // AscendC::Duplicate<DTYPE_Y>(accumulateLocal,
              // static_cast<DTYPE_Y>(0), this->SIZE);
              AscendC::LocalTensor<DTYPE_X> workLocal =
                  workQueue.AllocTensor<DTYPE_X>();
              // for (int32_t j = 0; j < inputShape[3]; ++j) {
              // 1、拷贝数据
              AscendC::LocalTensor<DTYPE_X> tempFloatLocal =
                  tempQueueFloat.AllocTensor<DTYPE_X>();
              // AscendC::LocalTensor<DTYPE_Y> tempFloatLocal =
              // tempQueueFloat.AllocTensor<DTYPE_Y>();
              // AscendC::DataCopy(tempFloatLocal, xGm[i * SIZE + j * all_size],
              // SIZE);
              AscendC::DataCopy(
                  tempFloatLocal,
                  xGm[k2 * inputShape[1] * inputShape[2] * inputShape[3] +
                      k1 * inputShape[2] * inputShape[3] + k * inputShape[3]],
                  SIZE);
              tempQueueFloat.EnQue(tempFloatLocal);
              tempFloatLocal = tempQueueFloat.DeQue<DTYPE_X>();
              // AscendC::printf("pppp = %f %fIn",xGm(e),
              // tempFloatLocal.GetValue(e));
              // 2、计算
              AscendC::Exp(tempFloatLocal, tempFloatLocal, this->SIZE);
              AscendC::ReduceSum(accumulateLocal, tempFloatLocal, workLocal,
                                 inputShape[3]);
              // }
              // 3、取对数
              AscendC::Ln(accumulateLocal, accumulateLocal, this->SIZE);
              // 4. 写回
              accumulateQueueFloat.EnQue(accumulateLocal);
              accumulateLocal = accumulateQueueFloat.DeQue<DTYPE_Y>();
              accumulateLocal.GetValue(0);
              AscendC::DataCopy(yGm[k2 * inputShape[1] * inputShape[2] +
                                    k1 * inputShape[2] + k],
                                accumulateLocal, this->SIZE);
              // AscendC::DataCopy(yGm[i * SIZE + k1 * inputShape[1] *
              // inputShape[3] + k * inputShape[3]], accumulateLocal,
              // this->SIZE);
              tempQueueFloat.FreeTensor(tempFloatLocal);
              workQueue.FreeTensor(workLocal);
              accumulateQueueFloat.FreeTensor(accumulateLocal);
            }
          }
        }
      }
    }
  }

 private:
  AscendC::TPipe pipe;
  uint32_t dimension = 0;
  uint32_t dataType = 0;
  uint32_t SIZE = 0;
  int32_t axes = 0;
  AscendC::GlobalTensor<DTYPE_AXES> axesGm;
  AscendC::GlobalTensor<DTYPE_X> xGm;
  AscendC::GlobalTensor<DTYPE_Y> yGm;
  AscendC::TBuf<AscendC::TPosition::VECCALC> temp1Float, temp2Float, tempBits;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> accumulateQueueFloat;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> tempQueueFloat;
  AscendC::TQue<AscendC::QuePosition::VECCALC, BUFFER_NUM> maxQueueFloat;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> accumulateQueueHalf;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> tempQueueHalf;
  AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> workQueue;
};

extern "C" __global__ __aicore__ void reduce_log_sum_exp(
    GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  // TODO: user kernel impl
  KernelLogSumExpReduce op;
  op.Init(x, axes, y, tiling_data.dimension, tiling_data.dataSize,
          tiling_data.dataType);
  op.Process(tiling_data.inputShape);
}