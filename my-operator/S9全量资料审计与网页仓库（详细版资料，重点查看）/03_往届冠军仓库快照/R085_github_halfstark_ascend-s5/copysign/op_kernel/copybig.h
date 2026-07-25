#include "kernel_operator.h"
#include <type_traits>
// #include <bfloat16.h>
// #include <iostream>
#define BUFFER_NUM 1
using namespace AscendC;
template<typename T> class KernelCopySignBig{
public:
    __aicore__ inline KernelCopySignBig(){}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR output, uint64_t tailBlockNum, uint64_t piter,uint64_t remainPiter, uint64_t smallDataNum,TPipe* pipeIn){
        
        // this->pipe = pipe;
        this->piter = piter;
        uint64_t bigDataNum = smallDataNum + 1; 
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint64_t globalIndex = bigDataNum * coreNum;
        if (coreNum < tailBlockNum) {
            this->coreDataNum = bigDataNum;
        } else {
            this->coreDataNum = smallDataNum;
            globalIndex -= (bigDataNum - smallDataNum)*(coreNum - tailBlockNum);
        }
        this->remainPiter = remainPiter;
        this->whole = piter*remainPiter;
        globalIndex *= whole;

        uint64_t totalLength = coreDataNum*whole;
        // printf("globalIndex:%d totalLength: %d coreDataNum:%d piter:%d remainPiter:%d\n",globalIndex, totalLength, coreDataNum,piter,remainPiter);
        inputGm.SetGlobalBuffer((__gm__ T*)input + globalIndex, totalLength);
        otherGm.SetGlobalBuffer((__gm__ T*)other, whole);
        outputGm.SetGlobalBuffer((__gm__ T*)output + globalIndex, totalLength);
        pipeIn->InitBuffer(inQueueInput, BUFFER_NUM, piter*sizeof(T));
        pipeIn->InitBuffer(inQueueOther, BUFFER_NUM, piter*sizeof(T));
        pipeIn->InitBuffer(outQueue, BUFFER_NUM, piter*sizeof(T));
         // if constexpr(!std::is_same_v<DTYPE_INPUT, float>) {
        pipeIn->InitBuffer(sign, piter*sizeof(T));
        signLocal = sign.Get<T>();
         // }
        // otherLocal = inQueueOther.AllocTensor<T>();
        // TODO: check 是否需要-1
        // printf("stride: %d\n", this->stride*1);
        // AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        // AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        // AscendC::DataCopyPad(otherLocal, otherGm, copyParams, padParams); 
    }
    __aicore__ void copyIn(uint64_t progress, uint64_t sub) {
        LocalTensor<T>inputLocal = inQueueInput.AllocTensor<T>();
        LocalTensor<T>otherLocal = inQueueOther.AllocTensor<T>();

        // TODO: check 是否需要-1
        // printf("stride: %d\n", this->stride*1);
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        // printf("sub: %d\n", sub);
        AscendC::DataCopyPad(inputLocal, inputGm[progress*whole + sub*piter], copyParams, padParams); // 从GM-
        AscendC::DataCopyPad(otherLocal, otherGm[sub*piter], copyParams, padParams); // 从GM-

        inQueueInput.EnQue<T>(inputLocal);
        inQueueOther.EnQue<T>(otherLocal);
    }
    
    __aicore__ void compute(uint64_t progress, uint64_t sub) {
        LocalTensor<T>inputLocal = inQueueInput.DeQue<T>();
        LocalTensor<T>otherLocal = inQueueOther.DeQue<T>();

        LocalTensor<T>outputLocal = outQueue.AllocTensor<T>();
        AscendC::Abs(inputLocal, inputLocal, piter);
        AscendC::Abs(signLocal, otherLocal, piter);
        AscendC::Div(signLocal, signLocal, otherLocal, piter);
        outputLocal = inputLocal * signLocal;    
        //  if constexpr(!std::is_same_v<T, float>) {
        // LocalTensor<T>negLocal = negIn.Get<T>();
        //                      T scalar = -1;
        // AscendC::Muls(negLocal, inputLocal, scalar, piter);
        //     for (uint64_t i = 0; i < piter; i++) {
        //         T absValue =  inputLocal.GetValue(i);
        //         T value = otherLocal.GetValue(i);
        //         float now = value;
        //         // float sign = *reinterpret_cast<float*>(&value);
        //         // printf("%f ", now);
        //         if (now < 0) {
        //            outputLocal.SetValue(i, negLocal.GetValue(i));
        //         } else {
        //            outputLocal.SetValue(i, absValue);
        //         }
        //     }
        //  } else {
        //     for (uint64_t i = 0; i < piter; i++) {
        //         T absValue =  inputLocal.GetValue(i);
        //         T value = otherLocal.GetValue(i);
        //         float sign = *reinterpret_cast<float*>(&value);
        //         if (sign < 0) {
        //            outputLocal.SetValue(i, -absValue);
        //         } else {
        //            outputLocal.SetValue(i, absValue);
        //         }
        //     }
        //  }
        inQueueInput.FreeTensor(inputLocal);
        inQueueOther.FreeTensor(otherLocal);

        outQueue.EnQue<T>(outputLocal);
    }
    __aicore__ void copyOut(uint64_t progress, uint64_t sub) {
        LocalTensor<T>outputLocal = outQueue.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        // AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad( outputGm[progress*whole + sub*piter],outputLocal, copyParams);
        outQueue.FreeTensor(outputLocal);

    }
    __aicore__ inline void process() {
        for (uint64_t i = 0; i < coreDataNum; i++) {
                for (uint64_t j = 0; j < remainPiter; j++) {
                    copyIn(i, j);
                    compute(i, j);
                    copyOut(i, j);
                }
        }
    }
    
    
private:
    // AscendC::TPipe* pipe;
    uint64_t piter,coreDataNum,remainPiter, whole;
    AscendC::GlobalTensor<T> inputGm, otherGm, outputGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInput,inQueueOther;
    TBuf<QuePosition::VECCALC> sign;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    LocalTensor<T>signLocal;

};
