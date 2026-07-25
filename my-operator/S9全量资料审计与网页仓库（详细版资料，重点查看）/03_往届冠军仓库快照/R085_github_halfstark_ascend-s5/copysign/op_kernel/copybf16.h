#include "kernel_operator.h"
#include <type_traits>
// #include <bfloat16.h>
// #include <iostream>
#define BUFFER_NUM 2
using namespace AscendC;
template<typename T> class KernelCopySignBig16{
public:
    __aicore__ inline KernelCopySignBig16(){}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR output, uint32_t tailBlockNum, uint32_t piter,uint32_t remainPiter, uint32_t smallDataNum,TPipe* pipeIn){
        // this->pipe = pipe;
        this->piter = piter;
        uint32_t bigDataNum = smallDataNum + 1; 
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalIndex = bigDataNum * coreNum;
        if (coreNum < tailBlockNum) {
            this->coreDataNum = bigDataNum;
        } else {
            this->coreDataNum = smallDataNum;
            globalIndex -= (bigDataNum - smallDataNum)*(coreNum - tailBlockNum);
        }
        this->remainPiter = remainPiter;
        this->whole = piter*remainPiter;
        globalIndex *= whole;

        uint32_t totalLength = coreDataNum*whole;
        // printf("globalIndex:%d totalLength: %d coreDataNum:%d piter:%d remainPiter:%d\n",globalIndex, totalLength, coreDataNum,piter,remainPiter);
        inputGm.SetGlobalBuffer((__gm__ T*)input + globalIndex, totalLength);
        otherGm.SetGlobalBuffer((__gm__ T*)other, whole);
        outputGm.SetGlobalBuffer((__gm__ T*)output + globalIndex, totalLength);
        pipeIn->InitBuffer(inQueueInput, BUFFER_NUM, piter*sizeof(T));
        pipeIn->InitBuffer(inQueueOther, BUFFER_NUM, piter*sizeof(T));
        pipeIn->InitBuffer(outQueue, BUFFER_NUM, piter*sizeof(T));
        pipeIn->InitBuffer(castBuf,  piter*sizeof(float));
        // otherLocal = inQueueOther.AllocTensor<T>();
        // TODO: check 是否需要-1
        // printf("stride: %d\n", this->stride*1);
        // AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        // AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        // AscendC::DataCopyPad(otherLocal, otherGm, copyParams, padParams); 
    }
    __aicore__ void copyIn(int32_t progress, int sub) {
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
    
    __aicore__ void compute(int32_t progress, int sub) {
        LocalTensor<T>inputLocal = inQueueInput.DeQue<T>();
        LocalTensor<T>otherLocal = inQueueOther.DeQue<T>();
        LocalTensor<float>castLocal = castBuf.Get<float>();
        LocalTensor<T>outputLocal = outQueue.AllocTensor<T>();
        AscendC::Cast(castLocal, inputLocal, AscendC::RoundMode::CAST_NONE, piter);
        AscendC::Abs(castLocal, castLocal, piter);
        for (int i = 0; i < piter; i++) {
            T value = otherLocal.GetValue(i);
            float now = value;
            float castValue = castLocal.GetValue(i);
                // float sign = *reinterpret_cast<float*>(&value);
            // printf("%f ", now);
            if (now < 0) {
                castLocal.SetValue(i, -castValue);
            } else {
                castLocal.SetValue(i, castValue);
            }
        }
        // printf("\n");
        inQueueInput.FreeTensor(inputLocal);
        inQueueOther.FreeTensor(otherLocal);
        AscendC::Cast(outputLocal, castLocal, AscendC::RoundMode::CAST_TRUNC, piter);
        outQueue.EnQue<T>(outputLocal);
    }
    __aicore__ void copyOut(int32_t progress, int sub) {
        LocalTensor<T>outputLocal = outQueue.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        // AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad( outputGm[progress*whole + sub*piter],outputLocal, copyParams);
        outQueue.FreeTensor(outputLocal);

    }
    __aicore__ inline void process() {
        for (int i = 0; i < coreDataNum; i++) {
                for (int j = 0; j < remainPiter; j++) {
                    copyIn(i, j);
                    compute(i, j);
                    copyOut(i, j);
                }
        }
    }
    
    
private:
    // AscendC::TPipe* pipe;
    uint32_t piter,coreDataNum,remainPiter, whole;
    AscendC::GlobalTensor<T> inputGm, otherGm, outputGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInput,inQueueOther;
    TBuf<QuePosition::VECCALC> negIn, castBuf;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    // LocalTensor<T>otherLocal;

};
