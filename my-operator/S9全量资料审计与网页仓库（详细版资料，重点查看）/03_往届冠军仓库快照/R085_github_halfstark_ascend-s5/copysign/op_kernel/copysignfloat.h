#include "kernel_operator.h"
#include <type_traits>
// #include <bfloat16.h>
// #include <iostream>
#define BUFFER_NUM 2
using namespace AscendC;

template<typename T> class KernelCopySignFloat{
public:
    __aicore__ inline KernelCopySignFloat(){}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR output, uint64_t tailBlockNum, uint64_t piter, uint64_t smallDataNum,uint64_t mid, uint64_t pre, TPipe* pipeIn){
        // this->pipe = pipe;
        this->piter = piter;
        uint64_t bigDataNum = smallDataNum + 1; 
        uint64_t coreNum = AscendC::GetBlockIdx();
        uint64_t globalIndex = bigDataNum * coreNum;
        if (coreNum < tailBlockNum) {
            this->coreDataNum = bigDataNum;
        } else {
            this->coreDataNum = smallDataNum;
            globalIndex -= (bigDataNum - smallDataNum)*(coreNum - tailBlockNum);
        }
        globalIndex *= (piter*mid*pre);
        uint64_t totalLength = coreDataNum*piter*mid*pre;
        this->mid = mid;
        this->pre = pre;
        printf("globalIndex:%d totalLength: %d coreDataNum:%d piter:%d\n",globalIndex, totalLength, coreDataNum,piter);
        inputGm.SetGlobalBuffer((__gm__ T*)input + globalIndex, totalLength);
        otherGm.SetGlobalBuffer((__gm__ T*)other, piter*pre);
        outputGm.SetGlobalBuffer((__gm__ T*)output + globalIndex, totalLength);
        pipeIn->InitBuffer(inQueueInput, BUFFER_NUM, piter*sizeof(T));
        pipeIn->InitBuffer(inQueueOther, BUFFER_NUM, piter*pre*sizeof(T));
        pipeIn->InitBuffer(outQueue, BUFFER_NUM, piter*sizeof(T));
        otherLocal = inQueueOther.AllocTensor<T>();
        // TODO: check 是否需要-1
        // printf("stride: %d\n", this->stride*1);
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*pre*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(otherLocal, otherGm, copyParams, padParams); 
    }
    __aicore__ void copyIn(uint64_t progress, int sub) {
        LocalTensor<T>inputLocal = inQueueInput.AllocTensor<T>();
        // TODO: check 是否需要-1
        // printf("stride: %d\n", this->stride*1);
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(inputLocal, inputGm[progress*pre*mid*piter + sub*piter], copyParams, padParams); // 从GM-
        inQueueInput.EnQue<T>(inputLocal);
    }
    
    __aicore__ void compute(uint64_t progress, int sub) {
        LocalTensor<T>inputLocal = inQueueInput.DeQue<T>();
        LocalTensor<T>outputLocal = outQueue.AllocTensor<T>();
        AscendC::Abs(inputLocal, inputLocal, piter);
            for (uint64_t i = 0; i < piter; i++) {
                T absValue =  inputLocal.GetValue(i);
                int first = sub/mid;
                T value = otherLocal.GetValue(first*piter+i);
                if (value < 0||value==-0) {
                   outputLocal.SetValue(i, -absValue);
                } else {
                   outputLocal.SetValue(i, absValue);
                }
            }
        inQueueInput.FreeTensor(inputLocal);
        outQueue.EnQue<T>(outputLocal);
    }
    __aicore__ void copyOut(uint64_t progress, int sub) {
        LocalTensor<T>outputLocal = outQueue.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad( outputGm[progress*pre*mid*piter + sub*piter],outputLocal, copyParams);
        outQueue.FreeTensor(outputLocal);

    }
    __aicore__ inline void process() {
        for (int i = 0; i < coreDataNum; i++) {
            for (int j = 0; j < pre*mid;j++) {
                copyIn(i, j);
                compute(i, j);
                copyOut(i, j);
            }
                // for (int j = 0; j < pre; j++) {
                // LocalTensor<T>otherLocal = inQueueOther.AllocTensor<T>();
                // // TODO: check 是否需要-1
                // // printf("stride: %d\n", this->stride*1);
                // AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
                // AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                // AscendC::DataCopyPad(otherLocal, otherGm[progress*piter], copyParams, padParams); // 从GM-
                // inQueueOther.EnQue<T>(otherLocal);
                //     for (int z = 0; z < mid;z++) {
                //         LocalTensor<T>inputLocal = inQueueInput.AllocTensor<T>();
                //         // TODO: check 是否需要-1
                //         // printf("stride: %d\n", this->stride*1);
                //         AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
                //         AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                //         AscendC::DataCopyPad(otherLocal, otherGm[progress*piter], copyParams, padParams); // 从GM-
                //         inQueueInput.EnQue<T>(otherLocal);
                //     }
                // }

        }
    }
    
    
private:
    // AscendC::TPipe* pipe;
    uint64_t piter,coreDataNum,mid, pre;
    AscendC::GlobalTensor<T> inputGm, otherGm, outputGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInput,inQueueOther;
    TBuf<QuePosition::VECCALC> negIn;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    LocalTensor<T>otherLocal;

};

