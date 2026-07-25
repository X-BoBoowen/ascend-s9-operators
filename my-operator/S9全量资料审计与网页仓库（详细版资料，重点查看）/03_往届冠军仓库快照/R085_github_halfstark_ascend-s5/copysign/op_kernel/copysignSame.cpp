#include "kernel_operator.h"
#include "copybig.h"
#include "copybf16.h"
#include <type_traits>
// #include <bfloat16.h>
// #include <iostream>
#define BUFFER_NUM 2
using namespace AscendC;

template<typename T> class KernelCopySignSame{
public:
    __aicore__ inline KernelCopySignSame(){}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR output, uint64_t tailBlockNum, uint64_t piter, uint64_t smallDataNum,TPipe* pipeIn){
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
        globalIndex *= piter;
        uint64_t totalLength = coreDataNum*piter;
        // printf("globalIndex:%d totalLength: %d coreDataNum:%d piter:%d\n",globalIndex, totalLength, coreDataNum,piter);
        inputGm.SetGlobalBuffer((__gm__ T*)input + globalIndex, totalLength);
        otherGm.SetGlobalBuffer((__gm__ T*)other, piter);
        outputGm.SetGlobalBuffer((__gm__ T*)output + globalIndex, totalLength);
        pipeIn->InitBuffer(inQueueInput, BUFFER_NUM, piter*sizeof(T));
        pipeIn->InitBuffer(inQueueOther, BUFFER_NUM, piter*sizeof(T));
        pipeIn->InitBuffer(outQueue, BUFFER_NUM, piter*sizeof(T));
         // if constexpr(!std::is_same_v<DTYPE_INPUT, float>) {
         //    pipeIn->InitBuffer(negIn, piter*sizeof(T));
         // }
        // otherLocal = inQueueOther.AllocTensor<T>();
        // // TODO: check 是否需要-1
        // // printf("stride: %d\n", this->stride*1);
        // AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        // AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        // AscendC::DataCopyPad(otherLocal, otherGm, copyParams, padParams); 
    }
    __aicore__ void copyIn(uint64_t progress) {
        LocalTensor<T>inputLocal = inQueueInput.AllocTensor<T>();
        // TODO: check 是否需要-1
        // printf("stride: %d\n", this->stride*1);
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(inputLocal, inputGm[progress*piter], copyParams, padParams); // 从GM-
        AscendC::DataCopyPad(otherLocal, otherGm[progress*piter], copyParams, padParams); 

        inQueueInput.EnQue<T>(inputLocal);
        inQueueOther.EnQue<T>(otherLocal);

    }
    
    __aicore__ void compute(uint64_t progress) {
        LocalTensor<T>inputLocal = inQueueInput.DeQue<T>();
        LocalTensor<T>otherLocal = inQueueOther.DeQue<T>();

        LocalTensor<T>outputLocal = outQueue.AllocTensor<T>();
        AscendC::Abs(inputLocal, inputLocal, piter);
         if constexpr(!std::is_same_v<T, float>) {
            for (uint64_t i = 0; i < piter; i++) {
                LocalTensor<T>negLocal = negIn.Get<T>();
                T scalar = -1;
                AscendC::Muls(negLocal, inputLocal, scalar, piter);
                T absValue =  inputLocal.GetValue(i);
                T value = otherLocal.GetValue(i);
                float now = value;
                // float sign = *reinterpret_cast<float*>(&value);
                // printf("%f ", now);
                if (now < 0) {
                   outputLocal.SetValue(i, negLocal.GetValue(i));
                } else {
                   outputLocal.SetValue(i, absValue);
                }
            }
         } else {
            for (uint64_t i = 0; i < piter; i++) {
                T absValue =  inputLocal.GetValue(i);
                T value = otherLocal.GetValue(i);
                float sign = *reinterpret_cast<float*>(&value);
                if (sign < 0) {
                   outputLocal.SetValue(i, -absValue);
                } else {
                   outputLocal.SetValue(i, absValue);
                }
            }
         }
        inQueueInput.FreeTensor(inputLocal);
        outQueue.EnQue<T>(outputLocal);
    }
    __aicore__ void copyOut(uint64_t progress) {
        LocalTensor<T>outputLocal = outQueue.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        // AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad( outputGm[progress*piter],outputLocal, copyParams);
        outQueue.FreeTensor(outputLocal);

    }
    __aicore__ inline void process() {
        for (int i = 0; i < coreDataNum; i++) {
                copyIn(i);
                compute(i);
                copyOut(i);
        }
    }
    
    
private:
    // AscendC::TPipe* pipe;
    uint64_t piter,coreDataNum;
    AscendC::GlobalTensor<T> inputGm, otherGm, outputGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInput,inQueueOther;
    TBuf<QuePosition::VECCALC> negIn;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    // LocalTensor<T>otherLocal;

};



// extern "C" __global__ __aicore__ void copysign(GM_ADDR input, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
//     GET_TILING_DATA(tiling_data, tiling);
//           TPipe pipe;
    

//     // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 增加这一行
//     if (TILING_KEY_IS(1)) {
//         if constexpr(std::is_same_v<DTYPE_INPUT, float16_t>||std::is_same_v<DTYPE_INPUT, float>){
//                     KernelCopySign<DTYPE_INPUT>op;
//             op.Init(input, other, out, tiling_data.tail, tiling_data.periter, tiling_data.smallBatch, &pipe);
//             op.process();

//         } else {
//             KernelCopySign<float16_t>op;
//             op.Init(input, other, out, tiling_data.tail, tiling_data.periter, tiling_data.smallBatch, &pipe);
//             op.process();
//         }
//     }  
//     if (TILING_KEY_IS(2)) {
//         if constexpr(std::is_same_v<DTYPE_INPUT, float16_t>||std::is_same_v<DTYPE_INPUT, float>){
//             KernelCopySignBig<DTYPE_INPUT>op;
//             op.Init(input, other, out, tiling_data.tail, tiling_data.periter,tiling_data.remainPer, tiling_data.smallBatch, &pipe);
//             op.process();

//         } else {
//             KernelCopySignBig<float16_t>op;
//             op.Init(input, other, out, tiling_data.tail, tiling_data.periter, tiling_data.remainPer,tiling_data.smallBatch, &pipe);
//             op.process();
//         }

//     }
//      if (TILING_KEY_IS(3))  {
//             KernelCopySignBig16<float16_t>op;
//             op.Init(input, other, out, tiling_data.tail, tiling_data.periter, tiling_data.remainPer,tiling_data.smallBatch, &pipe);
//             op.process();
//      }
//     // TODO: user kernel impl
// }