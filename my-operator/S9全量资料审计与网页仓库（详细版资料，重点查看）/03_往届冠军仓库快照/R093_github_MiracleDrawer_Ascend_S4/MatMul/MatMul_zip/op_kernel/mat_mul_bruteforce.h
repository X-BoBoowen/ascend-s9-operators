#pragma once

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

namespace MatMulBruteforce{

class MatMulBruteforceKernel {
public:
    __aicore__ inline MatMulBruteforceKernel() {}
    
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y, GM_ADDR bias, GM_ADDR z, int32_t batchSize, int32_t mLength, int32_t kLength, int32_t nLength, bool hasBias, int32_t biasBatch, int32_t biasMLength, int32_t biasNLength){
        xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * mLength * kLength * 2);
        yGm.SetGlobalBuffer((__gm__ float*)y, batchSize * nLength * kLength * 2);
        if (hasBias){
            biasGm.SetGlobalBuffer((__gm__ float*)bias, biasBatch * biasMLength * biasNLength * 2);
        }
        zGm.SetGlobalBuffer((__gm__ float*)z, batchSize * mLength * nLength * 2);
        PipeBarrier<PIPE_ALL>();
        int32_t xs = 0;
        while (xs <= (int32_t)500000) xs++;
        for (int32_t nowBatch = 0; nowBatch < batchSize; nowBatch++){
            int32_t initialOffsetX = nowBatch * mLength * kLength * 2;
            int32_t initialOffsetY = nowBatch * nLength * kLength * 2;
            int32_t initialOffsetZ = nowBatch * mLength * nLength * 2;
            for (int32_t i = 0; i < mLength; i++){
                for (int32_t j = 0; j < nLength; j++){ //j*2 && j*2+1
                    float cReal,cImag;
                    cReal = 0;
                    cImag = 0;
                    for (int32_t k = 0; k < kLength; k++){
                        float aReal=0,aImag=0,bReal=0,bImag=0;
                        aReal = xGm.GetValue(initialOffsetX + i * kLength * 2 + k * 2);
                        aImag = xGm.GetValue(initialOffsetX + i * kLength * 2 + k * 2 + 1);
                        bReal = yGm.GetValue(initialOffsetY + k * nLength * 2 + j * 2);
                        bImag = yGm.GetValue(initialOffsetY + k * nLength * 2 + j * 2 + 1);
                        cReal = cReal + aReal * bReal - aImag * bImag;
                        cImag = cImag + aReal * bImag + aImag * bReal;
                    }
                    if (hasBias){
                        int32_t biasB = nowBatch % biasBatch;
                        int32_t biasI = i % biasMLength;
                        int32_t biasJ = j % biasNLength;
                        float biasReal = biasGm.GetValue(biasB * biasNLength * biasMLength * 2 + biasI * biasNLength * 2 + biasJ * 2);
                        float biasImag = biasGm.GetValue(biasB * biasNLength * biasMLength * 2 + biasI * biasNLength * 2 + biasJ * 2 + 1);
                        cReal += biasReal;
                        cImag += biasImag;
                    }
                    zGm.SetValue(initialOffsetZ + i * nLength * 2 + j * 2, cReal);
                    zGm.SetValue(initialOffsetZ + i * nLength * 2 + j * 2 + 1, cImag);
                    // zGm.SetValue(i * nLength * 2 + j * 2, cReal);
                    // zGm.SetValue(i * nLength * 2 + j * 2 + 1, cImag);
                }
            }
        }
    }
    GlobalTensor<float> xGm, yGm, biasGm, zGm;

};

}