/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file glu_jvp_fp32.h
 */
#ifndef GLU_JVP_FP32_H
#define GLU_JVP_FP32_H
#include <type_traits>
#include "kernel_operator.h"

using namespace AscendC;
constexpr uint16_t BufferNum = 1;
class KernelGluJvpFp32 {
public:
    __aicore__ inline KernelGluJvpFp32() {}
    __aicore__ inline void Init(GM_ADDR glu_out, GM_ADDR input, GM_ADDR v, GM_ADDR jvp_out, int HI, int J, int KS, uint32_t smallSize, uint16_t incSize, uint16_t formerNum, TPipe *pipeIn) {
        this->KS = KS;
        int totalSize = HI * J * KS;
        this->HI = HI;
        this->J = J;
        this->pipe = pipeIn;
        this->hi_block_size = J * KS / 2;
        uint32_t beginIndex = 0;
        const uint16_t formerNum_fp32 = formerNum;
        if (GetBlockIdx() < formerNum_fp32) {
            this->size = smallSize + incSize;
            beginIndex = this->size * GetBlockIdx();
        } else {
            this->size = smallSize;
            beginIndex = this->size * GetBlockIdx() + formerNum_fp32 * incSize;
        }

        this->hi_block_num = this->size / this->hi_block_size;
        glu_outGm.SetGlobalBuffer((__gm__ float *)glu_out + beginIndex, size);
        xGm.SetGlobalBuffer((__gm__ float *)input + 2 * beginIndex, 2 * size);
        vGm.SetGlobalBuffer((__gm__ float *)v + 2 * beginIndex, 2 * size);
        jvp_outGm.SetGlobalBuffer((__gm__ float *)jvp_out + beginIndex, size);
        uint32_t spaceSize = min((uint32_t)(hi_block_size * sizeof(float)), (uint32_t)(160 / 5) * 1024 / BufferNum);
        this->n_elements_per_iter = spaceSize / sizeof(float);
        this->bigLoopTimes = hi_block_num;
        this->smallLoopTimes = (hi_block_size + n_elements_per_iter - 1) / n_elements_per_iter;
        pipe->InitBuffer(glu_outBuf, BufferNum, spaceSize + 32);
        pipe->InitBuffer(x1Buf, BufferNum, spaceSize + 32);
        pipe->InitBuffer(v1Buf, BufferNum, spaceSize + 32);
        pipe->InitBuffer(v2Buf, BufferNum, spaceSize + 32);
        pipe->InitBuffer(jvp_outBuf, BufferNum, spaceSize + 32);
    }
    __aicore__ inline void Process_iter(uint32_t inputIndex, uint32_t outIndex, uint32_t iterSize) {
        LocalTensor<float> glu_outLocal = glu_outBuf.AllocTensor<float>();
        LocalTensor<float> x1Local = x1Buf.AllocTensor<float>();
        LocalTensor<float> v1Local = v1Buf.AllocTensor<float>();
        LocalTensor<float> v2Local = v2Buf.AllocTensor<float>();
        uint16_t blockCount = 1;
        uint32_t blockLen = iterSize * sizeof(float);
        DataCopyExtParams copyParams{blockCount, blockLen, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(glu_outLocal, glu_outGm[outIndex], copyParams, padParams);
        DataCopyPad(x1Local, xGm[inputIndex], copyParams, padParams);
        DataCopyPad(v1Local, vGm[inputIndex], copyParams, padParams);
        DataCopyPad(v2Local, vGm[inputIndex + hi_block_size], copyParams, padParams);
        glu_outBuf.EnQue<float>(glu_outLocal);
        x1Buf.EnQue<float>(x1Local);
        v1Buf.EnQue<float>(v1Local);
        v2Buf.EnQue<float>(v2Local);
        glu_outLocal = glu_outBuf.DeQue<float>();
        x1Local = x1Buf.DeQue<float>();
        v1Local = v1Buf.DeQue<float>();
        v2Local = v2Buf.DeQue<float>();
        LocalTensor<float> jvp_outLocal = jvp_outBuf.AllocTensor<float>();
        Div(jvp_outLocal, glu_outLocal, x1Local, iterSize);
        Mul(x1Local, glu_outLocal, jvp_outLocal, iterSize);
        Sub(x1Local, glu_outLocal, x1Local, iterSize);
        Mul(jvp_outLocal, jvp_outLocal, v1Local, iterSize);
        MulAddDst(jvp_outLocal, x1Local, v2Local, iterSize);
        glu_outBuf.FreeTensor<float>(glu_outLocal);
        v1Buf.FreeTensor<float>(v1Local);
        x1Buf.FreeTensor<float>(x1Local);
        v2Buf.FreeTensor<float>(v2Local);

        jvp_outBuf.EnQue<float>(jvp_outLocal);
        jvp_outLocal = jvp_outBuf.DeQue<float>();
        DataCopyExtParams storeParams{1, (uint32_t)(iterSize * sizeof(float)), 0, 0, 0};
        DataCopyPad(jvp_outGm[outIndex], jvp_outLocal, storeParams);
        jvp_outBuf.FreeTensor<float>(jvp_outLocal);
    }
    __aicore__ inline void Process() {
        uint32_t blockBeginIndex = 0;
        for (uint32_t i = 0; i < bigLoopTimes; i++) {
            uint32_t iterInputIndex = 0;
            for (uint32_t j = 0; j < smallLoopTimes; j++) {
                uint32_t iterSize_fp32 = min(n_elements_per_iter, hi_block_size - iterInputIndex);
                Process_iter(blockBeginIndex * 2 + iterInputIndex, blockBeginIndex + iterInputIndex, iterSize_fp32);
                iterInputIndex += n_elements_per_iter;
            }
            blockBeginIndex += hi_block_size;
        }
    }

private:
    TPipe *pipe;
    GlobalTensor<float> glu_outGm;
    GlobalTensor<float> jvp_outGm;
    TQue<QuePosition::VECIN, BufferNum> glu_outBuf;
    TQue<QuePosition::VECIN, BufferNum> x1Buf;
    GlobalTensor<float> xGm;
    GlobalTensor<float> vGm;
    TQue<QuePosition::VECIN, BufferNum> v1Buf;
    uint32_t size;
    uint32_t smallLoopTimes;
    uint32_t hi_block_size;
    uint32_t hi_block_num;
    uint32_t bigLoopTimes;
    uint32_t n_elements_per_iter;
    uint32_t n_hi_blocks_per_iter;
    TQue<QuePosition::VECIN, BufferNum> v2Buf;
    TQue<QuePosition::VECOUT, BufferNum> jvp_outBuf;
    int KS;
    int J;
    int HI;
};
#endif // GLU_JVP_FP32_H