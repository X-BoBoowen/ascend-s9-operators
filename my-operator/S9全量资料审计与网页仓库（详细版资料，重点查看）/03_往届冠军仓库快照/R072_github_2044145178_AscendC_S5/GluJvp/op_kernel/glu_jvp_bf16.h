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
 * @file glu_jvp_bf16.h
 */
#ifndef GLU_JVP_BF16_H
#define GLU_JVP_BF16_H
#include <type_traits>
#include "kernel_operator.h"
using namespace AscendC;
const uint16_t BufferNum_bf16 = 1;
class KernelGluJvpBf16 {
public:
    __aicore__ inline KernelGluJvpBf16() {}
    __aicore__ inline void Init(GM_ADDR glu_out, GM_ADDR input, GM_ADDR v, GM_ADDR jvp_out, int HI, int J, int KS, uint32_t smallSize, uint32_t incSize, uint16_t formerNum, TPipe *pipeIn) {
        this->pipe = pipeIn;
        this->HI = HI;
        this->J = J;
        this->KS = KS;
        int totalSize = HI * J * KS;
        this->hi_block_size = J * KS / 2;
        uint32_t beginIndex = 0;
        if (GetBlockIdx() < formerNum) {
            this->size = smallSize + incSize;
            beginIndex = this->size * GetBlockIdx();
        } else {
            this->size = smallSize;
            beginIndex = this->size * GetBlockIdx() + formerNum * incSize;
        }
        this->hi_block_num = this->size / this->hi_block_size;
        glu_outGm.SetGlobalBuffer((__gm__ bfloat16_t *)glu_out + beginIndex, size);
        xGm.SetGlobalBuffer((__gm__ bfloat16_t *)input + 2 * beginIndex, 2 * size);
        vGm.SetGlobalBuffer((__gm__ bfloat16_t *)v + 2 * beginIndex, 2 * size);
        jvp_outGm.SetGlobalBuffer((__gm__ bfloat16_t *)jvp_out + beginIndex, size);
        uint32_t spaceSize = min((uint32_t)(hi_block_size * sizeof(float)), (uint32_t)(160 / 5) * 1024 / 2);
        this->n_elements_per_iter = spaceSize / sizeof(float);
        this->bigLoopTimes = hi_block_num;
        this->smallLoopTimes = (hi_block_size + n_elements_per_iter - 1) / n_elements_per_iter;
        pipe->InitBuffer(x1Buf, BufferNum_bf16, spaceSize + 32);
        pipe->InitBuffer(x2Buf, BufferNum_bf16, spaceSize + 32);
        pipe->InitBuffer(v1Buf, BufferNum_bf16, spaceSize + 32);
        pipe->InitBuffer(v2Buf, BufferNum_bf16, spaceSize + 32);
        pipe->InitBuffer(jvp_outBuf, BufferNum_bf16, spaceSize + 32);
        pipe->InitBuffer(inputTmpBuf, BufferNum_bf16, spaceSize / 2 + 32);
    }

    __aicore__ inline void Process_iter(uint32_t inputIndex, uint32_t outIndex, uint32_t iterSize) {
        uint16_t blockCount = 1;
        uint32_t blockLen = iterSize * sizeof(bfloat16_t);
        DataCopyExtParams copyParams{blockCount, blockLen, 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
        LocalTensor<float> x1Local = x1Buf.AllocTensor<float>();
        LocalTensor<float> x2Local = x2Buf.AllocTensor<float>();
        LocalTensor<float> v1Local = v1Buf.AllocTensor<float>();
        LocalTensor<float> v2Local = v2Buf.AllocTensor<float>();
        LocalTensor<bfloat16_t> x1Local_bf16 = inputTmpBuf.AllocTensor<bfloat16_t>();
        DataCopyPad(x1Local_bf16, xGm[inputIndex], copyParams, padParams);
        inputTmpBuf.EnQue<bfloat16_t>(x1Local_bf16);
        x1Local_bf16 = inputTmpBuf.DeQue<bfloat16_t>();
        Cast(x1Local, x1Local_bf16, RoundMode::CAST_NONE, iterSize);
        inputTmpBuf.FreeTensor<bfloat16_t>(x1Local_bf16);
        LocalTensor<bfloat16_t> x2Local_bf16 = inputTmpBuf.AllocTensor<bfloat16_t>();
        DataCopyPad(x2Local_bf16, xGm[inputIndex + hi_block_size], copyParams, padParams);
        inputTmpBuf.EnQue<bfloat16_t>(x2Local_bf16);
        x2Local_bf16 = inputTmpBuf.DeQue<bfloat16_t>();
        Cast(x2Local, x2Local_bf16, RoundMode::CAST_NONE, iterSize);
        inputTmpBuf.FreeTensor<bfloat16_t>(x2Local_bf16);
        LocalTensor<bfloat16_t> v1Local_bf16 = inputTmpBuf.AllocTensor<bfloat16_t>();
        DataCopyPad(v1Local_bf16, vGm[inputIndex], copyParams, padParams);
        inputTmpBuf.EnQue<bfloat16_t>(v1Local_bf16);
        v1Local_bf16 = inputTmpBuf.DeQue<bfloat16_t>();
        Cast(v1Local, v1Local_bf16, RoundMode::CAST_NONE, iterSize);
        inputTmpBuf.FreeTensor<bfloat16_t>(v1Local_bf16);
        LocalTensor<bfloat16_t> v2Local_bf16 = inputTmpBuf.AllocTensor<bfloat16_t>();
        DataCopyPad(v2Local_bf16, vGm[inputIndex + hi_block_size], copyParams, padParams);
        inputTmpBuf.EnQue<bfloat16_t>(v2Local_bf16);
        v2Local_bf16 = inputTmpBuf.DeQue<bfloat16_t>();
        Cast(v2Local, v2Local_bf16, RoundMode::CAST_NONE, iterSize);
        inputTmpBuf.FreeTensor<bfloat16_t>(v2Local_bf16);
        x1Buf.EnQue<float>(x1Local);
        x2Buf.EnQue<float>(x2Local);
        v1Buf.EnQue<float>(v1Local);
        v2Buf.EnQue<float>(v2Local);
        x1Local = x1Buf.DeQue<float>();
        x2Local = x2Buf.DeQue<float>();
        v1Local = v1Buf.DeQue<float>();
        v2Local = v2Buf.DeQue<float>();
        LocalTensor<float> jvp_outLocal = jvp_outBuf.AllocTensor<float>();
        Sigmoid(jvp_outLocal, x2Local, iterSize);
        Mul(x2Local, x1Local, jvp_outLocal, iterSize);
        Mul(v1Local, jvp_outLocal, v1Local, iterSize);
        Mul(jvp_outLocal, jvp_outLocal, v2Local, iterSize);
        Sub(v2Local, v2Local, jvp_outLocal, iterSize);
        Mul(jvp_outLocal, x2Local, v2Local, iterSize);
        Add(jvp_outLocal, v1Local, jvp_outLocal, iterSize);
        x1Buf.FreeTensor<float>(x1Local);
        x2Buf.FreeTensor<float>(x2Local);
        v1Buf.FreeTensor<float>(v1Local);
        v2Buf.FreeTensor<float>(v2Local);
        LocalTensor<bfloat16_t> jvp_outLocal_bf16 = jvp_outLocal.ReinterpretCast<bfloat16_t>();
        Cast(jvp_outLocal_bf16, jvp_outLocal, RoundMode::CAST_RINT, iterSize);
        jvp_outBuf.EnQue<float>(jvp_outLocal);
        jvp_outLocal = jvp_outBuf.DeQue<float>();
        DataCopyExtParams storeParams{1, (uint32_t)(iterSize * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(jvp_outGm[outIndex], jvp_outLocal_bf16, storeParams);
        jvp_outBuf.FreeTensor<float>(jvp_outLocal);
    }
    __aicore__ inline void Process() {
        uint32_t blockBeginIndex = 0;
        for (uint32_t i = 0; i < bigLoopTimes; i++) {
            uint32_t iterInputIndex = 0;
            for (uint32_t j = 0; j < smallLoopTimes; j++) {
                uint32_t iterSize = min(n_elements_per_iter, hi_block_size - iterInputIndex);
                Process_iter(blockBeginIndex * 2 + iterInputIndex, blockBeginIndex + iterInputIndex, iterSize);
                iterInputIndex += n_elements_per_iter;
            }
            blockBeginIndex += hi_block_size;
        }
    }

private:
    TPipe *pipe;
    GlobalTensor<bfloat16_t> glu_outGm;
    GlobalTensor<bfloat16_t> jvp_outGm;
    TQue<QuePosition::VECIN, BufferNum_bf16> inputTmpBuf;
    TQue<QuePosition::VECIN, BufferNum_bf16> x1Buf;
    TQue<QuePosition::VECIN, BufferNum_bf16> x2Buf;
    TQue<QuePosition::VECIN, BufferNum_bf16> v1Buf;
    TQue<QuePosition::VECIN, BufferNum_bf16> v2Buf;
    TQue<QuePosition::VECOUT, BufferNum_bf16> jvp_outBuf;
    GlobalTensor<bfloat16_t> xGm;
    GlobalTensor<bfloat16_t> vGm;
    int KS;
    int J;
    int HI;
    uint32_t size;
    uint32_t smallLoopTimes;
    uint32_t bigLoopTimes;
    uint32_t n_elements_per_iter;
    uint32_t n_hi_blocks_per_iter;
    uint32_t hi_block_size;
    uint32_t hi_block_num;
};
#endif // GLU_JVP_BF16_H
