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
 * @file glu_jvp_fp32_spec.h
 */
#ifndef GLU_JVP_FP32_SPEC_H
#define GLU_JVP_FP32_SPEC_H
#include <type_traits>
#include "kernel_operator.h"

using namespace AscendC;
constexpr uint16_t BufferNum_SPEC = 1;
class KernelGluJvpFp32Spec {
public:
    __aicore__ inline KernelGluJvpFp32Spec() {}
    __aicore__ inline void Init(GM_ADDR glu_out, GM_ADDR input, GM_ADDR v, GM_ADDR jvp_out, int HI, int J, int KS, uint32_t smallSize, uint32_t incSize, uint16_t formerNum, TPipe *pipeIn) {
        int totalSize = HI * J * KS;
        this->hi_block_size = J * KS / 2;
        uint32_t beginIndex = 0;
        this->pipe = pipeIn;
        this->HI = HI;
        this->J = J;
        this->KS = KS;
        if (GetBlockIdx() < formerNum) {
            this->size = smallSize + incSize;
            beginIndex = this->size * GetBlockIdx();
        } else {
            this->size = smallSize;
            beginIndex = this->size * GetBlockIdx() + formerNum * incSize;
        }
        this->hi_block_num = this->size / this->hi_block_size;
        glu_outGm.SetGlobalBuffer((__gm__ float *)glu_out + beginIndex, size);
        xGm.SetGlobalBuffer((__gm__ float *)input + 2 * beginIndex, 2 * size);
        vGm.SetGlobalBuffer((__gm__ float *)v + 2 * beginIndex, 2 * size);
        jvp_outGm.SetGlobalBuffer((__gm__ float *)jvp_out + beginIndex, size);
        uint32_t spaceSize = hi_block_size * sizeof(float);
        this->n_elements_per_iter = hi_block_size;
        this->bigLoopTimes = hi_block_num;
        pipe->InitBuffer(jvp_outBuf, BufferNum_SPEC, spaceSize + 32);
        pipe->InitBuffer(xQue, 2, spaceSize * 2 + 32);
        pipe->InitBuffer(vQue, BufferNum_SPEC, spaceSize * 2 + 32);
        pipe->InitBuffer(x2Buf, spaceSize + 32);
        pipe->InitBuffer(indexBuf, spaceSize + 32);
        pipe->InitBuffer(v2Buf, spaceSize + 32);
        this->indexLocal = indexBuf.AllocTensor<int32_t>();
        CreateVecIndex(indexLocal, (int32_t)hi_block_size, hi_block_size);
        ShiftLeft(indexLocal, indexLocal, (int32_t)2, hi_block_size);
        int32_t eventIDS_V_MTE2_0 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        SetFlag<AscendC::HardEvent::V_MTE2>(eventIDS_V_MTE2_0);
        WaitFlag<AscendC::HardEvent::V_MTE2>(eventIDS_V_MTE2_0);
    }
    __aicore__ inline void Process_iter(uint32_t inputIndex, uint32_t outIndex, uint32_t iterSize) {
        uint16_t blockCount = 1;
        uint32_t blockLen = iterSize * sizeof(float) * 2;
        DataCopyExtParams copyParams{blockCount, blockLen, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        LocalTensor<float> xLocal = xQue.AllocTensor<float>();
        DataCopyPad(xLocal, xGm[inputIndex], copyParams, padParams);
        LocalTensor<float> vLocal = vQue.AllocTensor<float>();
        DataCopyPad(vLocal, vGm[inputIndex], copyParams, padParams);
        xQue.EnQue<float>(xLocal);
        vQue.EnQue<float>(vLocal);
        xLocal = xQue.DeQue<float>();
        vLocal = vQue.DeQue<float>();
        LocalTensor<float> x1Local = xLocal;
        LocalTensor<float> x2Local = x2Buf.AllocTensor<float>();
        Gather(x2Local, xLocal, indexLocal.ReinterpretCast<uint32_t>(), (uint32_t)0, iterSize);
        LocalTensor<float> v1Local = vLocal;
        LocalTensor<float> v2Local = v2Buf.AllocTensor<float>();
        Gather(v2Local, vLocal, indexLocal.ReinterpretCast<uint32_t>(), (uint32_t)0, iterSize);
        LocalTensor<float> jvp_outLocal = jvp_outBuf.AllocTensor<float>();
        Sigmoid(jvp_outLocal, x2Local, iterSize);
        Mul(x2Local, x1Local, jvp_outLocal, iterSize);
        Mul(v1Local, jvp_outLocal, v1Local, iterSize);
        Mul(jvp_outLocal, jvp_outLocal, v2Local, iterSize);
        Sub(v2Local, v2Local, jvp_outLocal, iterSize);
        Mul(jvp_outLocal, x2Local, v2Local, iterSize);
        Add(jvp_outLocal, v1Local, jvp_outLocal, iterSize);
        xQue.FreeTensor<float>(xLocal);
        vQue.FreeTensor<float>(vLocal);
        jvp_outBuf.EnQue<float>(jvp_outLocal);
        jvp_outLocal = jvp_outBuf.DeQue<float>();
        DataCopyExtParams storeParams{1, (uint32_t)(iterSize * sizeof(float)), 0, 0, 0};
        DataCopyPad(jvp_outGm[outIndex], jvp_outLocal, storeParams);
        jvp_outBuf.FreeTensor<float>(jvp_outLocal);
    }
    __aicore__ inline void Process() {
        uint32_t blockBeginIndex = 0;
        for (uint32_t i = 0; i < bigLoopTimes; i++) {
            Process_iter(blockBeginIndex * 2, blockBeginIndex, hi_block_size);
            blockBeginIndex += hi_block_size;
        }
    }

private:
    TPipe *pipe;
    GlobalTensor<float> glu_outGm;
    TBuf<QuePosition::VECCALC> x2Buf;
    TBuf<QuePosition::VECCALC> v2Buf;
    GlobalTensor<float> xGm;
    GlobalTensor<float> vGm;
    GlobalTensor<float> jvp_outGm;
    TQue<QuePosition::VECIN, BufferNum_SPEC> xQue;
    TQue<QuePosition::VECIN, BufferNum_SPEC> vQue;
    TQue<QuePosition::VECOUT, BufferNum_SPEC> jvp_outBuf;
    TBuf<QuePosition::VECCALC> indexBuf;
    LocalTensor<int32_t> indexLocal;
    int J;
    uint32_t n_elements_per_iter;
    int KS;
    int HI;
    uint32_t size;
    uint32_t smallLoopTimes;
    uint32_t hi_block_size;
    uint32_t bigLoopTimes;
    uint32_t hi_block_num;
    uint32_t n_hi_blocks_per_iter;
};
#endif // GLU_JVP_FP32_SPEC_H