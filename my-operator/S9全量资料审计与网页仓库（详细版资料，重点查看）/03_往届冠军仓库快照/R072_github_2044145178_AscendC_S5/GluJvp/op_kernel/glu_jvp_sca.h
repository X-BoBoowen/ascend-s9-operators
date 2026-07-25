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
 * @file glu_jvp_sca.h
 */
#ifndef GLU_JVP_SCA_H
#define GLU_JVP_SCA_H
#include <type_traits>
#include "kernel_operator.h"

using namespace AscendC;
class KernelGluJvpSca {
public:
    __aicore__ inline KernelGluJvpSca() {}
    __aicore__ inline void Init(GM_ADDR glu_out, GM_ADDR input, GM_ADDR v, GM_ADDR jvp_out, int HI, int J, int KS, TPipe *pipeIn) {
        this->totalSize = HI * J * KS;
        glu_outGm.SetGlobalBuffer((__gm__ DTYPE_INPUT *)glu_out, totalSize / 2);
        inputGm.SetGlobalBuffer((__gm__ DTYPE_INPUT *)input, totalSize);
        vGm.SetGlobalBuffer((__gm__ DTYPE_INPUT *)v, totalSize);
        jvp_outGm.SetGlobalBuffer((__gm__ DTYPE_INPUT *)jvp_out, totalSize / 2);
        pipe = pipeIn;
        pipe->InitBuffer(inBuf, 512);
        pipe->InitBuffer(outBuf, 512);
        this->HI = HI;
        this->J = J;
        this->KS = KS;
    }
    __aicore__ inline void Process() {
        const int total_iterations = HI * (J / 2) * KS;
        for (int index = 0; index < total_iterations; index++) {
            int ks = index % KS;
            int temp = index / KS;
            int j = temp % (J / 2) + J / 2;
            int hi = temp / (J / 2);
            int input1_idx = hi * J * KS + (j - J / 2) * KS + ks;
            int input2_idx = hi * J * KS + j * KS + ks;
            if constexpr (std::is_same<DTYPE_INPUT, bfloat16_t>::value) {
                float x1_val = ToFloat(inputGm.GetValue(input1_idx));
                float x2_val = ToFloat(inputGm.GetValue(input2_idx));
                float v1_val = ToFloat(vGm.GetValue(input1_idx));
                float v2_val = ToFloat(vGm.GetValue(input2_idx));
                LocalTensor<float> inLocal = inBuf.Get<float>();
                inLocal.SetValue(0, x2_val);
                LocalTensor<float> outLocal = outBuf.Get<float>();
                Sigmoid(outLocal, inLocal, 32);
                float sigmoid_b = outLocal.GetValue(0);
                float glu_out_val = x1_val * sigmoid_b;
                float out = sigmoid_b * v1_val + glu_out_val * (v2_val - sigmoid_b * v2_val);
                LocalTensor<bfloat16_t> outLocal_bf16 = outBuf.Get<bfloat16_t>();
                inLocal.SetValue(0, out);
                Cast(outLocal_bf16, inLocal, RoundMode::CAST_RINT, 32);
                bfloat16_t out_bf16 = outLocal_bf16.GetValue(0);
                jvp_outGm.SetValue(index, out_bf16);
            } else {
                float x1_val = inputGm.GetValue(input1_idx);
                float x2_val = inputGm.GetValue(input2_idx);
                float v1_val = vGm.GetValue(input1_idx);
                float v2_val = vGm.GetValue(input2_idx);
                LocalTensor<float> inLocal = inBuf.Get<float>();
                inLocal.SetValue(0, x2_val);
                LocalTensor<float> outLocal = outBuf.Get<float>();
                Sigmoid(outLocal, inLocal, 32);
                float sigmoid_b = outLocal.GetValue(0);
                float glu_out_val = x1_val * sigmoid_b;
                float out = sigmoid_b * v1_val + glu_out_val * (v2_val - sigmoid_b * v2_val);
                jvp_outGm.SetValue(index, out);
            }
        }
    }

private:
    TPipe *pipe;
    GlobalTensor<DTYPE_INPUT> glu_outGm;
    GlobalTensor<DTYPE_INPUT> vGm;
    GlobalTensor<DTYPE_INPUT> jvp_outGm;
    TBuf<QuePosition::VECCALC> inBuf;
    GlobalTensor<DTYPE_INPUT> inputGm;
    TBuf<QuePosition::VECCALC> outBuf;
    int KS;
    int J;
    int HI;
    int totalSize;
};
#endif // GLU_JVP_SCA_H