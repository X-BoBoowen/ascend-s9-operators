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
 * @file glu_jvp.cpp
 */
#include <type_traits>
#include "kernel_operator.h"
#include "glu_jvp_sca.h"
#include "glu_jvp_fp32.h"
#include "glu_jvp_bf16.h"
#include "glu_jvp_fp32_spec.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void glu_jvp(GM_ADDR glu_out, GM_ADDR input, GM_ADDR v, GM_ADDR jvp_out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        KernelGluJvpFp32 op;
        op.Init(glu_out, input, v, jvp_out, tiling_data.HI, tiling_data.J, tiling_data.KS, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum,
                &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelGluJvpBf16 op;
        op.Init(glu_out, input, v, jvp_out, tiling_data.HI, tiling_data.J, tiling_data.KS, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum,
                &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelGluJvpSca op;
        op.Init(glu_out, input, v, jvp_out, tiling_data.HI, tiling_data.J, tiling_data.KS, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        KernelGluJvpFp32Spec op;
        op.Init(glu_out, input, v, jvp_out, tiling_data.HI, tiling_data.J, tiling_data.KS, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum,
                &pipe);
        op.Process();
    }
}