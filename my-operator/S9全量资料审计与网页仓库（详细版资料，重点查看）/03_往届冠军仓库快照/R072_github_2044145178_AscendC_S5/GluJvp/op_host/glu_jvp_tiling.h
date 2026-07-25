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
 * @file glu_jvp_tiling.h
 */
#ifndef GLU_JVP_TILING_H
#define GLU_JVP_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GluJvpTilingData)
TILING_DATA_FIELD_DEF(int, HI);
TILING_DATA_FIELD_DEF(int, J);
TILING_DATA_FIELD_DEF(int, KS);
TILING_DATA_FIELD_DEF(uint32_t, smallSize);
TILING_DATA_FIELD_DEF(uint16_t, incSize);
TILING_DATA_FIELD_DEF(uint16_t, formerNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GluJvp, GluJvpTilingData)
} // namespace optiling
#endif
