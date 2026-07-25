/**
 * @file heaviside_custom_tiling.h
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
 #ifndef HEAVISIDE_CUSTOM_TILING_H
 #define HEAVISIDE_CUSTOM_TILING_H
 #include "register/tilingdata_base.h"
 
 namespace optiling {
 // 用8: [dim, d0, d1, d2, d3, d4, d5, d6]，dim<=7，可支持7维
 BEGIN_TILING_DATA_DEF(TilingData)
 TILING_DATA_FIELD_DEF(uint32_t, dt);
 TILING_DATA_FIELD_DEF(uint32_t, xlength);
 TILING_DATA_FIELD_DEF(uint32_t, vlength);
 // 下面三个数组分别存 input, values, output shape
 TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, input_shape_info);
 TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, values_shape_info);
 TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, output_shape_info);
 END_TILING_DATA_DEF;
 
 REGISTER_TILING_DATA_CLASS(Heaviside, TilingData)
 } // namespace optiling
 #endif // HEAVISIDE_CUSTOM_TILING_H
 
