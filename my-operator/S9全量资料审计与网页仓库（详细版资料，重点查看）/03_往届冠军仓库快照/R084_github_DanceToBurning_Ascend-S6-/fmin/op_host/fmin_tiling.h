 #include "register/tilingdata_base.h"
 
 namespace optiling {
 BEGIN_TILING_DATA_DEF(TilingData)
 TILING_DATA_FIELD_DEF(uint32_t, dt);
 TILING_DATA_FIELD_DEF(uint32_t, xlength);
 TILING_DATA_FIELD_DEF(uint32_t, vlength);
 TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, input_shape_info);
 TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, values_shape_info);
 TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, output_shape_info);
 END_TILING_DATA_DEF;
 
 REGISTER_TILING_DATA_CLASS(Fmin, TilingData)
 }
 #endif
 