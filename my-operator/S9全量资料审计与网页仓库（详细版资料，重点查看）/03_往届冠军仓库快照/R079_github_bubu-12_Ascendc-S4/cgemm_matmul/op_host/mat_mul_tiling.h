
#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, if_bias);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, x_shape_info);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, y_shape_info);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, bias_shape_info);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, z_shape_info);
    END_TILING_DATA_DEF;
    
    REGISTER_TILING_DATA_CLASS(MatMul, TilingData)
    } // namespace optiling
