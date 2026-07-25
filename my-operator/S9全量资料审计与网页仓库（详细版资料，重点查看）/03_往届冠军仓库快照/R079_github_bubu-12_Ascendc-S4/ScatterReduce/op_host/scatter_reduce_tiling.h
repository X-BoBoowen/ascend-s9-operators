
#include "register/tilingdata_base.h"

namespace optiling {

  enum ReduceType : uint32_t {
    REDUCE_SUM = 0,
    REDUCE_PROD = 1,
    REDUCE_MEAN = 2,
    REDUCE_AMAX = 3,
    REDUCE_AMIN = 4,
    REDUCE_UNKNOWN = 99 // For error handling
};
const uint32_t SHAPE_INFO_DIM = 8;

BEGIN_TILING_DATA_DEF(ScatterReduceTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, dim);              // The dimension to scatter/reduce along
  TILING_DATA_FIELD_DEF(uint32_t, reduce_code);      // Integer code for the reduction type 
  TILING_DATA_FIELD_DEF(uint32_t, include_self_flag); // 0 for false, 1 for true

  // Data type flag (0 for float32, 1 for float16) 
  TILING_DATA_FIELD_DEF(uint32_t, dtype_flag);

  // Shape information for tensors
  TILING_DATA_FIELD_DEF_ARR(uint32_t, SHAPE_INFO_DIM, self_shape_info);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, SHAPE_INFO_DIM, index_shape_info);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, SHAPE_INFO_DIM, src_shape_info);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, SHAPE_INFO_DIM, y_shape_info); 
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterReduce, ScatterReduceTilingData)
}

