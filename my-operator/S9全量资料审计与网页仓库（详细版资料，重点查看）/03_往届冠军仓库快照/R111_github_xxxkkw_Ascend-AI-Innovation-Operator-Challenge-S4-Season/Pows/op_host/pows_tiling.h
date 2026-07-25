
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PowsTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, totalLength);
  TILING_DATA_FIELD_DEF(uint64_t, loopCount);
  TILING_DATA_FIELD_DEF(uint64_t, leftNum);
  TILING_DATA_FIELD_DEF(uint64_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, y_dimensional); 
  TILING_DATA_FIELD_DEF(uint32_t,x1TotalLength);
  TILING_DATA_FIELD_DEF(uint32_t,x2TotalLength);
  TILING_DATA_FIELD_DEF(uint32_t,x1Size);
  TILING_DATA_FIELD_DEF(uint32_t,x2Size);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, y_ndarray);  
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, x1_ndarray);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, x2_ndarray);  
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, y_sumndarray);  
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, x1_sumndarray);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, x2_sumndarray); 
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Pows, PowsTilingData)
}
