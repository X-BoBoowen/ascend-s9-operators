
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LogcumsumexpTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 10, x_ndarray);
  TILING_DATA_FIELD_DEF(int32_t, x_dimensional);
  TILING_DATA_FIELD_DEF(int32_t, tileDataMaxNum);
  TILING_DATA_FIELD_DEF(int32_t, axis);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Logcumsumexp, LogcumsumexpTilingData)
}
