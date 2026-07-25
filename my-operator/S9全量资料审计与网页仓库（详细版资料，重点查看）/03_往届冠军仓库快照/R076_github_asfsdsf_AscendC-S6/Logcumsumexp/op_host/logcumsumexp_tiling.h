
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LogcumsumexpTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, interval);
  TILING_DATA_FIELD_DEF(int32_t, dim);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 10, input_ndarray);
  TILING_DATA_FIELD_DEF(int32_t, input_dimensional);
  TILING_DATA_FIELD_DEF(int32_t, tileDataMaxNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Logcumsumexp, LogcumsumexpTilingData)
}
