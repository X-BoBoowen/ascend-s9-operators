
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GcdTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, x1dimNum);
  TILING_DATA_FIELD_DEF(uint32_t, x2dimNum);
  TILING_DATA_FIELD_DEF(uint32_t, x1dimCnt);
  TILING_DATA_FIELD_DEF(uint32_t, x2dimCnt);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, x1Arr);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, x2Arr);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreProcessNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreProcessNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Gcd, GcdTilingData)
}
