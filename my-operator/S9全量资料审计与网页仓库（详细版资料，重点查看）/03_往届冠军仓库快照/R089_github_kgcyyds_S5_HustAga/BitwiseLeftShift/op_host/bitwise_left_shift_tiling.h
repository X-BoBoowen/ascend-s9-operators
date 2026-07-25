
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BitwiseLeftShiftTilingData)
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, n1);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, n2);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreProcessNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreProcessNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BitwiseLeftShift, BitwiseLeftShiftTilingData)
}
