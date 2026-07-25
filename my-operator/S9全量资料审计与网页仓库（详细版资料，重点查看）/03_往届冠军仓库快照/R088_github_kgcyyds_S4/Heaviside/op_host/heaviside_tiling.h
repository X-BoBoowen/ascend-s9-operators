
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HeavisideTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreProcessNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreProcessNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, valuesNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Heaviside, HeavisideTilingData)
}
