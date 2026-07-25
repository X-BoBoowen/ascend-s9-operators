
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HeavisideTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, smallSize);
  TILING_DATA_FIELD_DEF(uint16_t, incSize);
  TILING_DATA_FIELD_DEF(uint16_t, formerNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Heaviside, HeavisideTilingData)



BEGIN_TILING_DATA_DEF(HeavisideTilingData_BroadCast)
  TILING_DATA_FIELD_DEF(uint32_t, size);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, 8, mmInputDims);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, 8, mmValuesDims);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, 8, mmOutputDims);
  TILING_DATA_FIELD_DEF(uint8_t, nOutputDims);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Heaviside_5, HeavisideTilingData_BroadCast)
}
