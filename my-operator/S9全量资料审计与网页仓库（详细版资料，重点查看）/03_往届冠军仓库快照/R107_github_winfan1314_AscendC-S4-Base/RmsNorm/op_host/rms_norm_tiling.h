#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RmsNormTilingData)
  TILING_DATA_FIELD_DEF(float, epsilon);
  TILING_DATA_FIELD_DEF(uint32_t, x1TotalLength);
  TILING_DATA_FIELD_DEF(uint32_t, x2TotalLength);
  TILING_DATA_FIELD_DEF(uint32_t, batchNum);
  TILING_DATA_FIELD_DEF(uint32_t, batchLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, lastTileLength);
  TILING_DATA_FIELD_DEF(uint32_t, rstdTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, rstdTileLength);
  TILING_DATA_FIELD_DEF(uint32_t, rstdLastTileLength);
  TILING_DATA_FIELD_DEF(uint32_t, resLength);
  TILING_DATA_FIELD_DEF(uint32_t, mulLastTileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmsNorm, RmsNormTilingData)
}
