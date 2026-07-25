
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GluGradJvpTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, coreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, finalTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, iterStep);
  TILING_DATA_FIELD_DEF(uint32_t, stride);
  TILING_DATA_FIELD_DEF(uint32_t, axesDim);
  TILING_DATA_FIELD_DEF(uint32_t, smallBatch);
  TILING_DATA_FIELD_DEF(uint32_t, tail);
  TILING_DATA_FIELD_DEF(uint32_t, padtimes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GluGradJvp, GluGradJvpTilingData)
}
