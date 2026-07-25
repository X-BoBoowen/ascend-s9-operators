
#include "register/tilingdata_base.h"

namespace optiling {
  BEGIN_TILING_DATA_DEF(ReshapeTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, tileLength)
    TILING_DATA_FIELD_DEF(uint32_t, tileNumber)
    TILING_DATA_FIELD_DEF(uint32_t, reminder)
  END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Reshape, ReshapeTilingData)
}
