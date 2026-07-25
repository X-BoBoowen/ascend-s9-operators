
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EyeTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, row);
  TILING_DATA_FIELD_DEF(uint32_t, col);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreProcessNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreProcessNum);
  TILING_DATA_FIELD_DEF(uint32_t, dtype);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Eye, EyeTilingData)
}
