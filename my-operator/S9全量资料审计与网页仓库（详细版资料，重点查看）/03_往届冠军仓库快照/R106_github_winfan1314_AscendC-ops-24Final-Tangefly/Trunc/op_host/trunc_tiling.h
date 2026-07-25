
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TruncTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, placeHolder);
END_TILING_DATA_DEF;
BEGIN_TILING_DATA_DEF(TruncTIBasic)
        TILING_DATA_FIELD_DEF(uint32_t, tileLength)
        TILING_DATA_FIELD_DEF(uint32_t, tileNumber)
        TILING_DATA_FIELD_DEF(uint32_t, reminder)
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(TruncTIBasic2)
        TILING_DATA_FIELD_DEF(uint32_t, tileLength)
        TILING_DATA_FIELD_DEF(uint32_t, tileNumber)
        TILING_DATA_FIELD_DEF(uint32_t, reminder)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Trunc, TruncTilingData)
REGISTER_TILING_DATA_CLASS(Trunc_1, TruncTIBasic)
REGISTER_TILING_DATA_CLASS(Trunc_2, TruncTIBasic2)
}
