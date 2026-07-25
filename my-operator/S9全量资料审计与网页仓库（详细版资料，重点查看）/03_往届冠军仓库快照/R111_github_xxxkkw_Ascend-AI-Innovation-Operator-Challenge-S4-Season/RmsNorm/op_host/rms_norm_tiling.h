
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RmsNormTilingData)
TILING_DATA_FIELD_DEF(uint64_t,rstdGmLength);
TILING_DATA_FIELD_DEF(uint64_t, rstdLength);
TILING_DATA_FIELD_DEF(uint64_t, totalLength);
TILING_DATA_FIELD_DEF(uint64_t, tileLoop);
TILING_DATA_FIELD_DEF(uint64_t, maxPerTime);
TILING_DATA_FIELD_DEF(uint64_t, loopCount);
TILING_DATA_FIELD_DEF(uint64_t, leftNum);
TILING_DATA_FIELD_DEF(uint64_t, leftPerTime);
TILING_DATA_FIELD_DEF(uint64_t, tileLength);
TILING_DATA_FIELD_DEF(uint64_t,tileNum);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, factor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmsNorm, RmsNormTilingData)
}
