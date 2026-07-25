
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GcdTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreLen);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreTileLen);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreTileNum);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreTailTileLen);
    TILING_DATA_FIELD_DEF(uint32_t, smallCoreLen);
    TILING_DATA_FIELD_DEF(uint32_t, smallCoreTileLen);
    TILING_DATA_FIELD_DEF(uint32_t, smallCoreTileNum);
    TILING_DATA_FIELD_DEF(uint32_t, smallCoreTailTileLen);
    TILING_DATA_FIELD_DEF_ARR(int64_t, 10, x1Shape);
    TILING_DATA_FIELD_DEF_ARR(int64_t, 10, x2Shape);
    TILING_DATA_FIELD_DEF_ARR(int64_t, 10, yShape);
    TILING_DATA_FIELD_DEF(int64_t, shapeDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Gcd, GcdTilingData)
}
