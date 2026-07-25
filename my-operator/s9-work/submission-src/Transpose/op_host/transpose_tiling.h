#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TransposeTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);
    TILING_DATA_FIELD_DEF(uint32_t, cols);
    TILING_DATA_FIELD_DEF(uint32_t, tileCols);
    TILING_DATA_FIELD_DEF(uint32_t, baseTilesPerBlock);
    TILING_DATA_FIELD_DEF(uint32_t, extraBlocks);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Transpose, TransposeTilingData)
}
