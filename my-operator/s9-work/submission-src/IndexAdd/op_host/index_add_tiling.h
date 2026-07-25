#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IndexAddTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, outputRows);
    TILING_DATA_FIELD_DEF(uint32_t, indexCount);
    TILING_DATA_FIELD_DEF(uint32_t, rowWidth);
    TILING_DATA_FIELD_DEF(uint32_t, baseRowsPerBlock);
    TILING_DATA_FIELD_DEF(uint32_t, extraBlocks);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexAdd, IndexAddTilingData)
}
