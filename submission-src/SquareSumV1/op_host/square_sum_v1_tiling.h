#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SquareSumV1TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, outer);
    TILING_DATA_FIELD_DEF(uint32_t, reduceLen);
    TILING_DATA_FIELD_DEF(uint32_t, paddedReduce);
    TILING_DATA_FIELD_DEF(uint32_t, baseRowsPerBlock);
    TILING_DATA_FIELD_DEF(uint32_t, extraBlocks);
    TILING_DATA_FIELD_DEF(uint32_t, tileRows);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SquareSumV1, SquareSumV1TilingData)
}
