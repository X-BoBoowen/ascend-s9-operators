#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SquareSumFastTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, outer);
    TILING_DATA_FIELD_DEF(uint32_t, reduceLen);
    TILING_DATA_FIELD_DEF(uint32_t, paddedReduce);
    TILING_DATA_FIELD_DEF(uint32_t, baseRowsPerBlock);
    TILING_DATA_FIELD_DEF(uint32_t, extraBlocks);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SquareSumFast, SquareSumFastTilingData)
}
