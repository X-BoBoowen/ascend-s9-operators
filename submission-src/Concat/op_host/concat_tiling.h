#ifndef CONCAT_TILING_H
#define CONCAT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConcatTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, outer);
    TILING_DATA_FIELD_DEF(uint64_t, outRowBytes);
    TILING_DATA_FIELD_DEF(uint64_t, innerElements);
    TILING_DATA_FIELD_DEF(uint32_t, inputCount);
    TILING_DATA_FIELD_DEF(uint32_t, rank);
    TILING_DATA_FIELD_DEF(uint32_t, dim);
    TILING_DATA_FIELD_DEF(uint32_t, elementBytes);
    TILING_DATA_FIELD_DEF(uint64_t, baseRowsPerBlock);
    TILING_DATA_FIELD_DEF(uint32_t, extraBlocks);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 256, inputRowBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Concat, ConcatTilingData)
}

#endif
