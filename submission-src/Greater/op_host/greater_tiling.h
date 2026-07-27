#ifndef GREATER_TILING_H
#define GREATER_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GreaterTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, outputElements);
    TILING_DATA_FIELD_DEF(uint64_t, baseElementsPerBlock);
    TILING_DATA_FIELD_DEF(uint64_t, partitionUnitElements);
    TILING_DATA_FIELD_DEF(uint32_t, extraBlocks);
    TILING_DATA_FIELD_DEF(uint32_t, rank);
    TILING_DATA_FIELD_DEF(uint32_t, selfContiguous);
    TILING_DATA_FIELD_DEF(uint32_t, otherContiguous);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, outputDims);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, selfStrides);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, otherStrides);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Greater, GreaterTilingData)
}

#endif
