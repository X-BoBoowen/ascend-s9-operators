#ifndef CONCAT_TILING_H
#define CONCAT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
// Per-input dim extents are stored instead of row bytes: an extent always
// fits in uint32 for any legal shape, so 2048 inputs cost 8 KB of tiling
// while the kernel recovers row bytes as extent * innerBytes in uint64.
BEGIN_TILING_DATA_DEF(ConcatTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, outer);
    TILING_DATA_FIELD_DEF(uint64_t, outRowBytes);
    TILING_DATA_FIELD_DEF(uint64_t, innerBytes);
    TILING_DATA_FIELD_DEF(uint64_t, totalBytes);
    TILING_DATA_FIELD_DEF(uint64_t, baseRowsPerBlock);
    TILING_DATA_FIELD_DEF(uint64_t, baseWorkBlocks);
    TILING_DATA_FIELD_DEF(uint32_t, inputCount);
    TILING_DATA_FIELD_DEF(uint32_t, mode);
    TILING_DATA_FIELD_DEF(uint32_t, elementBytes);
    TILING_DATA_FIELD_DEF(uint32_t, rank);
    TILING_DATA_FIELD_DEF(uint32_t, dim);
    TILING_DATA_FIELD_DEF(uint32_t, extraBlocks);
    TILING_DATA_FIELD_DEF(uint32_t, extraWorkBlocks);
    TILING_DATA_FIELD_DEF(uint32_t, workUnitBytes);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 2048, dimExtents);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Concat, ConcatTilingData)
}

#endif
