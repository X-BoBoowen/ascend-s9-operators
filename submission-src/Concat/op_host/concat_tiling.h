#ifndef CONCAT_FAST_TILING_H
#define CONCAT_FAST_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConcatFastTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, outer);
    TILING_DATA_FIELD_DEF(uint32_t, outInner);
    TILING_DATA_FIELD_DEF(uint32_t, inputCount);
    TILING_DATA_FIELD_DEF(uint32_t, baseRowsPerBlock);
    TILING_DATA_FIELD_DEF(uint32_t, extraBlocks);
    TILING_DATA_FIELD_DEF(uint32_t, tileRows);
    TILING_DATA_FIELD_DEF(uint32_t, maxAlignedWidth);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 32, widths);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 32, offsets);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConcatFast, ConcatFastTilingData)
}

#endif
