#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TransposeTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);
    TILING_DATA_FIELD_DEF(uint32_t, cols);
    TILING_DATA_FIELD_DEF(uint32_t, tileCols);
    TILING_DATA_FIELD_DEF(uint32_t, baseTilesPerBlock);
    TILING_DATA_FIELD_DEF(uint32_t, extraTileBlocks);
    TILING_DATA_FIELD_DEF(uint64_t, totalElements);
    TILING_DATA_FIELD_DEF(uint64_t, baseElementsPerBlock);
    TILING_DATA_FIELD_DEF(uint32_t, extraBlocks);
    TILING_DATA_FIELD_DEF(uint32_t, rank);
    TILING_DATA_FIELD_DEF(uint32_t, identity);
    TILING_DATA_FIELD_DEF(uint32_t, fast2D);
    TILING_DATA_FIELD_DEF(uint32_t, gatherLastDim);
    TILING_DATA_FIELD_DEF(uint32_t, gatherInputStride);
    TILING_DATA_FIELD_DEF(uint32_t, gatherChunksPerRun);
    TILING_DATA_FIELD_DEF(uint64_t, contiguousElements);
    TILING_DATA_FIELD_DEF(uint64_t, baseRunsPerBlock);
    TILING_DATA_FIELD_DEF(uint32_t, extraRunBlocks);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 6, outputDims);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 6, inputStrides);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Transpose, TransposeTilingData)
}
