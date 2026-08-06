#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SquareSumV1TilingData)
    TILING_DATA_FIELD_DEF(uint64_t, inputElements);
    TILING_DATA_FIELD_DEF(uint64_t, outputElements);
    TILING_DATA_FIELD_DEF(uint64_t, reduceElements);
    TILING_DATA_FIELD_DEF(uint64_t, innerElements);
    TILING_DATA_FIELD_DEF(uint64_t, trailingReduceElements);
    TILING_DATA_FIELD_DEF(uint64_t, baseOutputsPerBlock);
    TILING_DATA_FIELD_DEF(uint32_t, extraBlocks);
    TILING_DATA_FIELD_DEF(uint32_t, outputRank);
    TILING_DATA_FIELD_DEF(uint32_t, reduceRank);
    TILING_DATA_FIELD_DEF(uint32_t, fastPath);
    TILING_DATA_FIELD_DEF(uint32_t, reduceMode);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 5, outputDims);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 5, outputInputStrides);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 5, reduceDims);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 5, reduceInputStrides);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SquareSumV1, SquareSumV1TilingData)
}
