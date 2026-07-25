#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReduceLogSumExpTilingData)
TILING_DATA_FIELD_DEF_ARR(uint32_t, 4, inputShape);
TILING_DATA_FIELD_DEF(uint32_t, dataType);
TILING_DATA_FIELD_DEF(uint32_t, dimension);
TILING_DATA_FIELD_DEF(uint32_t, dataSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReduceLogSumExp, ReduceLogSumExpTilingData)
}  // namespace optiling
