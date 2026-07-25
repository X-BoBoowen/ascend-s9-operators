
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CopysignTilingData)
TILING_DATA_FIELD_DEF(uint32_t, smallSize);
TILING_DATA_FIELD_DEF(uint32_t, incSize);
TILING_DATA_FIELD_DEF(uint32_t, totalSize);
TILING_DATA_FIELD_DEF(uint16_t, formerNum);

TILING_DATA_FIELD_DEF_ARR(uint16_t, 8, mmInputDims);
TILING_DATA_FIELD_DEF_ARR(uint16_t, 8, mmOtherDims);
TILING_DATA_FIELD_DEF_ARR(uint16_t, 8, mmOutputDims);
TILING_DATA_FIELD_DEF(uint8_t, nOutputDims);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Copysign, CopysignTilingData)

} // namespace optiling
