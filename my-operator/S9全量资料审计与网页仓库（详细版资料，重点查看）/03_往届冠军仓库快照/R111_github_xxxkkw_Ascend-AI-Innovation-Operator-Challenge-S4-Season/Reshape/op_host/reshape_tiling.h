
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReshapeTilingData)
TILING_DATA_FIELD_DEF(uint32_t, tileLength);
TILING_DATA_FIELD_DEF(uint32_t, loopCount);
TILING_DATA_FIELD_DEF(uint32_t, leftNum);
TILING_DATA_FIELD_DEF(uint32_t, shapeLength);
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(int, axis);
TILING_DATA_FIELD_DEF(int, num_axes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Reshape, ReshapeTilingData)
}
