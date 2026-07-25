
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterReduceTilingData)
TILING_DATA_FIELD_DEF_ARR(uint16_t, 8, mmInputDims);
TILING_DATA_FIELD_DEF(uint32_t, inputSize);
TILING_DATA_FIELD_DEF(uint8_t, dim);
TILING_DATA_FIELD_DEF(uint8_t, nInputDims);
TILING_DATA_FIELD_DEF(uint8_t, type);
TILING_DATA_FIELD_DEF(int8_t, includeSelf);


TILING_DATA_FIELD_DEF(uint32_t, totalTilingNum);
TILING_DATA_FIELD_DEF(uint32_t, KS);
TILING_DATA_FIELD_DEF(uint32_t, J);
TILING_DATA_FIELD_DEF(uint32_t, lineSize);





END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterReduce, ScatterReduceTilingData)
}
