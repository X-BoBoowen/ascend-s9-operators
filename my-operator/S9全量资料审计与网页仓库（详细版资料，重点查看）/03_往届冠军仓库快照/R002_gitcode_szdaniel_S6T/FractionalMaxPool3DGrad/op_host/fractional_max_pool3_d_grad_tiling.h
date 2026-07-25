
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FractionalMaxPool3DGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, N);
TILING_DATA_FIELD_DEF(uint32_t, C);
TILING_DATA_FIELD_DEF(uint32_t, TI);
TILING_DATA_FIELD_DEF(uint32_t, HI);
TILING_DATA_FIELD_DEF(uint32_t, WI);
TILING_DATA_FIELD_DEF(uint32_t, TO);
TILING_DATA_FIELD_DEF(uint32_t, HO);
TILING_DATA_FIELD_DEF(uint32_t, WO);

TILING_DATA_FIELD_DEF(uint32_t, formerNum);
TILING_DATA_FIELD_DEF(uint32_t, tailNum);
TILING_DATA_FIELD_DEF(uint32_t, tailLength);
TILING_DATA_FIELD_DEF(uint32_t, formerLength);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FractionalMaxPool3DGrad, FractionalMaxPool3DGradTilingData)
} // namespace optiling
