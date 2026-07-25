
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FractionalMaxPool3DGradTilingData)
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, inputSize);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, outputSize);
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, c);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreProcessNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreProcessNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FractionalMaxPool3DGrad, FractionalMaxPool3DGradTilingData)
}
