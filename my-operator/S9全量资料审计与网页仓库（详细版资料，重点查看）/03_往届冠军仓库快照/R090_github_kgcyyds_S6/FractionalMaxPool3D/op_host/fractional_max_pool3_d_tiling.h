
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FractionalMaxPool3DTilingData)
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, kernelSize);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, inputSize);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, outputSize);
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, c);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);
  TILING_DATA_FIELD_DEF(uint64_t, bigCoreProcessNum);
  TILING_DATA_FIELD_DEF(uint64_t, smallCoreProcessNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FractionalMaxPool3D, FractionalMaxPool3DTilingData)
}
