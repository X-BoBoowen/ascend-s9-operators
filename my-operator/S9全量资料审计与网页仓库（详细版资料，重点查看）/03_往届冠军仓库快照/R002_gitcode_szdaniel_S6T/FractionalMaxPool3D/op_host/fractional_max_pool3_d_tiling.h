
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FractionalMaxPool3DTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, formerNum);
  TILING_DATA_FIELD_DEF(uint32_t, formerLength);
  TILING_DATA_FIELD_DEF(uint32_t, tailNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailLength);

  TILING_DATA_FIELD_DEF(uint32_t, tStep);
  TILING_DATA_FIELD_DEF(uint32_t, hStep);
  TILING_DATA_FIELD_DEF(uint32_t, wStep);

  TILING_DATA_FIELD_DEF(uint32_t, poolSizeT);
  TILING_DATA_FIELD_DEF(uint32_t, poolSizeH);
  TILING_DATA_FIELD_DEF(uint32_t, poolSizeW);

  TILING_DATA_FIELD_DEF(uint32_t, outputT);
  TILING_DATA_FIELD_DEF(uint32_t, outputH);
  TILING_DATA_FIELD_DEF(uint32_t, outputW);

  TILING_DATA_FIELD_DEF(uint32_t, inputT);
  TILING_DATA_FIELD_DEF(uint32_t, inputH);
  TILING_DATA_FIELD_DEF(uint32_t, inputW);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FractionalMaxPool3D, FractionalMaxPool3DTilingData)
}
