
#ifndef FRESNELCOS_TILING_H
#define FRESNELCOS_TILING_H
#include "register/tilingdata_base.h"
// #include "foreach_cos_tilling."
namespace optiling {
BEGIN_TILING_DATA_DEF(FresnelCosTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, tiling_size);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
  TILING_DATA_FIELD_DEF(uint32_t, aivNum);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FresnelCos, FresnelCosTilingData)
}



#endif // FRESNELCOS_TILING_H