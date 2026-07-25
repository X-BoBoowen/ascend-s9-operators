
#include "common.h"
#include "register/tilingdata_base.h"

namespace optiling {
  BEGIN_TILING_DATA_DEF(CopysignTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, maxDim);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, shapeInf1);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, shapeInf2);
  END_TILING_DATA_DEF;
  REGISTER_TILING_DATA_CLASS(Copysign, CopysignTilingData)
  
  BEGIN_TILING_DATA_DEF(CopysignTilingData_0)
  TILING_DATA_FIELD_DEF(uint32_t, totalNum);
  END_TILING_DATA_DEF;
  REGISTER_TILING_DATA_CLASS(Copysign_0, CopysignTilingData_0)
  
  BEGIN_TILING_DATA_DEF(CopysignTilingData_2)
  TILING_DATA_FIELD_DEF_ARR(uint16_t, MAXDIM_2, shapeInf);
  END_TILING_DATA_DEF;
  REGISTER_TILING_DATA_CLASS(Copysign_2, CopysignTilingData_2)
  
  }  // namespace optiling