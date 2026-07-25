
#include "register/tilingdata_base.h"

#define MAX_DIM_NUMBER 4
namespace optiling {
BEGIN_TILING_DATA_DEF(CopysignTilingData)
  TILING_DATA_FIELD_DEF(int32_t, size);
  TILING_DATA_FIELD_DEF(int32_t, length);
  TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_DIM_NUMBER, shape);
  TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_DIM_NUMBER, n1);
  TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_DIM_NUMBER, n2);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Copysign, CopysignTilingData)
}
