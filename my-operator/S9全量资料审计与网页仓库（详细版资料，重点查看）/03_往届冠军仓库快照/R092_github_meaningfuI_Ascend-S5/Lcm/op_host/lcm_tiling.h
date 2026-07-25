
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LcmTilingData)
  TILING_DATA_FIELD_DEF(int32_t, size);
  TILING_DATA_FIELD_DEF(int32_t, length);
  TILING_DATA_FIELD_DEF(int32_t, tag1);
  TILING_DATA_FIELD_DEF(int32_t, tag2);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 3, n1);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 3, n2);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Lcm, LcmTilingData)
}
