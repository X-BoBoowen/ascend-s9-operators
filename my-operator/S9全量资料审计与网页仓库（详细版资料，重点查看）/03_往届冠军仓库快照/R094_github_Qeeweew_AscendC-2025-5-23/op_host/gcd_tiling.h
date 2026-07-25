
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GcdTilingData)
  TILING_DATA_FIELD_DEF(int, N0);
  TILING_DATA_FIELD_DEF(int, N1);
  TILING_DATA_FIELD_DEF(int, N2);
  TILING_DATA_FIELD_DEF(int, N3);
  TILING_DATA_FIELD_DEF(int, N4);
  TILING_DATA_FIELD_DEF(int, broadcast_mask);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Gcd, GcdTilingData)
}
