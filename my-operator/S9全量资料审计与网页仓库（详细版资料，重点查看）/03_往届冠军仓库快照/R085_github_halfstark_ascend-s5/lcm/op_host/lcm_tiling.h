
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LcmTilingData)
  TILING_DATA_FIELD_DEF(uint16_t, nAcores);
  TILING_DATA_FIELD_DEF(uint16_t, nBcores);
  TILING_DATA_FIELD_DEF(uint16_t, maxBlockPerIter);
  TILING_DATA_FIELD_DEF(uint32_t, blockPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, mid);
  TILING_DATA_FIELD_DEF(uint32_t, pre);
  TILING_DATA_FIELD_DEF(uint32_t, last);
  TILING_DATA_FIELD_DEF(uint32_t, smallBatch);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Lcm, LcmTilingData)
}
