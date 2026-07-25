
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LogcumsumexpTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, ADimLength);
  TILING_DATA_FIELD_DEF(uint32_t, PDimLength);

  // former block
  TILING_DATA_FIELD_DEF(uint32_t, formerNum);
  TILING_DATA_FIELD_DEF(uint32_t, formerLength);
  TILING_DATA_FIELD_DEF(uint32_t, formerTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, formerTileLength);
  TILING_DATA_FIELD_DEF(uint32_t, formerLastTileLength);

  // tail block
  TILING_DATA_FIELD_DEF(uint32_t, tailNum); 
  TILING_DATA_FIELD_DEF(uint32_t, tailLength);
  TILING_DATA_FIELD_DEF(uint32_t, tailTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailTileLength);
  TILING_DATA_FIELD_DEF(uint32_t, tailLastTileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Logcumsumexp, LogcumsumexpTilingData)
}
