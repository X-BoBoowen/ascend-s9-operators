
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HypotTilingData)
  // broad cast tile info
  TILING_DATA_FIELD_DEF_ARR(int, 4, xShape);
  TILING_DATA_FIELD_DEF_ARR(int, 4, yShape);
  TILING_DATA_FIELD_DEF_ARR(int, 4, zShape);
  TILING_DATA_FIELD_DEF(int, dimnum);

  // TILING_DATA_FIELD_DEF(int, tileCutAxis);
  // TILING_DATA_FIELD_DEF(int, blockCutAxis);


  // broadcast: cut axis tile info
  // no broadcast: tile info
//   TILING_DATA_FIELD_DEF(int, tileLen);
//   TILING_DATA_FIELD_DEF(int, tileNum);
//   TILING_DATA_FIELD_DEF(int, lastTileLen);

  // no broadcast
//   TILING_DATA_FIELD_DEF(int, blockLength);
//   TILING_DATA_FIELD_DEF(int, tailBlockLength);

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

//   TILING_DATA_FIELD_DEF(int, broadcast_flag);   // 0: no broadcast, 1: broadcast

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Hypot, HypotTilingData)
}
