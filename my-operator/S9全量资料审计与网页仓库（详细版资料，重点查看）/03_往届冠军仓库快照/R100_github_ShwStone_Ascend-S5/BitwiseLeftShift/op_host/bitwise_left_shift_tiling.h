#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BitwiseLeftShiftTilingData)
TILING_DATA_FIELD_DEF(int64_t, former_length);
TILING_DATA_FIELD_DEF(int64_t, tail_length);
TILING_DATA_FIELD_DEF(int64_t, former_num);
TILING_DATA_FIELD_DEF(int64_t, tile_length);

TILING_DATA_FIELD_DEF(uint8_t, board_cast);
TILING_DATA_FIELD_DEF(int64_t, out_dim1);
TILING_DATA_FIELD_DEF(int64_t, out_dim2);
TILING_DATA_FIELD_DEF(int64_t, out_dim3);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BitwiseLeftShift, BitwiseLeftShiftTilingData)
}
