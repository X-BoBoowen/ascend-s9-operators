
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CopysignTilingData)
TILING_DATA_FIELD_DEF(int32_t, former_length);
TILING_DATA_FIELD_DEF(int32_t, tail_length);
TILING_DATA_FIELD_DEF(int32_t, former_num);
TILING_DATA_FIELD_DEF(int32_t, tile_length);

TILING_DATA_FIELD_DEF(int32_t, vector_length);
TILING_DATA_FIELD_DEF(int32_t, align_vector_length);

TILING_DATA_FIELD_DEF(uint8_t, board_cast);
TILING_DATA_FIELD_DEF(int32_t, out_dim1);
TILING_DATA_FIELD_DEF(int32_t, out_dim2);
TILING_DATA_FIELD_DEF(int32_t, out_dim3);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Copysign, CopysignTilingData)
} // namespace optiling
