#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IndexAddFastTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerBlock);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexAddFast, IndexAddFastTilingData)
}
