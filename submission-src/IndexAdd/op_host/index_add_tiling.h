#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IndexAddTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, totalElements);
    TILING_DATA_FIELD_DEF(uint64_t, sourceElements);
    TILING_DATA_FIELD_DEF(uint64_t, outer);
    TILING_DATA_FIELD_DEF(uint64_t, dimSize);
    TILING_DATA_FIELD_DEF(uint64_t, inner);
    TILING_DATA_FIELD_DEF(uint64_t, indexCount);
    TILING_DATA_FIELD_DEF(uint64_t, taskCount);
    TILING_DATA_FIELD_DEF(uint32_t, dimGroup);
    TILING_DATA_FIELD_DEF(uint32_t, dimGroups);
    TILING_DATA_FIELD_DEF(uint32_t, innerChunks);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexAdd, IndexAddTilingData)
}
