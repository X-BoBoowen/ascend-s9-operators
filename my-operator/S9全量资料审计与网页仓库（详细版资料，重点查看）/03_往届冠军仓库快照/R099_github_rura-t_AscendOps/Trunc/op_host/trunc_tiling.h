
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TruncTilingData)  
TILING_DATA_FIELD_DEF(uint32_t, block_size); 
TILING_DATA_FIELD_DEF(uint32_t, core_size); 
TILING_DATA_FIELD_DEF(uint32_t, tile_num); 
TILING_DATA_FIELD_DEF(uint32_t, final_length); 
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Trunc, TruncTilingData)
}
