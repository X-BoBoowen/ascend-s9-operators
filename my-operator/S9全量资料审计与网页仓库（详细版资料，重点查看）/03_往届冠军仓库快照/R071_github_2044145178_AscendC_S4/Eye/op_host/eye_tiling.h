
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EyeTilingData)
    TILING_DATA_FIELD_DEF(uint16_t, totalMatrixNum);
    TILING_DATA_FIELD_DEF(uint16_t, numRows);
    TILING_DATA_FIELD_DEF(uint16_t, numColumns);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Eye, EyeTilingData)



BEGIN_TILING_DATA_DEF(EyeTilingData_slice)
    TILING_DATA_FIELD_DEF(uint64_t, mask0);
    TILING_DATA_FIELD_DEF(uint64_t, mask1);
    TILING_DATA_FIELD_DEF(uint64_t, mask_remain0);
    TILING_DATA_FIELD_DEF(uint64_t, mask_remain1);
    TILING_DATA_FIELD_DEF(int, totalMatrixNum);
    TILING_DATA_FIELD_DEF(int, numRows);
    TILING_DATA_FIELD_DEF(int, numColumns);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Eye_2, EyeTilingData_slice)

}
