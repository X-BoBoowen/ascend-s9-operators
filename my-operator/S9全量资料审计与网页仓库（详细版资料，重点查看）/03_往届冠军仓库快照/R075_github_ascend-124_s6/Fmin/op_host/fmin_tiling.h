#include "register/tilingdata_base.h"

namespace optiling
{
    BEGIN_TILING_DATA_DEF(FminTilingData)
        TILING_DATA_FIELD_DEF(int, size);
        TILING_DATA_FIELD_DEF(int, low_size);
        TILING_DATA_FIELD_DEF(int, high_size);
        TILING_DATA_FIELD_DEF(bool, swap_input_other);
        TILING_DATA_FIELD_DEF_ARR(int, 4, input_n);
        TILING_DATA_FIELD_DEF_ARR(int, 4, other_n);
        TILING_DATA_FIELD_DEF_ARR(int, 4, out_n);
        TILING_DATA_FIELD_DEF_ARR(long, 2, broadcast_offset);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(Fmin, FminTilingData)
}
