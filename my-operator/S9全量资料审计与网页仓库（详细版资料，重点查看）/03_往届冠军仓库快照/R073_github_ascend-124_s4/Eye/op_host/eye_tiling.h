#include "register/tilingdata_base.h"

namespace optiling
{
    BEGIN_TILING_DATA_DEF(EyeTilingData)
        TILING_DATA_FIELD_DEF(int, num_rows);
        TILING_DATA_FIELD_DEF(int, num_columns);
        TILING_DATA_FIELD_DEF(int, batch_size);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(Eye, EyeTilingData)
}
