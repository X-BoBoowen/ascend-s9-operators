
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SortTilingData)
TILING_DATA_FIELD_DEF(uint32_t, dataType);                // 运行时数据类型
TILING_DATA_FIELD_DEF(uint32_t, dimension);               // 维度信息
TILING_DATA_FIELD_DEF(uint32_t, dataSize);                // 输入元素个数
TILING_DATA_FIELD_DEF(int32_t, axis);
TILING_DATA_FIELD_DEF(bool, descending);
TILING_DATA_FIELD_DEF(bool, stable);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Sort, SortTilingData)
}
