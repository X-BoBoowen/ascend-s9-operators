
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GatherV3TilingData)
TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, inputXShape);       // 输入数据维度信息
// TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, inputIndicesShape);       // 输入数据维度信息
TILING_DATA_FIELD_DEF(uint32_t, dataType);                // 运行时数据类型
TILING_DATA_FIELD_DEF(uint32_t, dimension);               // 维度信息
TILING_DATA_FIELD_DEF(uint32_t, dataSize);                // 输入元素个数
TILING_DATA_FIELD_DEF(uint32_t, batchDims);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GatherV3, GatherV3TilingData)
}
