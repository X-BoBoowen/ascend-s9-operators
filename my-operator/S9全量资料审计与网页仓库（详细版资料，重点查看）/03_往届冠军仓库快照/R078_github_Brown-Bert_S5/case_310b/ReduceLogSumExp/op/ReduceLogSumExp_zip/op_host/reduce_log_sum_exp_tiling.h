
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReduceLogSumExpTilingData)
TILING_DATA_FIELD_DEF_ARR(uint32_t, 4, inputShape);       // 输入数据维度信息
TILING_DATA_FIELD_DEF(uint32_t, dataType);                // 运行时数据类型
TILING_DATA_FIELD_DEF(uint32_t, dimension);               // 维度信息
TILING_DATA_FIELD_DEF(uint32_t, dataSize);                // 输入元素个数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReduceLogSumExp, ReduceLogSumExpTilingData)
}
