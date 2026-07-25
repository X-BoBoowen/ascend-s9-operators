
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SelectV2TilingData)
TILING_DATA_FIELD_DEF(uint64_t, smallCoreDataNum);        // 小核处理的总数据量
TILING_DATA_FIELD_DEF(uint64_t, bigCoreDataNum);          // 大核处理的总数据量
TILING_DATA_FIELD_DEF(uint64_t, smallCoreCarryNum);       // 小核搬运数据的次数
TILING_DATA_FIELD_DEF(uint64_t, bigCoreCarryNum);         // 大核搬运数据的次数
TILING_DATA_FIELD_DEF(uint64_t, tileDataNum);             // 单核能处理最大数据量
TILING_DATA_FIELD_DEF(uint64_t, smallCoreFinallDealNum);  // 小核最后一次处理的数据量
TILING_DATA_FIELD_DEF(uint64_t, bigCoreFinallDealNum);    // 大核最后一次处理的数据量
TILING_DATA_FIELD_DEF(uint64_t, bigCoreNum);              // 大核个数
TILING_DATA_FIELD_DEF(uint64_t, dataType);                // 运行时数据类型

TILING_DATA_FIELD_DEF(uint64_t, isBroadCast);                // 标志
TILING_DATA_FIELD_DEF(uint64_t, rows);                // 行数
TILING_DATA_FIELD_DEF(uint64_t, interval);                // 数据间隔
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SelectV2, SelectV2TilingData)
}
