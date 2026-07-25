#include "register/tilingdata_base.h"

namespace optiling {
    // 基础结构体（框架需要）
    BEGIN_TILING_DATA_DEF(GatherV3TilingData)
        TILING_DATA_FIELD_DEF(uint32_t, placeHolder);
    END_TILING_DATA_DEF;

    // 大数据块拷贝版本
    BEGIN_TILING_DATA_DEF(GatherTilingDataWithDataCopy)
        TILING_DATA_FIELD_DEF(uint32_t, batchNumber);
        // TILING_DATA_FIELD_DEF(uint32_t, innDimLength);
        TILING_DATA_FIELD_DEF(uint32_t, axisDim);
        TILING_DATA_FIELD_DEF(uint32_t, outerDimLength);
        TILING_DATA_FIELD_DEF(uint32_t, indicesLength);
        TILING_DATA_FIELD_DEF(uint32_t, maxLength);
        TILING_DATA_FIELD_DEF(uint32_t, tileLength);
        TILING_DATA_FIELD_DEF(uint32_t, reminder);
        TILING_DATA_FIELD_DEF(uint32_t, batchLength);
        TILING_DATA_FIELD_DEF(uint32_t, innerDimLength);
        // TILING_DATA_FIELD_DEF(uint32_t, batchLength1);
    END_TILING_DATA_DEF;

    // 标量拷贝版本
    BEGIN_TILING_DATA_DEF(GatherTilingDataScalarCopy)
        TILING_DATA_FIELD_DEF(uint32_t, batchNumber);
        // TILING_DATA_FIELD_DEF(uint32_t, innDimLength);
        TILING_DATA_FIELD_DEF(uint32_t, axisDim);
        TILING_DATA_FIELD_DEF(uint32_t, outerDimLength);
        TILING_DATA_FIELD_DEF(uint32_t, indicesLength);
        TILING_DATA_FIELD_DEF(uint32_t, batchLength);
        TILING_DATA_FIELD_DEF(uint32_t, innerDimLength);
        // TILING_DATA_FIELD_DEF(uint32_t, batchLength1);
    END_TILING_DATA_DEF;
    
    // 注册各种tiling数据类型
    REGISTER_TILING_DATA_CLASS(GatherV3, GatherV3TilingData)
    REGISTER_TILING_DATA_CLASS(GatherV3_0, GatherTilingDataWithDataCopy)  // 数据块拷贝
    REGISTER_TILING_DATA_CLASS(GatherV3_1, GatherTilingDataScalarCopy)      // 标量拷贝
}

