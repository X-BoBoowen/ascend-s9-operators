#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(GatherTilingData)
        TILING_DATA_FIELD_DEF(uint32_t, placeHolder);
    END_TILING_DATA_DEF;

    BEGIN_TILING_DATA_DEF(GatherTilingDataWithDataCopy)
        TILING_DATA_FIELD_DEF(uint32_t, batchNumber);
        TILING_DATA_FIELD_DEF(uint32_t, batchLength);
        TILING_DATA_FIELD_DEF(uint32_t, indicesLength);
        TILING_DATA_FIELD_DEF(uint32_t, sliceLength);
        TILING_DATA_FIELD_DEF(uint32_t, maxLength);
        TILING_DATA_FIELD_DEF(uint32_t, tileLength);
        TILING_DATA_FIELD_DEF(uint32_t, reminder);
    END_TILING_DATA_DEF;

    BEGIN_TILING_DATA_DEF(GatherTilingDataScalarCopy)
        TILING_DATA_FIELD_DEF(uint32_t, batchNumber);
        TILING_DATA_FIELD_DEF(uint32_t, batchLength);
        TILING_DATA_FIELD_DEF(uint32_t, indicesLength);
        TILING_DATA_FIELD_DEF(uint32_t, sliceLength);
    END_TILING_DATA_DEF;

    BEGIN_TILING_DATA_DEF(GatherTilingDataWithMiniBatch)
        TILING_DATA_FIELD_DEF(uint32_t, batchNumber);
        TILING_DATA_FIELD_DEF(uint32_t, batchLength);
        TILING_DATA_FIELD_DEF(uint32_t, indicesLength);
    END_TILING_DATA_DEF;

    BEGIN_TILING_DATA_DEF(GatherTilingDataWithKernelGather)
        TILING_DATA_FIELD_DEF(uint32_t, batchNumber);
        TILING_DATA_FIELD_DEF(uint32_t, batchLength);
        TILING_DATA_FIELD_DEF(uint32_t, indicesLength);
        TILING_DATA_FIELD_DEF(uint32_t, maxLength);
        TILING_DATA_FIELD_DEF(uint32_t, tileLength);
        TILING_DATA_FIELD_DEF(uint32_t, reminder);
    END_TILING_DATA_DEF;
    
    REGISTER_TILING_DATA_CLASS(Gather, GatherTilingData)
    REGISTER_TILING_DATA_CLASS(Gather_0, GatherTilingDataWithDataCopy)
    REGISTER_TILING_DATA_CLASS(Gather_1, GatherTilingDataScalarCopy)
    REGISTER_TILING_DATA_CLASS(Gather_2, GatherTilingDataWithMiniBatch)
    REGISTER_TILING_DATA_CLASS(Gather_3, GatherTilingDataWithKernelGather)
}
  