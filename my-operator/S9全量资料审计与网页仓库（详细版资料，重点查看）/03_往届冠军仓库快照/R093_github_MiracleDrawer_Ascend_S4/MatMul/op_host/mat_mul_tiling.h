#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatMulTilingData)
  TILING_DATA_FIELD_DEF(int32_t, bigCoreNumA);
  TILING_DATA_FIELD_DEF(int32_t, bigCoreStrideA);
  TILING_DATA_FIELD_DEF(int32_t, bigCoreNumB);
  TILING_DATA_FIELD_DEF(int32_t, bigCoreStrideB);
  TILING_DATA_FIELD_DEF(int32_t, maxCalcNumStage1);
  TILING_DATA_FIELD_DEF(int32_t, mLength);
  TILING_DATA_FIELD_DEF(int32_t, nLength);
  TILING_DATA_FIELD_DEF(int32_t, kLength);
  TILING_DATA_FIELD_DEF(int32_t, batchSize);
  TILING_DATA_FIELD_DEF(int32_t, batchSizeX);
  TILING_DATA_FIELD_DEF(int32_t, batchSizeY);
  TILING_DATA_FIELD_DEF(int32_t, batchSizeZ);

  TILING_DATA_FIELD_DEF(bool, hasBias);
  TILING_DATA_FIELD_DEF(int32_t, biasBatch);
  TILING_DATA_FIELD_DEF(int32_t, biasMLength);
  TILING_DATA_FIELD_DEF(int32_t, biasNLength);

  TILING_DATA_FIELD_DEF(int32_t, singleM);
  TILING_DATA_FIELD_DEF(int32_t, singleN);
  TILING_DATA_FIELD_DEF(int32_t, blockDimM);
  TILING_DATA_FIELD_DEF(int32_t, blockDimN);
  TILING_DATA_FIELD_DEF(int32_t, selectApiTempSize);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);

  TILING_DATA_FIELD_DEF(int32_t, bigCoreNumM);
  TILING_DATA_FIELD_DEF(int32_t, bigCoreStrideM);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatMul, MatMulTilingData)
}
