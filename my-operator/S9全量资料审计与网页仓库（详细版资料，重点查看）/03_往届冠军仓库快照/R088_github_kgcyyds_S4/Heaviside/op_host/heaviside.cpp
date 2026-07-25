
#include "heaviside_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"

#define HEAVISIDE_TILING_0 0
#define HEAVISIDE_TILING_1 1

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  HeavisideTilingData tiling;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint64_t ubSize;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  uint32_t inputBytes = 0;
  ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), inputBytes);
  uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
  uint32_t valuesNum = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
  uint32_t inputDim = context->GetInputShape(0)->GetStorageShape().GetDimNum();
  uint32_t valuesDim = context->GetInputShape(1)->GetStorageShape().GetDimNum();
  uint32_t useCoreNum = (inputNum < 40) ? inputNum : 40;
  if(valuesNum != 1)
    useCoreNum = 1;
  uint32_t bigCoreNum = inputNum % useCoreNum;
  uint32_t smallCoreProcessNum = inputNum / useCoreNum;
  uint32_t bigCoreProcessNum = (bigCoreNum == 0) ? smallCoreProcessNum : smallCoreProcessNum + 1;
  uint32_t tileBlockNum = ubSize / BLOCK_SIZE / 3;
  uint32_t tileDataNum = tileBlockNum * BLOCK_SIZE / inputBytes;
  int repeatNum = 256 / inputBytes;
  tileDataNum = (tileDataNum + repeatNum - 1) / repeatNum * repeatNum;
  tiling.set_bigCoreNum(bigCoreNum);
  tiling.set_bigCoreProcessNum(bigCoreProcessNum);
  tiling.set_smallCoreProcessNum(smallCoreProcessNum);
  tiling.set_tileDataNum(tileDataNum);
  tiling.set_valuesNum(valuesNum);
  if(valuesNum != 1)
    context->SetTilingKey(HEAVISIDE_TILING_0);
  else
    context->SetTilingKey(HEAVISIDE_TILING_1);
  
  context->SetBlockDim(useCoreNum);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class Heaviside : public OpDef {
public:
    explicit Heaviside(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Heaviside);
}
