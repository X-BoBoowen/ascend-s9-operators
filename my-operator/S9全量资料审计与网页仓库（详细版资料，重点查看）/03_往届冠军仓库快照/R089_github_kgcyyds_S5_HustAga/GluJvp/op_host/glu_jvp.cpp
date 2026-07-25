
#include "glu_jvp_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define TILING_0 1
#define TILING_1 2

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  GluJvpTilingData tiling;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint64_t ubSize;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  auto num_cores = ascendcPlatform.GetCoreNum();
  auto dt = context->GetInputTensor(0)->GetDataType();
  uint32_t sizeofdata = 2;
  if(dt == ge::DT_FLOAT)
    sizeofdata = 4;
  const int64_t *dim = context->GetAttrs()->GetInt(0);
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  const gert::StorageShape* x2_shape = context->GetInputShape(1);
  auto dim1 = x1_shape->GetStorageShape().GetDimNum();
  uint32_t stride = 1;
  if(*dim < 0)
  {
    for(int i = *dim + dim1; i < dim1; i ++)
      stride *= x2_shape->GetStorageShape().GetDim(i);
  }
  else
  {
    for(int i = *dim; i < dim1; i ++)
      stride *= x2_shape->GetStorageShape().GetDim(i);
  }
  stride /= 2;
  uint32_t batch_sz = x1_shape->GetStorageShape().GetShapeSize() / stride;
  uint32_t useCoreNum = (batch_sz < num_cores) ? batch_sz : num_cores;
  uint32_t bigCoreNum = batch_sz % useCoreNum;
  uint32_t smallCoreProcessNum = batch_sz / useCoreNum;
  uint32_t bigCoreProcessNum = (bigCoreNum == 0) ? smallCoreProcessNum : smallCoreProcessNum + 1;
  uint32_t tileBlockNum = ubSize / BLOCK_SIZE;
  if(dt == ge::DT_FLOAT)
    tileBlockNum /= 6;
  else
    tileBlockNum /= 14;
  uint32_t tileDataNum = tileBlockNum * BLOCK_SIZE / sizeofdata;
  tiling.set_stride(stride);
  tiling.set_tileDataNum(tileDataNum);
  tiling.set_bigCoreNum(bigCoreNum);
  tiling.set_bigCoreProcessNum(bigCoreProcessNum);
  tiling.set_smallCoreProcessNum(smallCoreProcessNum);
  if(stride <= tileDataNum)
    context->SetTilingKey(TILING_0);
  else
    context->SetTilingKey(TILING_1);
  context->SetBlockDim(useCoreNum);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class GluJvp : public OpDef {
public:
    explicit GluJvp(const char* name) : OpDef(name)
    {
        this->Input("glu_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("jvp_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(GluJvp);
}
