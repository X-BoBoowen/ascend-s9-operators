
#include "bitwise_left_shift_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define TILING_0 1
#define TILING_1 2
#define TILING_2 3

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  BitwiseLeftShiftTilingData tiling;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint64_t ubSize;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  auto num_cores = ascendcPlatform.GetCoreNum();
  uint32_t sizeofdata = 1;
  auto dt = context->GetInputTensor(0)->GetDataType();
  if(dt == ge::DT_INT8) sizeofdata = 1;
  else if(dt == ge::DT_INT16) sizeofdata = 2;
  else if(dt == ge::DT_INT32) sizeofdata = 4;
  else sizeofdata = 8;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  auto dim1 = x1_shape->GetStorageShape().GetDimNum();
  uint32_t n1[3] = {1, 1, 1};
  for(int i = 0; i < dim1; i ++)
    n1[i] = x1_shape->GetStorageShape().GetDim(i);
  const gert::StorageShape* x2_shape = context->GetInputShape(1);
  auto dim2 = x2_shape->GetStorageShape().GetDimNum();
  uint32_t n2[3] = {1, 1, 1};
  for(int i = 0; i < dim2; i ++)
    n2[i + (dim1 - dim2)] = x2_shape->GetStorageShape().GetDim(i);
  uint32_t batch_sz = x1_shape->GetStorageShape().GetShapeSize();
  if(n1[0] == n2[0] && n1[1] != n2[1] && n1[2] == n2[2])
    batch_sz = n1[0];
  uint32_t useCoreNum = (batch_sz < num_cores) ? batch_sz : num_cores;
  uint32_t bigCoreNum = batch_sz % useCoreNum;
  uint32_t smallCoreProcessNum = batch_sz / useCoreNum;
  uint32_t bigCoreProcessNum = (bigCoreNum == 0) ? smallCoreProcessNum : smallCoreProcessNum + 1;
  uint32_t tileBlockNum = ubSize / BLOCK_SIZE;
  if(sizeofdata == 4)
    tileBlockNum /= 3;
  uint32_t tileDataNum = tileBlockNum * BLOCK_SIZE / sizeofdata;
  tiling.set_n1(n1);
  tiling.set_n2(n2);
  tiling.set_tileDataNum(tileDataNum);
  tiling.set_bigCoreNum(bigCoreNum);
  tiling.set_bigCoreProcessNum(bigCoreProcessNum);
  tiling.set_smallCoreProcessNum(smallCoreProcessNum);
  int status = 1;
  for(int i = 0; i < 3; i ++)
    if(n1[i] != n2[i])
      status = 0;
  if(status == 0)
  {
    if(n1[0] == n2[0] && n1[1] != n2[1] && n1[2] == n2[2])
      context->SetTilingKey(TILING_2);
    else
      context->SetTilingKey(TILING_0);
  }
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
class BitwiseLeftShift : public OpDef {
public:
    explicit BitwiseLeftShift(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("other")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(BitwiseLeftShift);
}
