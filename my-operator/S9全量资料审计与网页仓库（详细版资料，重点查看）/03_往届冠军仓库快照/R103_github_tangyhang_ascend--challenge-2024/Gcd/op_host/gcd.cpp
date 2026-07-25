
#include "gcd_tiling.h"
#include "register/op_def_registry.h"

#define GCD_TILING_0 1
#define GCD_TILING_1 2
#define GCD_TILING_2 3

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  GcdTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  const gert::StorageShape* x2_shape = context->GetInputShape(1);
  int x1_dimNum = x1_shape->GetStorageShape().GetDimNum();
  int x2_dimNum = x2_shape->GetStorageShape().GetDimNum();
  int x1_shapeSz = x1_shape->GetStorageShape().GetShapeSize();
  int x2_shapeSz = x2_shape->GetStorageShape().GetShapeSize();
  int batch_sz = x1_shapeSz;
  uint32_t useCoreNum = (batch_sz < 40) ? batch_sz : 40;
  uint32_t bigCoreNum = batch_sz % useCoreNum;
  uint32_t smallCoreProcessNum = batch_sz / useCoreNum;
  uint32_t bigCoreProcessNum = (bigCoreNum == 0) ? smallCoreProcessNum : smallCoreProcessNum + 1;
  uint32_t x1Arr[5], x2Arr[5];
  for(int i = 0; i < x1_dimNum; i++)
    x1Arr[i] = x1_shape->GetStorageShape().GetDim(i);
  for(int i = 0; i < x2_dimNum; i++)
    x2Arr[i] = x2_shape->GetStorageShape().GetDim(i);
  tiling.set_x1dimNum(x1_shapeSz);
  tiling.set_x2dimNum(x2_shapeSz);
  tiling.set_x1dimCnt(x1_dimNum);
  tiling.set_x2dimCnt(x2_dimNum);
  tiling.set_x1Arr(x1Arr);
  tiling.set_x2Arr(x2Arr);
  tiling.set_bigCoreNum(bigCoreNum);
  tiling.set_bigCoreProcessNum(bigCoreProcessNum);
  tiling.set_smallCoreProcessNum(smallCoreProcessNum);
  if(x1_shapeSz == x2_shapeSz)
    context->SetTilingKey(GCD_TILING_0);
  else
  {
    useCoreNum = 1;
    context->SetTilingKey(GCD_TILING_1);
  }
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
}


namespace ops {
class Gcd : public OpDef {
public:
    explicit Gcd(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Gcd);
}
