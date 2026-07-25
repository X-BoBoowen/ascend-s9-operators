
#include "eye_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  EyeTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  const int64_t *num_rows = context->GetAttrs()->GetInt(0);
  const int64_t *num_columns = context->GetAttrs()->GetInt(1);
  const int64_t *dtype = context->GetAttrs()->GetInt(3);
  int32_t batch_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum() - 2; i++)
    batch_sz *= x1_shape->GetStorageShape().GetDim(i);
  uint32_t useCoreNum = (batch_sz < 40) ? batch_sz : 40;
  uint32_t bigCoreNum = batch_sz % useCoreNum;
  uint32_t smallCoreProcessNum = batch_sz / useCoreNum;
  uint32_t bigCoreProcessNum = (bigCoreNum == 0) ? smallCoreProcessNum : smallCoreProcessNum + 1;
  tiling.set_row(*num_rows);
  tiling.set_col(*num_columns);
  tiling.set_bigCoreNum(bigCoreNum);
  tiling.set_bigCoreProcessNum(bigCoreProcessNum);
  tiling.set_smallCoreProcessNum(smallCoreProcessNum);
  tiling.set_dtype(*dtype);
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
class Eye : public OpDef {
public:
    explicit Eye(const char* name) : OpDef(name)
    {
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_DOUBLE})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_DOUBLE})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("num_rows").Int();
        this->Attr("num_columns").Int();
        this->Attr("batch_shape").ListInt();
        this->Attr("dtype").Int();

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Eye);
}
