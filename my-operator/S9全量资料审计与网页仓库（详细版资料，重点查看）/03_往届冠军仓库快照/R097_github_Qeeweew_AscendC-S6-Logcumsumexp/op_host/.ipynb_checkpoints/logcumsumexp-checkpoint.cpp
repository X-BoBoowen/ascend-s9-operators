
#include "logcumsumexp_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  LogcumsumexpTilingData tiling;
  const gert::StorageShape* input_shape = context->GetInputShape(0);
  int dim_num = input_shape->GetStorageShape().GetDimNum();

  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  int dim = *attrs->GetAttrPointer<int>(0);
  uint32_t length = input_shape->GetStorageShape().GetDim(dim);
  uint32_t batch_size = 1;
  uint32_t batch_size_vec = 1;
  for (int i = 0; i < dim; i++) {
      batch_size *= input_shape->GetStorageShape().GetDim(i);
  }
  for (int i = dim + 1; i < dim_num; i++) {
    batch_size_vec *= input_shape->GetStorageShape().GetDim(i);
  }
  tiling.set_length(length);
  tiling.set_batch_size(batch_size);
  tiling.set_batch_size_vec(batch_size_vec);

  if (dim == dim_num - 1) {
    context->SetTilingKey(0);
  } else {
    context->SetTilingKey(1);
  }

  printf("%d %d %d\n", length, batch_size, batch_size_vec);
  context->SetBlockDim(20);
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
class Logcumsumexp : public OpDef {
public:
    explicit Logcumsumexp(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Logcumsumexp);
}
