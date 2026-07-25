
#include "gcd_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  GcdTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  const gert::StorageShape* x2_shape = context->GetInputShape(1);
  int N[5] = {1, 1, 1, 1, 1};
  int broadcast_mask = 0;
  int offset = x1_shape->GetStorageShape().GetDimNum() - x2_shape->GetStorageShape().GetDimNum();
  const int dim = std::max(x1_shape->GetStorageShape().GetDimNum(), x2_shape->GetStorageShape().GetDimNum());
  const int offsetA = dim - x1_shape->GetStorageShape().GetDimNum();
  const int offsetB = dim - x2_shape->GetStorageShape().GetDimNum();

  for (int i = 0; i < dim; i++) {
    if (i >= offsetA) {
        N[i] = x1_shape->GetStorageShape().GetDim(i - offsetA);
    }
    int M_i = i < offsetB ? 1 : x2_shape->GetStorageShape().GetDim(i - offsetB);
    if (M_i != N[i]) {
        broadcast_mask |= (1<<i);
    }
  }
  tiling.set_N0(N[0]);
  tiling.set_N1(N[1]);
  tiling.set_N2(N[2]);
  tiling.set_N3(N[3]);
  tiling.set_N4(N[4]);
  tiling.set_broadcast_mask(broadcast_mask);

  context->SetBlockDim(40);
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
