
#include "reduce_log_sum_exp_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  ReduceLogSumExpTilingData tiling;
  // const gert::StorageShape* x1_shape = context->GetInputShape(0);
  // int32_t data_sz = 1;
  // for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
  //     data_sz *= x1_shape->GetStorageShape().GetDim(i);
  // tiling.set_size(data_sz);
  // context->SetBlockDim(8);
  // tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
  // context->GetRawTilingData()->GetCapacity());
  // context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  // 获取数据类型 0 float 1 half 2 int8 3 int32
  auto dt = context->GetInputTensor(0)->GetDataType();

  // 获取UB内存大小
  // uint64_t ubSize;
  // auto ascendcPlatform =
  // platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  // ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

  // 获取AiCore的物理核数
  // auto coreNum = ascendcPlatform.GetCoreNum();

  // 获取维度信息
  auto dimension = context->GetInputShape(0)->GetStorageShape().GetDimNum();

  // 填写维度的详细信息
  uint32_t inputShape[4] = {};
  uint32_t dataSize = 1;
  for (int i = 0; i < dimension; i++) {
    inputShape[i] = context->GetInputShape(0)->GetStorageShape().GetDim(i);
    dataSize *= inputShape[i];
  }

  // 将上述计算的值全部回填到tiling中
  tiling.set_dataType(dt);
  tiling.set_inputShape(inputShape);
  tiling.set_dimension(dimension);
  tiling.set_dataSize(dataSize);
  std::cout << "Tiling data: dimension = " << dimension << ", inputShape = ["
            << inputShape[0] << ", " << inputShape[1] << ", " << inputShape[2]
            << ", " << inputShape[3] << "]"
            << ", dataType = " << dt << ", dataSize = " << dataSize
            << std::endl;

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->SetBlockDim(8);

  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* x1_shape = context->GetInputShape(0);
  gert::Shape* y_shape = context->GetOutputShape(0);
  *y_shape = *x1_shape;
  return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType);
  return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class ReduceLogSumExp : public OpDef {
 public:
  explicit ReduceLogSumExp(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("axes")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);

    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend310b");
  }
};

OP_ADD(ReduceLogSumExp);
}  // namespace ops
