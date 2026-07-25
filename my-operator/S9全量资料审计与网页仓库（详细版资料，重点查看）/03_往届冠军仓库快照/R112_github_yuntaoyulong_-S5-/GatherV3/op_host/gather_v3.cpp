#include "gather_v3_tiling.h"
#include "host_inc.h"
#include "register/op_def_registry.h"

using std::vector;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  // 获取Ascend平台信息
  uint64_t ub_size;
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  auto aivNum = ascendcPlatform.GetCoreNum();
  context->SetBlockDim(aivNum);

  // 获取输入张量形状
  gert::Shape originShape = context->GetInputTensor(0)->GetOriginShape();
  const int32_t shapeSize = originShape.GetDimNum();
  vector<int32_t> xShape(shapeSize);
  for (int i = 0; i < shapeSize; ++i)
    xShape[i] = originShape[i];

  // 获取batch_dims参数（默认0）
  int32_t batch_dims = 0;
  if (context->GetAttrs()->GetInt(0)) {
    batch_dims = *context->GetAttrs()->GetInt(0);
  }
  if (batch_dims < 0)
    batch_dims += shapeSize;

  // 获取axis输入（第三输入张量）
  int32_t axis = 0;
  const gert::Tensor *axisTensor = context->GetInputTensor(2);
  if (axisTensor) {
    // 正确获取axis数据的方式
    uint32_t dataSize = axisTensor->GetSize();
    if (dataSize >= sizeof(int32_t)) {
      // 使用GetDataOrigin方式获取原始数据
      const void *data = axisTensor->GetData<int32_t>();
      if (data) {
        std::memcpy(&axis, data, sizeof(int32_t));
      }
    }
  }

  // 归一化轴（处理负索引）
  if (axis < 0)
    axis += shapeSize;
  if (axis < 0 || axis >= shapeSize || axis < batch_dims) {
    return GRAPH_FAILED;
  }

  // 计算各维度长度
  uint32_t xTotalLength = context->GetInputTensor(0)->GetShapeSize();
  uint32_t indicesTotalLength = context->GetInputTensor(1)->GetShapeSize();

  // 计算批次数量
  uint32_t batchNumber = 1;
  for (int32_t i = 0; i < batch_dims; ++i) {
    batchNumber *= xShape[i];
  }

  // 计算内部维度（batch_dims到axis-1）
  // uint32_t innDimLength = 1;
  // for(int32_t i = batch_dims; i < axis; ++i) {
  //     innDimLength *= xShape[i];
  // }

  // 计算外部维度（axis+1到结束）
  uint32_t outerDimLength = 1;
  for (int32_t i = axis + 1; i < shapeSize; ++i) {
    outerDimLength *= xShape[i];
  }
  uint32_t innDimLength = 1;
  for (int32_t i = batch_dims; i < axis; ++i) {
    innDimLength *= xShape[i];
  }
  // 获取轴维度长度
  const uint32_t axisDim = xShape[axis];
  const uint32_t indicesLength = indicesTotalLength / batchNumber;
  const uint32_t batchLength = innDimLength * axisDim * outerDimLength;
  // 数据类型大小
  uint32_t sizeOfDataType =
      GetSizeByDataType(context->GetInputTensor(0)->GetDataType());
  const uint32_t innerDimLength = innDimLength;
  // 根据外维度大小选择tiling策略
  if (outerDimLength * sizeOfDataType <= 32) {
    // 标量拷贝模式 (小数据)
    context->SetTilingKey(1);
    GatherTilingDataScalarCopy tiling;
    SET(batchNumber);

    tiling.set_innerDimLength(innerDimLength);

    SET(axisDim);

    SET(outerDimLength);
    SET(batchLength);
    SET(indicesLength);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  } else {
    // 数据块拷贝模式 (大数据)
    context->SetTilingKey(0);
    GatherTilingDataWithDataCopy tiling;

    // 计算分块参数
    uint32_t tileLength =
        FLOOR(ub_size / (2 * sizeOfDataType), 512 / sizeOfDataType);
    uint32_t tileNumber = outerDimLength / tileLength;
    uint32_t maxLength = tileNumber * tileLength;
    uint32_t reminder = CEIL(outerDimLength % tileLength, 32 / sizeOfDataType);
    // 设置tiling参数
    SET(batchNumber);

    tiling.set_innerDimLength(innerDimLength);
    SET(batchLength);
    SET(axisDim);

    SET(outerDimLength);

    SET(indicesLength);
    SET(maxLength);
    SET(tileLength);
    SET(reminder);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  }
  return GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
  const gert::Shape *x_shape = context->GetInputShape(0);
  const gert::Shape *indices_shape = context->GetInputShape(1);

  // 创建输出形状
  int32_t x_rank = x_shape->GetDimNum();
  gert::Shape y_shape = *x_shape;

  // 获取axis参数
  if (context->GetInputTensor(2)->GetSize() < sizeof(int32_t)) {
    return GRAPH_FAILED;
  }
  int32_t axis = context->GetInputTensor(2)->GetData<int32_t>()[0];

  // 归一化轴
  if (axis < 0)
    axis += x_rank;
  if (axis < 0 || axis >= x_rank) {
    return GRAPH_FAILED;
  }

  // 替换axis维度为索引维度
  gert::Shape *output_shape = context->GetOutputShape(0);
  if (output_shape == nullptr) {
    return GRAPH_FAILED;
  }
  *output_shape = y_shape;
  return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class GatherV3 : public OpDef {
public:
  explicit GatherV3(const char *name) : OpDef(name) {
    // 定义输入输出
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat(
            {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

    this->Input("indices")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat(
            {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

    this->Input("axis")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat(
            {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat(
            {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

    // 定义属性
    this->Attr("batch_dims").AttrType(OPTIONAL).Int(0);
    this->Attr("negative_index_support").AttrType(OPTIONAL).Bool(false);

    // 设置推理和tiling函数
    this->SetInferShape(ge::InferShape);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend310b");
  }
};

OP_ADD(GatherV3);
} // namespace ops