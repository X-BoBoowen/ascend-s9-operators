
#include "sort_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  SortTilingData tiling;
   // 获取数据类型 0 float 1 half 2 int8 3 int32
  auto dt = context->GetInputTensor(0)->GetDataType();
 //   auto dt1 = context->GetInputTensor(1)->GetDataType();

  auto axis = *(context->GetAttrs()->GetInt(0));

  bool ds = *(context->GetAttrs()->GetBool(1));

  bool st = *(context->GetAttrs()->GetBool(2));

  // std::cout << "batchDimsasdfgsda = " << bl << std::endl;
 
 
  // 获取UB内存大小
  uint64_t ubSize;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
 //   std::cout << "ubsize = " << ubSize << std::endl;
 
  // 获取AiCore的物理核数
  auto coreNum = ascendcPlatform.GetCoreNum();

  // 获取维度信息
  auto dimension_x = context->GetInputShape(0)->GetStorageShape().GetDimNum();
//   auto dimension_indices = context->GetInputShape(1)->GetStorageShape().GetDimNum();

  // 填写维度的详细信息
//   uint32_t inputXShape[5] = {};
//   for (int i = 0; i < dimension_x; ++i){
//     inputXShape[i] = context->GetInputShape(0)->GetStorageShape().GetDim(i);
//   }
//   uint32_t inputIndicesShape[3] = {};
  uint32_t dataSize = 1;
  for (int i = 0; i < dimension_x; ++i){
    dataSize *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
  }
 
  // 将上述计算的值全部回填到tiling中
  tiling.set_dataType(dt);
//   tiling.set_dim(dim);
//   tiling.set_inputXShape(inputXShape);
  tiling.set_dimension(dimension_x);
  tiling.set_dataSize(dataSize);
  tiling.set_axis(axis);
  tiling.set_descending(ds);
  tiling.set_stable(st);

 
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  // size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  // currentWorkspace[0] = 0;
  context->SetBlockDim(coreNum);
 
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
class Sort : public OpDef {
public:
    explicit Sort(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT8, ge::DT_UINT8, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT8, ge::DT_UINT8, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("axis").AttrType(OPTIONAL).Int(-1);
        this->Attr("descending").AttrType(OPTIONAL).Bool(false);
        this->Attr("stable").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Sort);
}
