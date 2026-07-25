
#include "fractional_max_pool3_d_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define TILING_0 1
#define TILING_1 2

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  FractionalMaxPool3DTilingData tiling;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto num_cores = ascendcPlatform.GetCoreNum();
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  const gert::TypedContinuousVector<int64_t>* kernel_size = context->GetAttrs()->GetListInt(0);
  const gert::TypedContinuousVector<int64_t>* output_size = context->GetAttrs()->GetListInt(1);
  const float* output_ratio_t = context->GetAttrs()->GetFloat(2);
  const float* output_ratio_h = context->GetAttrs()->GetFloat(3);
  const float* output_ratio_w = context->GetAttrs()->GetFloat(4);
  const bool* return_indices = context->GetAttrs()->GetBool(5);
  auto inputDimNum = x1_shape->GetStorageShape().GetDimNum();
  uint32_t n, c;
  uint32_t kernelSize[3], inputSize[3], outputSize[3];
  if(kernel_size->GetSize() == 1)
  {
    for(int i = 0; i < 3; i ++)
      kernelSize[i] = *kernel_size->GetData();
  }
  else
  {
    for(int i = 0; i < 3; i ++)
      kernelSize[i] = *(kernel_size->GetData() + i);
  }
  if(inputDimNum == 4)
  {
    n = 1, c = x1_shape->GetStorageShape().GetDim(0);
    for(int i = 0; i < 3; i ++)
      inputSize[i] = x1_shape->GetStorageShape().GetDim(i + 1);
  }
  else
  {
    n = x1_shape->GetStorageShape().GetDim(0), c = x1_shape->GetStorageShape().GetDim(1);
    for(int i = 0; i < 3; i ++)
      inputSize[i] = x1_shape->GetStorageShape().GetDim(i + 2);
  }
  if(output_size->GetSize() == 3)
  {
    for(int i = 0; i < 3; i ++)
      outputSize[i] = *(output_size->GetData() + i);
  }
  else
  {
    outputSize[0] = (*output_ratio_t) * inputSize[0];
    outputSize[1] = (*output_ratio_h) * inputSize[1];
    outputSize[2] = (*output_ratio_w) * inputSize[2];
  }
  uint64_t batch_sz = (uint64_t)n * c * outputSize[0] * outputSize[1] * outputSize[2];
  uint32_t useCoreNum = (batch_sz < num_cores) ? batch_sz : num_cores;
  uint32_t bigCoreNum = batch_sz % useCoreNum;
  uint64_t smallCoreProcessNum = batch_sz / useCoreNum;
  uint64_t bigCoreProcessNum = (bigCoreNum == 0) ? smallCoreProcessNum : smallCoreProcessNum + 1;
  tiling.set_kernelSize(kernelSize);
  tiling.set_inputSize(inputSize);
  tiling.set_outputSize(outputSize);
  tiling.set_n(n);
  tiling.set_c(c);
  tiling.set_bigCoreNum(bigCoreNum);
  tiling.set_bigCoreProcessNum(bigCoreProcessNum);
  tiling.set_smallCoreProcessNum(smallCoreProcessNum);
  if(*return_indices == false)
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
class FractionalMaxPool3D : public OpDef {
public:
    explicit FractionalMaxPool3D(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("random_sample")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("kernel_size").ListInt();
        this->Attr("output_size").AttrType(OPTIONAL).ListInt({});
        this->Attr("output_ratio_t").AttrType(OPTIONAL).Float(0.0);
        this->Attr("output_ratio_h").AttrType(OPTIONAL).Float(0.0);
        this->Attr("output_ratio_w").AttrType(OPTIONAL).Float(0.0);
        this->Attr("return_indices").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(FractionalMaxPool3D);
}
