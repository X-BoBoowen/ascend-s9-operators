
#include "logcumsumexp_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cstdint>

constexpr int32_t LARGE_CYCLES=100000;
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    LogcumsumexpTilingData tiling;

    auto shape_x = context->GetInputTensor(0)->GetOriginShape();
    auto dim = *(context->GetAttrs()->GetInt(0));

    int32_t input_ndarray[10];
    int32_t input_dimensional;
    uint64_t interval = 1;

    input_dimensional = shape_x.GetDimNum();

    for(int i = 0; i < input_dimensional; i++)
    {
        input_ndarray[i] = shape_x.GetDim(i);
        // size *= input_ndarray[i];
    }

    int32_t loopCount = 1;

    for (int32_t i = 0; i < dim; i++) {
      loopCount *= input_ndarray[i];
    }
    for (int32_t i = dim + 1; i < input_dimensional; i++) {
      interval *= input_ndarray[i];
    }

    int32_t cycles = input_ndarray[dim];

    tiling.set_interval(interval);
    tiling.set_dim(*context->GetAttrs()->GetInt(0));
    tiling.set_input_ndarray(input_ndarray);
    tiling.set_input_dimensional(input_dimensional);

    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); //获取硬件平台存储空间 UB 的内存大小
    ubSize=ubSize-8192*sizeof(uint8_t);

    int32_t tileDataMaxNum;
    auto dt = context->GetInputTensor(0)->GetDataType();
    if (dt == ge::DT_FLOAT16){
        int32_t max_mem_size=4*2;
        if (cycles>LARGE_CYCLES){
            tileDataMaxNum = ubSize / (2*4+4*4+max_mem_size) / 32 * 32;// *6;
        }else{
            // tileDataMaxNum = ubSize / 2 / 5 / 32 * 32;// *6;
            tileDataMaxNum = ubSize / (2*4+3*4+max_mem_size) / 32 * 32;// *6;
        }
    } else if(dt == ge::DT_BF16){  // 
        int32_t max_mem_size=4*2;
        if (cycles>LARGE_CYCLES){
            tileDataMaxNum = ubSize / (2*4+4*4+max_mem_size) / 32 * 32;// *6;
        }else{
            tileDataMaxNum = ubSize / (2*4+3*4+max_mem_size) / 32 * 32;// *6;
        }
    } else{ // float32
        int32_t max_mem_count=2;
        if (cycles>LARGE_CYCLES){
            tileDataMaxNum = ubSize / 4 / (6+max_mem_count) / 32 * 32;// *6;
        }else{
            tileDataMaxNum = ubSize / 4 / (5+max_mem_count) / 32 * 32;// *6;
        }
    }

    uint32_t  aivNum = std::min((uint32_t)ascendcPlatform.GetCoreNumAiv(),(uint32_t)loopCount);
    tiling.set_tileDataMaxNum(tileDataMaxNum);//(5*1024);//(30*1024); //每次处理的数据量

    context->SetBlockDim(aivNum);
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
