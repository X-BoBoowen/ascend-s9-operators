
#include "logcumsumexp_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <iostream>
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    LogcumsumexpTilingData tiling;

    auto shape_x = context->GetInputTensor(0)->GetOriginShape();

    int32_t x_ndarray[10];
    int32_t x_dimensional;
    int32_t size = 1;

    x_dimensional = shape_x.GetDimNum();

    for(int i = 0; i < x_dimensional; i++)
    {
        x_ndarray[i] = shape_x.GetDim(i);
        std::cout<<i<<' '<<x_ndarray[i]<<'\n';
        size *= x_ndarray[i];
    }

    tiling.set_size(size);
    tiling.set_x_ndarray(x_ndarray);
    tiling.set_x_dimensional(x_dimensional);

    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); 
    
    int32_t tileDataMaxNum = ubSize / 32 / 4 / 9 * 32;
    tiling.set_tileDataMaxNum(tileDataMaxNum);
    auto axis_val= context->GetAttrs()->GetInt(0);
    tiling.set_axis(*axis_val);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
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
}


namespace ops {
class Logcumsumexp : public OpDef {
public:
    explicit Logcumsumexp(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").Int();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Logcumsumexp);
}
