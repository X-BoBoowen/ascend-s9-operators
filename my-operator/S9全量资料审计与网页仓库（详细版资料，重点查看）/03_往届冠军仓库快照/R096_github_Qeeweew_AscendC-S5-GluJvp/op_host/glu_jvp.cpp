
#include "glu_jvp_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion != platform_ascendc::SocVersion::ASCEND910B) {
        return ge::GRAPH_FAILED;
    }
    GluJvpTilingData tiling;
    const gert::StorageShape* input_shape = context->GetInputShape(1);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    int dim = *(attrs->GetAttrPointer<int>(0));
    int input_dim = input_shape->GetStorageShape().GetDimNum();
    if (dim < 0) {
        dim += input_dim;
    }

    int batch_size = 1;
    int length = 1;

    for (int i = 0; i < dim; i++) {
        batch_size *= input_shape->GetStorageShape().GetDim(i);
    }
    for (int i = dim; i < input_dim; i++) {
        length *= input_shape->GetStorageShape().GetDim(i);
    }
    printf("batch_size: %d, length: %d\n", batch_size, length);
    tiling.set_batch_size(batch_size);
    tiling.set_length(length);
    uint32_t n_core = ascendcPlatform.GetCoreNumAiv();
    context->SetBlockDim(n_core);
    if (batch_size >= n_core) {
        context->SetTilingKey(1);
    } else {
        context->SetTilingKey(2);
    }
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
class GluJvp : public OpDef {
public:
    explicit GluJvp(const char* name) : OpDef(name)
    {
        this->Input("glu_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("jvp_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(-1);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(GluJvp);
}
