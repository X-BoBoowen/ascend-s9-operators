#include "greater_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GreaterFastTilingData tiling;
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    uint64_t size = inputShape->GetStorageShape().GetShapeSize();
    uint32_t blockDim = 1;
    if (size >= 512 && size % 512 == 0) {
        const uint32_t blockUnits = static_cast<uint32_t>(size / 512);
        blockDim = blockUnits < 4 ? blockUnits : 4;
    }
    tiling.set_size(static_cast<uint32_t>(size / blockDim));
    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class GreaterFast : public OpDef {
public:
    explicit GreaterFast(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(GreaterFast);
}
