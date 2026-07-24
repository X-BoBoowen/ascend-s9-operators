#include "square_sum_v1_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    const gert::Shape& shape = inputShape->GetStorageShape();
    const size_t rank = shape.GetDimNum();
    const uint32_t reduceLen =
        static_cast<uint32_t>(shape.GetDim(rank - 1));
    const uint32_t outer =
        static_cast<uint32_t>(shape.GetShapeSize() / reduceLen);
    const uint32_t paddedReduce = (reduceLen + 15U) / 16U * 16U;
    const uint32_t blockDim = outer < 32U ? outer : 32U;

    SquareSumV1TilingData tiling;
    tiling.set_outer(outer);
    tiling.set_reduceLen(reduceLen);
    tiling.set_paddedReduce(paddedReduce);
    tiling.set_baseRowsPerBlock(outer / blockDim);
    tiling.set_extraBlocks(outer % blockDim);
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
    const size_t rank = inputShape->GetDimNum();
    outputShape->SetDim(rank - 1, 1);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class SquareSumV1 : public OpDef {
public:
    explicit SquareSumV1(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("axis").AttrType(REQUIRED).ListInt();
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SquareSumV1);
}
