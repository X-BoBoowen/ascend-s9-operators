#include "transpose_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    const gert::Shape& shape = inputShape->GetStorageShape();
    const uint32_t rows = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t cols = static_cast<uint32_t>(shape.GetDim(1));
    const uint32_t tileCols = cols / 16;
    const uint32_t totalTiles = (rows / 16) * tileCols;
    const uint32_t blockDim = totalTiles < 32 ? totalTiles : 32;

    TransposeFastTilingData tiling;
    tiling.set_rows(rows);
    tiling.set_cols(cols);
    tiling.set_tileCols(tileCols);
    tiling.set_baseTilesPerBlock(totalTiles / blockDim);
    tiling.set_extraBlocks(totalTiles % blockDim);
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
    outputShape->SetDim(0, inputShape->GetDim(1));
    outputShape->SetDim(1, inputShape->GetDim(0));
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class TransposeFast : public OpDef {
public:
    explicit TransposeFast(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(TransposeFast);
}
