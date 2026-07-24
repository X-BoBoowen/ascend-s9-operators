#include <algorithm>

#include "index_add_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* selfShape = context->GetInputShape(0);
    const gert::StorageShape* indexShape = context->GetInputShape(1);
    const gert::StorageShape* sourceShape = context->GetInputShape(2);
    const uint32_t outputRows = static_cast<uint32_t>(
        selfShape->GetStorageShape().GetDim(0));
    const uint32_t indexCount = static_cast<uint32_t>(
        indexShape->GetStorageShape().GetShapeSize());
    const uint32_t rowWidth = static_cast<uint32_t>(
        sourceShape->GetStorageShape().GetDim(1));
    constexpr uint32_t ROWS_PER_BLOCK = 8;
    constexpr uint32_t MAX_BLOCK_DIM = 32;
    const uint32_t blockDim = std::min(
        (indexCount + ROWS_PER_BLOCK - 1U) / ROWS_PER_BLOCK,
        MAX_BLOCK_DIM);

    IndexAddFastTilingData tiling;
    tiling.set_outputRows(outputRows);
    tiling.set_indexCount(indexCount);
    tiling.set_rowWidth(rowWidth);
    tiling.set_baseRowsPerBlock(indexCount / blockDim);
    tiling.set_extraBlocks(indexCount % blockDim);
    context->SetBlockDim(blockDim);
    context->SetTilingKey(indexCount % ROWS_PER_BLOCK == 0 ? 1 : 0);
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
class IndexAddFast : public OpDef {
public:
    explicit IndexAddFast(const char* name) : OpDef(name)
    {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("source")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(IndexAddFast);
}
