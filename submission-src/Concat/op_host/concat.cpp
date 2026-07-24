#include <algorithm>
#include <cstdint>

#include "concat_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const uint32_t inputCount = static_cast<uint32_t>(
        *context->GetAttrs()->GetInt(0));
    const gert::StorageShape* firstShape =
        context->GetRequiredInputShape(0);
    const gert::Shape& shape = firstShape->GetStorageShape();
    const size_t rank = shape.GetDimNum();

    uint64_t outer64 = 1;
    for (size_t i = 0; i + 1 < rank; ++i) {
        outer64 *= static_cast<uint64_t>(shape.GetDim(i));
    }
    const uint32_t outer = static_cast<uint32_t>(outer64);
    const gert::StorageShape* outputShape = context->GetOutputShape(0);
    const uint32_t outInner = static_cast<uint32_t>(
        outputShape->GetStorageShape().GetDim(rank - 1));

    uint32_t widths[32] = {};
    uint32_t offsets[32] = {};
    uint32_t runningOffset = 0;
    uint32_t maxWidth = 0;
    for (uint32_t i = 0; i < inputCount; ++i) {
        const gert::StorageShape* currentShape =
            i == 0
                ? context->GetRequiredInputShape(0)
                : context->GetOptionalInputShape(i);
        const gert::Shape& current =
            currentShape->GetStorageShape();
        const uint32_t width =
            static_cast<uint32_t>(current.GetDim(rank - 1));
        widths[i] = width;
        offsets[i] = runningOffset;
        runningOffset += width;
        maxWidth = std::max(maxWidth, width);
    }

    constexpr uint32_t MAX_BLOCK_DIM = 16;
    constexpr uint32_t UB_BUDGET_BYTES = 32 * 1024;
    constexpr uint32_t MAX_COPY_ROWS = 4095;
    const uint32_t blockDim = std::min(outer, MAX_BLOCK_DIM);
    const uint32_t maxAlignedWidth = (maxWidth + 15U) / 16U * 16U;
    const uint32_t maxRowsPerBlock =
        (outer + blockDim - 1U) / blockDim;
    const uint32_t capacityRows = std::max(
        1U,
        UB_BUDGET_BYTES /
            (maxAlignedWidth * static_cast<uint32_t>(sizeof(uint16_t))));
    const uint32_t tileRows = std::min(
        maxRowsPerBlock,
        std::min(capacityRows, MAX_COPY_ROWS));

    ConcatFastTilingData tiling;
    tiling.set_outer(outer);
    tiling.set_outInner(outInner);
    tiling.set_inputCount(inputCount);
    tiling.set_baseRowsPerBlock(outer / blockDim);
    tiling.set_extraBlocks(outer % blockDim);
    tiling.set_tileRows(tileRows);
    tiling.set_maxAlignedWidth(maxAlignedWidth);
    tiling.set_widths(widths);
    tiling.set_offsets(offsets);

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
    const size_t inputCount = static_cast<size_t>(
        *context->GetAttrs()->GetInt(0));
    const gert::Shape* firstShape =
        context->GetRequiredInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    *outputShape = *firstShape;

    const size_t rank = firstShape->GetDimNum();
    const int64_t rawDim = *context->GetAttrs()->GetInt(1);
    const size_t dim = static_cast<size_t>(
        rawDim < 0 ? rawDim + static_cast<int64_t>(rank) : rawDim);
    int64_t total = 0;
    for (size_t i = 0; i < inputCount; ++i) {
        const gert::Shape* currentShape =
            i == 0
                ? context->GetRequiredInputShape(0)
                : context->GetOptionalInputShape(i);
        total += currentShape->GetDim(dim);
    }
    outputShape->SetDim(dim, total);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class ConcatFast : public OpDef {
public:
    explicit ConcatFast(const char* name) : OpDef(name)
    {
        this->Input("x0")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
#define ADD_OPTIONAL_INPUT(input_name)             \
        this->Input(input_name)                    \
            .ParamType(OPTIONAL)                   \
            .DataType({ge::DT_FLOAT16})            \
            .Format({ge::FORMAT_ND})               \
            .UnknownShapeFormat({ge::FORMAT_ND})
        ADD_OPTIONAL_INPUT("x1");
        ADD_OPTIONAL_INPUT("x2");
        ADD_OPTIONAL_INPUT("x3");
        ADD_OPTIONAL_INPUT("x4");
        ADD_OPTIONAL_INPUT("x5");
        ADD_OPTIONAL_INPUT("x6");
        ADD_OPTIONAL_INPUT("x7");
        ADD_OPTIONAL_INPUT("x8");
        ADD_OPTIONAL_INPUT("x9");
        ADD_OPTIONAL_INPUT("x10");
        ADD_OPTIONAL_INPUT("x11");
        ADD_OPTIONAL_INPUT("x12");
        ADD_OPTIONAL_INPUT("x13");
        ADD_OPTIONAL_INPUT("x14");
        ADD_OPTIONAL_INPUT("x15");
#undef ADD_OPTIONAL_INPUT
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("inputCount").AttrType(REQUIRED).Int();
        this->Attr("dim").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConcatFast);
}
