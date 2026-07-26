#include "square_sum_v1_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
namespace {
constexpr uint32_t MAX_BLOCK_DIM = 32;
// Keep headroom under typical 910B Vector UB (~192KB).
// Per-tile footprint (BUFFER_NUM=2):
//   xQueue:  2 * tileRows * paddedReduce * sizeof(half)
//   yQueue:  2 * align16(tileRows) * sizeof(half)
//   xFloat:  tileRows * paddedReduce * sizeof(float)
//   sumFloat: align8(tileRows) * sizeof(float)
// ≈ tileRows * paddedReduce * 8 + O(tileRows)
constexpr uint32_t UB_BUDGET_BYTES = 128 * 1024;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t MAX_COPY_ROWS = 4095;

inline uint32_t MinU32(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

inline uint32_t MaxU32(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

inline uint32_t Align16(uint32_t value)
{
    return (value + 15U) / 16U * 16U;
}

inline uint32_t Align8(uint32_t value)
{
    return (value + 7U) / 8U * 8U;
}

inline uint32_t EstimateTileBytes(uint32_t tileRows, uint32_t paddedReduce)
{
    const uint32_t xQueueBytes =
        BUFFER_NUM * tileRows * paddedReduce * static_cast<uint32_t>(sizeof(uint16_t));
    const uint32_t yQueueBytes =
        BUFFER_NUM * Align16(tileRows) * static_cast<uint32_t>(sizeof(uint16_t));
    const uint32_t xFloatBytes =
        tileRows * paddedReduce * static_cast<uint32_t>(sizeof(float));
    const uint32_t sumFloatBytes =
        Align8(tileRows) * static_cast<uint32_t>(sizeof(float));
    return xQueueBytes + yQueueBytes + xFloatBytes + sumFloatBytes;
}

inline uint32_t ComputeTileRows(uint32_t maxRowsPerBlock, uint32_t paddedReduce)
{
    if (paddedReduce == 0U || maxRowsPerBlock == 0U) {
        return 1U;
    }

    // Binary search largest tileRows that fits the UB budget.
    uint32_t lo = 1U;
    uint32_t hi = MinU32(maxRowsPerBlock, MAX_COPY_ROWS);
    uint32_t best = 1U;
    while (lo <= hi) {
        const uint32_t mid = lo + (hi - lo) / 2U;
        if (EstimateTileBytes(mid, paddedReduce) <= UB_BUDGET_BYTES) {
            best = mid;
            lo = mid + 1U;
        } else {
            if (mid == 0U) {
                break;
            }
            hi = mid - 1U;
        }
    }
    return MaxU32(1U, best);
}
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    const gert::Shape& shape = inputShape->GetStorageShape();
    const size_t rank = shape.GetDimNum();
    const uint32_t reduceLen =
        static_cast<uint32_t>(shape.GetDim(rank - 1));
    const uint32_t outer =
        static_cast<uint32_t>(shape.GetShapeSize() / reduceLen);
    const uint32_t paddedReduce = Align16(reduceLen);
    const uint32_t blockDim = outer < MAX_BLOCK_DIM ? outer : MAX_BLOCK_DIM;
    const uint32_t baseRowsPerBlock = outer / blockDim;
    const uint32_t extraBlocks = outer % blockDim;
    const uint32_t maxRowsPerBlock = baseRowsPerBlock + (extraBlocks > 0U ? 1U : 0U);
    const uint32_t tileRows = ComputeTileRows(maxRowsPerBlock, paddedReduce);

    SquareSumV1TilingData tiling;
    tiling.set_outer(outer);
    tiling.set_reduceLen(reduceLen);
    tiling.set_paddedReduce(paddedReduce);
    tiling.set_baseRowsPerBlock(baseRowsPerBlock);
    tiling.set_extraBlocks(extraBlocks);
    tiling.set_tileRows(tileRows);
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
