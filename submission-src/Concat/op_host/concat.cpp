#include <cstdint>
#include <limits>

#include "concat_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static bool IsSpecialEmpty(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == 0;
}

static uint32_t GetElementBytes(const ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_INT8:
            return 1;
        case ge::DT_FLOAT16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        default:
            return 0;
    }
}

static bool SafeMultiply(
    const uint64_t left,
    const uint64_t right,
    uint64_t& result)
{
    if (left != 0 &&
        right > std::numeric_limits<uint64_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t inputCount = static_cast<uint32_t>(
        context->GetComputeNodeInputNum());
    constexpr uint32_t MAX_INPUT_COUNT = 256;
    if (inputCount == 0 || inputCount > MAX_INPUT_COUNT) {
        return ge::GRAPH_FAILED;
    }

    uint32_t referenceInput = 0;
    for (uint32_t i = 0; i < inputCount; ++i) {
        const gert::StorageShape* candidate =
            context->GetDynamicInputShape(0, i);
        if (candidate == nullptr) {
            return ge::GRAPH_FAILED;
        }
        if (!IsSpecialEmpty(candidate->GetStorageShape())) {
            referenceInput = i;
            break;
        }
    }
    const gert::StorageShape* referenceShape =
        context->GetDynamicInputShape(0, referenceInput);
    if (referenceShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& shape = referenceShape->GetStorageShape();
    const size_t rank = shape.GetDimNum();
    constexpr size_t MAX_RANK = 8;
    if (rank == 0 || rank > MAX_RANK) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr || attrs->GetInt(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t rawDim = *attrs->GetInt(0);
    const int64_t normalizedDim =
        rawDim < 0 ? rawDim + static_cast<int64_t>(rank) : rawDim;
    if (normalizedDim < 0 ||
        normalizedDim >= static_cast<int64_t>(rank)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t dim = static_cast<uint32_t>(normalizedDim);

    uint64_t outer64 = 1;
    for (size_t i = 0; i < dim; ++i) {
        const int64_t extent = shape.GetDim(i);
        if (extent < 0 ||
            !SafeMultiply(
                outer64,
                static_cast<uint64_t>(extent),
                outer64)) {
            return ge::GRAPH_FAILED;
        }
    }

    uint64_t innerElements = 1;
    for (size_t i = dim + 1; i < rank; ++i) {
        const int64_t extent = shape.GetDim(i);
        if (extent < 0 ||
            !SafeMultiply(
                innerElements,
                static_cast<uint64_t>(extent),
                innerElements)) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Tensor* firstTensor =
        context->GetDynamicInputTensor(0, referenceInput);
    if (firstTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t elementBytes =
        GetElementBytes(firstTensor->GetDataType());
    if (elementBytes == 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t concatenatedDim = 0;
    uint64_t inputRowBytes[MAX_INPUT_COUNT] = {};
    for (uint32_t i = 0; i < inputCount; ++i) {
        const gert::StorageShape* currentShape =
            context->GetDynamicInputShape(0, i);
        const gert::Tensor* currentTensor =
            context->GetDynamicInputTensor(0, i);
        if (currentShape == nullptr ||
            currentTensor == nullptr ||
            currentTensor->GetDataType() != firstTensor->GetDataType()) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape& current =
            currentShape->GetStorageShape();
        if (IsSpecialEmpty(current)) {
            inputRowBytes[i] = 0;
            continue;
        }
        if (current.GetDimNum() != rank) {
            return ge::GRAPH_FAILED;
        }
        for (size_t axis = 0; axis < rank; ++axis) {
            const int64_t extent = current.GetDim(axis);
            if (extent < 0 ||
                (axis != dim &&
                 extent != shape.GetDim(axis))) {
                return ge::GRAPH_FAILED;
            }
        }
        const uint64_t currentDim =
            static_cast<uint64_t>(current.GetDim(dim));
        if (currentDim >
            std::numeric_limits<uint64_t>::max() -
                concatenatedDim) {
            return ge::GRAPH_FAILED;
        }
        concatenatedDim += currentDim;
        uint64_t rowElements = 0;
        if (!SafeMultiply(
                currentDim,
                innerElements,
                rowElements) ||
            !SafeMultiply(
                rowElements,
                elementBytes,
                inputRowBytes[i])) {
            return ge::GRAPH_FAILED;
        }
    }

    uint64_t outRowElements = 0;
    uint64_t outRowBytes = 0;
    if (!SafeMultiply(
            concatenatedDim,
            innerElements,
            outRowElements) ||
        !SafeMultiply(
            outRowElements,
            elementBytes,
            outRowBytes)) {
        return ge::GRAPH_FAILED;
    }
    uint64_t totalBytes = 0;
    if (!SafeMultiply(outer64, outRowBytes, totalBytes)) {
        return ge::GRAPH_FAILED;
    }

    constexpr uint32_t MAX_ROW_BLOCK_DIM = 16;
    uint32_t rowBlockDim = outer64 == 0
        ? 1
        : static_cast<uint32_t>(
            outer64 < MAX_ROW_BLOCK_DIM
                ? outer64
                : MAX_ROW_BLOCK_DIM);
    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores == 0) {
        availableCores = MAX_ROW_BLOCK_DIM;
    }
    constexpr uint64_t TARGET_BYTES_PER_CORE = 64 * 1024;
    uint64_t desiredCores64 =
        totalBytes / TARGET_BYTES_PER_CORE +
        (totalBytes % TARGET_BYTES_PER_CORE != 0 ? 1U : 0U);
    if (desiredCores64 == 0) {
        desiredCores64 = 1;
    }
    if (desiredCores64 > availableCores) {
        desiredCores64 = availableCores;
    }
    const uint32_t desiredCores =
        static_cast<uint32_t>(desiredCores64);
    constexpr uint32_t MIN_DYNAMIC_ROW_CORES = 24;
    if (outer64 >= MAX_ROW_BLOCK_DIM &&
        desiredCores >= MIN_DYNAMIC_ROW_CORES &&
        desiredCores > rowBlockDim) {
        rowBlockDim = desiredCores;
    }
    const uint32_t copyMode =
        outer64 < MAX_ROW_BLOCK_DIM &&
        desiredCores > rowBlockDim
            ? 1U
            : 0U;
    const uint32_t blockDim =
        copyMode == 1 ? desiredCores : rowBlockDim;
    const uint64_t totalWorkBlocks =
        totalBytes / 32U +
        (totalBytes % 32U != 0 ? 1U : 0U);

    ConcatTilingData tiling;
    tiling.set_outer(outer64);
    tiling.set_outRowBytes(outRowBytes);
    tiling.set_innerElements(innerElements);
    tiling.set_inputCount(inputCount);
    tiling.set_rank(static_cast<uint32_t>(rank));
    tiling.set_dim(dim);
    tiling.set_elementBytes(elementBytes);
    tiling.set_baseRowsPerBlock(outer64 / blockDim);
    tiling.set_extraBlocks(static_cast<uint32_t>(
        outer64 % blockDim));
    tiling.set_copyMode(copyMode);
    tiling.set_baseWorkBlocks(totalWorkBlocks / blockDim);
    tiling.set_extraWorkBlocks(static_cast<uint32_t>(
        totalWorkBlocks % blockDim));
    tiling.set_inputRowBytes(inputRowBytes);

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
static bool IsSpecialEmpty(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == 0;
}

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const size_t inputCount = context->GetComputeNodeInputNum();
    if (inputCount == 0) {
        return ge::GRAPH_FAILED;
    }
    size_t referenceInput = 0;
    for (size_t i = 0; i < inputCount; ++i) {
        const gert::Shape* candidate =
            context->GetDynamicInputShape(0, i);
        if (candidate == nullptr) {
            return ge::GRAPH_FAILED;
        }
        if (!IsSpecialEmpty(*candidate)) {
            referenceInput = i;
            break;
        }
    }
    const gert::Shape* firstShape =
        context->GetDynamicInputShape(0, referenceInput);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (firstShape == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const size_t rank = firstShape->GetDimNum();
    constexpr size_t MAX_RANK = 8;
    if (rank == 0 || rank > MAX_RANK) {
        return ge::GRAPH_FAILED;
    }
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr || attrs->GetInt(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t rawDim = *attrs->GetInt(0);
    const int64_t normalizedDim =
        rawDim < 0 ? rawDim + static_cast<int64_t>(rank) : rawDim;
    if (normalizedDim < 0 ||
        normalizedDim >= static_cast<int64_t>(rank)) {
        return ge::GRAPH_FAILED;
    }
    const size_t dim = static_cast<size_t>(normalizedDim);

    *outputShape = *firstShape;
    int64_t total = 0;
    for (size_t i = 0; i < inputCount; ++i) {
        const gert::Shape* currentShape =
            context->GetDynamicInputShape(0, i);
        if (currentShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        if (IsSpecialEmpty(*currentShape)) {
            continue;
        }
        if (currentShape->GetDimNum() != rank) {
            return ge::GRAPH_FAILED;
        }
        for (size_t axis = 0; axis < rank; ++axis) {
            if (axis != dim &&
                currentShape->GetDim(axis) != firstShape->GetDim(axis)) {
                return ge::GRAPH_FAILED;
            }
        }
        const int64_t extent = currentShape->GetDim(dim);
        if (extent < 0 ||
            total > std::numeric_limits<int64_t>::max() - extent) {
            return ge::GRAPH_FAILED;
        }
        total += extent;
    }
    outputShape->SetDim(dim, total);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Concat : public OpDef {
public:
    explicit Concat(const char* name) : OpDef(name)
    {
        this->Input("inputs")
            .ParamType(DYNAMIC)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_FLOAT16,
                ge::DT_INT32,
                ge::DT_INT8})
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND})
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_FLOAT16,
                ge::DT_INT32,
                ge::DT_INT8})
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND})
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Concat);
}
