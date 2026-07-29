#include <cstdint>
#include <limits>

#include "concat_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t MAX_INPUT_COUNT = 2048;
constexpr size_t MAX_RANK = 8;
constexpr uint32_t ALIGN_BYTES = 32;

// One core needs enough bytes for a DMA burst to amortise kernel launch and
// tail handling. 4 KB reproduces the 16 cores that the earlier 8/16/32-core
// sweep picked for the 64 KB public shape, while replacing the 64 KB budget
// that used to pin everything below 1.5 MB to 16 cores regardless of size.
constexpr uint64_t MIN_BYTES_PER_CORE = 4 * 1024;
constexpr uint32_t FALLBACK_CORES = 40;

// Flat mode granularity: prefer 512 B units so each core issues few large
// transfers, falling back to 32 B when the payload is too small to split.
constexpr uint32_t WIDE_WORK_UNIT_BYTES = 512;

constexpr uint32_t MODE_ROW_BLOCK = 0;
constexpr uint32_t MODE_ROW_GENERIC = 1;
constexpr uint32_t MODE_FLAT = 2;

bool IsSpecialEmpty(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == 0;
}

uint32_t GetElementBytes(const ge::DataType dataType)
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

bool SafeMultiply(
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

uint64_t CeilDiv(const uint64_t value, const uint64_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t inputCount = static_cast<uint32_t>(
        context->GetComputeNodeInputNum());
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

    uint64_t outer = 1;
    for (size_t i = 0; i < dim; ++i) {
        const int64_t extent = shape.GetDim(i);
        if (extent < 0 ||
            !SafeMultiply(
                outer,
                static_cast<uint64_t>(extent),
                outer)) {
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
    uint64_t innerBytes = 0;
    if (!SafeMultiply(innerElements, elementBytes, innerBytes)) {
        return ge::GRAPH_FAILED;
    }

    uint64_t concatenatedDim = 0;
    uint32_t dimExtents[MAX_INPUT_COUNT] = {};
    bool allRowsAligned = true;
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
        const gert::Shape& current = currentShape->GetStorageShape();
        if (IsSpecialEmpty(current)) {
            dimExtents[i] = 0;
            continue;
        }
        if (current.GetDimNum() != rank) {
            return ge::GRAPH_FAILED;
        }
        for (size_t axis = 0; axis < rank; ++axis) {
            const int64_t extent = current.GetDim(axis);
            if (extent < 0 ||
                (axis != dim && extent != shape.GetDim(axis))) {
                return ge::GRAPH_FAILED;
            }
        }
        const int64_t rawExtent = current.GetDim(dim);
        if (rawExtent > static_cast<int64_t>(
                std::numeric_limits<uint32_t>::max())) {
            return ge::GRAPH_FAILED;
        }
        const uint64_t currentDim = static_cast<uint64_t>(rawExtent);
        if (currentDim >
            std::numeric_limits<uint64_t>::max() - concatenatedDim) {
            return ge::GRAPH_FAILED;
        }
        concatenatedDim += currentDim;
        dimExtents[i] = static_cast<uint32_t>(currentDim);

        uint64_t rowBytes = 0;
        if (!SafeMultiply(currentDim, innerBytes, rowBytes)) {
            return ge::GRAPH_FAILED;
        }
        if (rowBytes % ALIGN_BYTES != 0) {
            allRowsAligned = false;
        }
    }

    uint64_t outRowBytes = 0;
    if (!SafeMultiply(concatenatedDim, innerBytes, outRowBytes)) {
        return ge::GRAPH_FAILED;
    }
    uint64_t totalBytes = 0;
    if (!SafeMultiply(outer, outRowBytes, totalBytes)) {
        return ge::GRAPH_FAILED;
    }
    if (outRowBytes % ALIGN_BYTES != 0) {
        allRowsAligned = false;
    }

    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores == 0) {
        availableCores = FALLBACK_CORES;
    }

    // Single byte-based core budget for every mode. The previous revision
    // combined a 16-core row cap with a 24-core promotion threshold, which
    // left everything between 1 MB and 1.5 MB stranded on 16 cores.
    uint64_t byteCores = CeilDiv(totalBytes, MIN_BYTES_PER_CORE);
    if (byteCores == 0) {
        byteCores = 1;
    }
    if (byteCores > availableCores) {
        byteCores = availableCores;
    }

    uint32_t mode = MODE_FLAT;
    uint32_t blockDim = static_cast<uint32_t>(byteCores);
    uint64_t rowCores = outer < byteCores ? outer : byteCores;
    if (rowCores >= 1 && outer != 0) {
        // Row splitting keeps each core's output span contiguous, so it can
        // stage whole rows in UB and write them back with aligned bursts.
        // Only take it when it does not starve cores relative to flat mode.
        const bool rowsCoverCores = rowCores * 2 >= byteCores;
        if (rowsCoverCores) {
            mode = allRowsAligned ? MODE_ROW_BLOCK : MODE_ROW_GENERIC;
            blockDim = static_cast<uint32_t>(rowCores);
        }
    }

    uint32_t workUnitBytes = WIDE_WORK_UNIT_BYTES;
    uint64_t totalWorkBlocks = 0;
    if (mode == MODE_FLAT) {
        if (CeilDiv(totalBytes, WIDE_WORK_UNIT_BYTES) <
            static_cast<uint64_t>(blockDim)) {
            workUnitBytes = ALIGN_BYTES;
        }
        totalWorkBlocks = CeilDiv(totalBytes, workUnitBytes);
        if (totalWorkBlocks == 0) {
            totalWorkBlocks = 1;
        }
        if (totalWorkBlocks < blockDim) {
            blockDim = static_cast<uint32_t>(totalWorkBlocks);
        }
    }
    if (blockDim == 0) {
        blockDim = 1;
    }

    ConcatTilingData tiling;
    tiling.set_outer(outer);
    tiling.set_outRowBytes(outRowBytes);
    tiling.set_innerBytes(innerBytes);
    tiling.set_totalBytes(totalBytes);
    tiling.set_baseRowsPerBlock(
        mode == MODE_FLAT ? 0 : outer / blockDim);
    tiling.set_baseWorkBlocks(
        mode == MODE_FLAT ? totalWorkBlocks / blockDim : 0);
    tiling.set_inputCount(inputCount);
    tiling.set_mode(mode);
    tiling.set_elementBytes(elementBytes);
    tiling.set_rank(static_cast<uint32_t>(rank));
    tiling.set_dim(dim);
    tiling.set_extraBlocks(static_cast<uint32_t>(
        mode == MODE_FLAT ? 0 : outer % blockDim));
    tiling.set_extraWorkBlocks(static_cast<uint32_t>(
        mode == MODE_FLAT ? totalWorkBlocks % blockDim : 0));
    tiling.set_workUnitBytes(workUnitBytes);
    tiling.set_dimExtents(dimExtents);

    // Mode is dispatched at runtime from tiling data rather than through a
    // tiling key, so only one kernel binary has to be built.
    context->SetBlockDim(blockDim);
    // The extent array raised this structure past 8 KB; fail loudly rather
    // than let a short buffer truncate it into garbage tiling.
    if (context->GetRawTilingData() == nullptr ||
        context->GetRawTilingData()->GetCapacity() < tiling.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
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
    constexpr size_t MAX_INFER_RANK = 8;
    if (rank == 0 || rank > MAX_INFER_RANK) {
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
