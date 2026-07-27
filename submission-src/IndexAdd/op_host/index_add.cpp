#include <cstdint>
#include <limits>

#include "index_add_tiling.h"
#include "register/op_def_registry.h"

namespace {
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
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape* selfStorage =
        context->GetInputShape(0);
    const gert::StorageShape* indexStorage =
        context->GetInputShape(1);
    const gert::StorageShape* sourceStorage =
        context->GetInputShape(2);
    const gert::Tensor* selfTensor = context->GetInputTensor(0);
    const gert::Tensor* indexTensor = context->GetInputTensor(1);
    const gert::Tensor* sourceTensor = context->GetInputTensor(2);
    if (selfStorage == nullptr ||
        indexStorage == nullptr ||
        sourceStorage == nullptr ||
        selfTensor == nullptr ||
        indexTensor == nullptr ||
        sourceTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& selfShape = selfStorage->GetStorageShape();
    const gert::Shape& indexShape = indexStorage->GetStorageShape();
    const gert::Shape& sourceShape =
        sourceStorage->GetStorageShape();
    const size_t rank = selfShape.GetDimNum();
    constexpr size_t MAX_RANK = 8;
    if (rank == 0 ||
        rank > MAX_RANK ||
        indexShape.GetDimNum() != 1 ||
        sourceShape.GetDimNum() != rank ||
        selfTensor->GetDataType() != sourceTensor->GetDataType() ||
        indexTensor->GetDataType() != ge::DT_INT32) {
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

    const int64_t rawIndexCount = indexShape.GetDim(0);
    if (rawIndexCount < 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t indexCount =
        static_cast<uint64_t>(rawIndexCount);

    uint64_t outer = 1;
    uint64_t inner = 1;
    for (size_t axis = 0; axis < rank; ++axis) {
        const int64_t selfExtent = selfShape.GetDim(axis);
        const int64_t sourceExtent = sourceShape.GetDim(axis);
        if (selfExtent < 0 ||
            sourceExtent < 0 ||
            (axis == dim
                ? static_cast<uint64_t>(sourceExtent) != indexCount
                : sourceExtent != selfExtent)) {
            return ge::GRAPH_FAILED;
        }
        if (axis < dim &&
            !SafeMultiply(
                outer,
                static_cast<uint64_t>(selfExtent),
                outer)) {
            return ge::GRAPH_FAILED;
        }
        if (axis > dim &&
            !SafeMultiply(
                inner,
                static_cast<uint64_t>(selfExtent),
                inner)) {
            return ge::GRAPH_FAILED;
        }
    }

    const uint64_t dimSize =
        static_cast<uint64_t>(selfShape.GetDim(dim));
    uint64_t totalElements = 0;
    uint64_t sourceElements = 0;
    uint64_t temporary = 0;
    if (!SafeMultiply(outer, dimSize, temporary) ||
        !SafeMultiply(temporary, inner, totalElements) ||
        !SafeMultiply(outer, indexCount, temporary) ||
        !SafeMultiply(temporary, inner, sourceElements)) {
        return ge::GRAPH_FAILED;
    }

    constexpr uint64_t INNER_CHUNK = 256;
    constexpr uint64_t MAX_DIM_GROUP = 64;
    constexpr uint32_t MAX_BLOCK_DIM = 40;
    const uint64_t innerChunks =
        inner == 0 ? 0 : (inner + INNER_CHUNK - 1) / INNER_CHUNK;
    uint64_t parallelBase = 0;
    if (!SafeMultiply(outer, innerChunks, parallelBase)) {
        return ge::GRAPH_FAILED;
    }
    uint64_t desiredDimGroups = 1;
    if (parallelBase > 0 && parallelBase < MAX_BLOCK_DIM) {
        desiredDimGroups =
            (MAX_BLOCK_DIM + parallelBase - 1) / parallelBase;
    }
    if (desiredDimGroups > dimSize) {
        desiredDimGroups = dimSize;
    }
    uint64_t dimGroup = desiredDimGroups == 0
        ? 1
        : (dimSize + desiredDimGroups - 1) / desiredDimGroups;
    if (dimGroup > MAX_DIM_GROUP) {
        dimGroup = MAX_DIM_GROUP;
    }
    const uint64_t dimGroups =
        dimSize == 0 ? 0 : (dimSize + dimGroup - 1) / dimGroup;
    uint64_t taskCount = 0;
    if (!SafeMultiply(parallelBase, dimGroups, taskCount)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t blockDim = taskCount == 0
        ? 1
        : static_cast<uint32_t>(
            taskCount < MAX_BLOCK_DIM
                ? taskCount
                : MAX_BLOCK_DIM);

    IndexAddTilingData tiling;
    tiling.set_totalElements(totalElements);
    tiling.set_sourceElements(sourceElements);
    tiling.set_outer(outer);
    tiling.set_dimSize(dimSize);
    tiling.set_inner(inner);
    tiling.set_indexCount(indexCount);
    tiling.set_taskCount(taskCount);
    tiling.set_dimGroup(static_cast<uint32_t>(dimGroup));
    tiling.set_dimGroups(static_cast<uint32_t>(dimGroups));
    tiling.set_innerChunks(static_cast<uint32_t>(innerChunks));

    const bool useBatchedInt8 =
        selfTensor->GetDataType() == ge::DT_INT8 &&
        indexCount > dimSize * 4;
    context->SetTilingKey(useBatchedInt8 ? 2 : 1);
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
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class IndexAdd : public OpDef {
public:
    explicit IndexAdd(const char* name) : OpDef(name)
    {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_BF16,
                ge::DT_FLOAT16,
                ge::DT_INT32,
                ge::DT_INT8})
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND})
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND});
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_INT32,
                ge::DT_INT32,
                ge::DT_INT32,
                ge::DT_INT32,
                ge::DT_INT32})
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND})
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND});
        this->Input("source")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_BF16,
                ge::DT_FLOAT16,
                ge::DT_INT32,
                ge::DT_INT8})
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND})
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_BF16,
                ge::DT_FLOAT16,
                ge::DT_INT32,
                ge::DT_INT8})
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND})
            .UnknownShapeFormat({
                ge::FORMAT_ND,
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

OP_ADD(IndexAdd);
}
