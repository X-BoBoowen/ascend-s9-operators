#include <cstdint>
#include <limits>
#include <vector>

#include "greater_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr size_t MAX_RANK = 8;

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

ge::graphStatus BuildBroadcastMetadata(
    const gert::Shape& selfShape,
    const gert::Shape& otherShape,
    gert::Shape* outputShape,
    uint64_t outputDims[MAX_RANK],
    uint64_t selfStrides[MAX_RANK],
    uint64_t otherStrides[MAX_RANK],
    uint32_t& rank,
    uint64_t& outputElements,
    uint32_t& selfContiguous,
    uint32_t& otherContiguous)
{
    const size_t selfRank = selfShape.GetDimNum();
    const size_t otherRank = otherShape.GetDimNum();
    rank = static_cast<uint32_t>(
        selfRank > otherRank ? selfRank : otherRank);
    if (rank > MAX_RANK) {
        return ge::GRAPH_FAILED;
    }

    uint64_t selfDims[MAX_RANK] = {};
    uint64_t otherDims[MAX_RANK] = {};
    for (size_t axis = 0; axis < rank; ++axis) {
        const int64_t selfAxis =
            static_cast<int64_t>(axis) -
            static_cast<int64_t>(rank - selfRank);
        const int64_t otherAxis =
            static_cast<int64_t>(axis) -
            static_cast<int64_t>(rank - otherRank);
        const int64_t selfExtent =
            selfAxis < 0 ? 1 : selfShape.GetDim(selfAxis);
        const int64_t otherExtent =
            otherAxis < 0 ? 1 : otherShape.GetDim(otherAxis);
        if (selfExtent < 0 || otherExtent < 0 ||
            (selfExtent != otherExtent &&
             selfExtent != 1 &&
             otherExtent != 1)) {
            return ge::GRAPH_FAILED;
        }
        selfDims[axis] = static_cast<uint64_t>(selfExtent);
        otherDims[axis] = static_cast<uint64_t>(otherExtent);
        outputDims[axis] =
            selfDims[axis] > otherDims[axis]
                ? selfDims[axis]
                : otherDims[axis];
    }

    uint64_t selfRunningStride = 1;
    uint64_t otherRunningStride = 1;
    for (int32_t axis = static_cast<int32_t>(rank) - 1;
         axis >= 0;
         --axis) {
        selfStrides[axis] =
            selfDims[axis] == 1 && outputDims[axis] != 1
                ? 0
                : selfRunningStride;
        otherStrides[axis] =
            otherDims[axis] == 1 && outputDims[axis] != 1
                ? 0
                : otherRunningStride;
        if (!SafeMultiply(
                selfRunningStride,
                selfDims[axis],
                selfRunningStride) ||
            !SafeMultiply(
                otherRunningStride,
                otherDims[axis],
                otherRunningStride)) {
            return ge::GRAPH_FAILED;
        }
    }

    outputElements = 1;
    for (size_t axis = 0; axis < rank; ++axis) {
        if (!SafeMultiply(
                outputElements,
                outputDims[axis],
                outputElements)) {
            return ge::GRAPH_FAILED;
        }
    }

    selfContiguous = selfRank == rank ? 1U : 0U;
    otherContiguous = otherRank == rank ? 1U : 0U;
    for (size_t axis = 0; axis < rank; ++axis) {
        if (selfDims[axis] != outputDims[axis]) {
            selfContiguous = 0;
        }
        if (otherDims[axis] != outputDims[axis]) {
            otherContiguous = 0;
        }
    }

    if (outputShape != nullptr) {
        outputShape->SetDimNum(rank);
        for (size_t axis = 0; axis < rank; ++axis) {
            outputShape->SetDim(
                axis,
                static_cast<int64_t>(outputDims[axis]));
        }
    }
    return ge::GRAPH_SUCCESS;
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
    const gert::StorageShape* otherStorage =
        context->GetInputShape(1);
    if (selfStorage == nullptr || otherStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t outputDims[MAX_RANK] = {};
    uint64_t selfStrides[MAX_RANK] = {};
    uint64_t otherStrides[MAX_RANK] = {};
    uint32_t rank = 0;
    uint64_t outputElements = 0;
    uint32_t selfContiguous = 0;
    uint32_t otherContiguous = 0;
    if (BuildBroadcastMetadata(
            selfStorage->GetStorageShape(),
            otherStorage->GetStorageShape(),
            nullptr,
            outputDims,
            selfStrides,
            otherStrides,
            rank,
            outputElements,
            selfContiguous,
            otherContiguous) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint64_t partitionUnitElements = 1;
    const auto updatePartitionUnit =
        [&](const uint64_t strides[MAX_RANK],
            const uint32_t contiguous) {
            if (contiguous != 0 || rank == 0) {
                return;
            }
            bool scalar = true;
            for (uint32_t axis = 0; axis < rank; ++axis) {
                scalar = scalar && strides[axis] == 0;
            }
            if (scalar) {
                return;
            }
            uint64_t runElements = 1;
            uint64_t expectedStride = 1;
            for (int32_t axis = static_cast<int32_t>(rank) - 1;
                 axis >= 0;
                 --axis) {
                if (strides[axis] != expectedStride) {
                    break;
                }
                runElements *= outputDims[axis];
                expectedStride *= outputDims[axis];
            }
            uint64_t constantRunElements = 1;
            for (int32_t axis = static_cast<int32_t>(rank) - 1;
                 axis >= 0;
                 --axis) {
                if (strides[axis] != 0) {
                    break;
                }
                constantRunElements *= outputDims[axis];
            }
            if (constantRunElements > runElements) {
                runElements = constantRunElements;
            }
            if (runElements > partitionUnitElements) {
                partitionUnitElements = runElements;
            }
        };
    updatePartitionUnit(selfStrides, selfContiguous);
    updatePartitionUnit(otherStrides, otherContiguous);

    constexpr uint64_t ELEMENTS_PER_CORE_TARGET = 4096;
    constexpr uint32_t MAX_BLOCK_DIM = 40;
    uint64_t desiredBlocks =
        (outputElements + ELEMENTS_PER_CORE_TARGET - 1) /
        ELEMENTS_PER_CORE_TARGET;
    if (desiredBlocks == 0) {
        desiredBlocks = 1;
    }
    const uint64_t partitionUnits =
        outputElements / partitionUnitElements;
    if (partitionUnits != 0 && desiredBlocks > partitionUnits) {
        desiredBlocks = partitionUnits;
    }
    const uint32_t blockDim = static_cast<uint32_t>(
        desiredBlocks < MAX_BLOCK_DIM
            ? desiredBlocks
            : MAX_BLOCK_DIM);

    GreaterTilingData tiling;
    tiling.set_outputElements(outputElements);
    const uint64_t baseUnits = partitionUnits / blockDim;
    tiling.set_baseElementsPerBlock(
        baseUnits * partitionUnitElements);
    tiling.set_partitionUnitElements(partitionUnitElements);
    tiling.set_extraBlocks(static_cast<uint32_t>(
        partitionUnits % blockDim));
    tiling.set_rank(rank);
    tiling.set_selfContiguous(selfContiguous);
    tiling.set_otherContiguous(otherContiguous);
    tiling.set_outputDims(outputDims);
    tiling.set_selfStrides(selfStrides);
    tiling.set_otherStrides(otherStrides);

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
    const gert::Shape* selfShape = context->GetInputShape(0);
    const gert::Shape* otherShape = context->GetInputShape(1);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (selfShape == nullptr ||
        otherShape == nullptr ||
        outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t outputDims[MAX_RANK] = {};
    uint64_t selfStrides[MAX_RANK] = {};
    uint64_t otherStrides[MAX_RANK] = {};
    uint32_t rank = 0;
    uint64_t outputElements = 0;
    uint32_t selfContiguous = 0;
    uint32_t otherContiguous = 0;
    return BuildBroadcastMetadata(
        *selfShape,
        *otherShape,
        outputShape,
        outputDims,
        selfStrides,
        otherStrides,
        rank,
        outputElements,
        selfContiguous,
        otherContiguous);
}
}

namespace ops {
class Greater : public OpDef {
public:
    explicit Greater(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> inputTypes = {
            ge::DT_FLOAT,
            ge::DT_BF16,
            ge::DT_FLOAT16,
            ge::DT_INT32,
            ge::DT_INT8};
        const std::vector<ge::Format> formats(
            inputTypes.size(),
            ge::FORMAT_ND);
        const std::vector<ge::DataType> outputTypes(
            inputTypes.size(),
            ge::DT_BOOL);

        this->Input("self")
            .ParamType(REQUIRED)
            .DataType(inputTypes)
            .Format(formats)
            .UnknownShapeFormat(formats);
        this->Input("other")
            .ParamType(REQUIRED)
            .DataType(inputTypes)
            .Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType(outputTypes)
            .Format(formats)
            .UnknownShapeFormat(formats);

        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Greater);
}
