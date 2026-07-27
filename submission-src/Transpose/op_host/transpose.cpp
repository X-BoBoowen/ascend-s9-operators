#include <cstdint>
#include <limits>

#include "transpose_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr size_t MAX_RANK = 6;

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

ge::graphStatus BuildMetadata(
    const gert::Shape& inputShape,
    const gert::RuntimeAttrs* attrs,
    gert::Shape* outputShape,
    uint64_t outputDims[MAX_RANK],
    uint64_t inputStridesForOutput[MAX_RANK],
    uint32_t permutation[MAX_RANK],
    uint32_t& rank,
    uint32_t& identity,
    uint64_t& totalElements)
{
    rank = static_cast<uint32_t>(inputShape.GetDimNum());
    if (rank == 0 || rank > MAX_RANK || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::TypedContinuousVector<int64_t>* dimsAttr =
        attrs->GetListInt(0);
    if (dimsAttr == nullptr || dimsAttr->GetSize() != rank) {
        return ge::GRAPH_FAILED;
    }

    uint64_t inputDims[MAX_RANK] = {};
    uint64_t inputStrides[MAX_RANK] = {};
    bool used[MAX_RANK] = {};
    totalElements = 1;
    for (size_t axis = 0; axis < rank; ++axis) {
        const int64_t extent = inputShape.GetDim(axis);
        if (extent < 0 ||
            !SafeMultiply(
                totalElements,
                static_cast<uint64_t>(extent),
                totalElements)) {
            return ge::GRAPH_FAILED;
        }
        inputDims[axis] = static_cast<uint64_t>(extent);
    }

    uint64_t runningStride = 1;
    for (int32_t axis = static_cast<int32_t>(rank) - 1;
         axis >= 0;
         --axis) {
        inputStrides[axis] = runningStride;
        if (!SafeMultiply(
                runningStride,
                inputDims[axis],
                runningStride)) {
            return ge::GRAPH_FAILED;
        }
    }

    const int64_t* dims = dimsAttr->GetData();
    identity = 1;
    for (size_t outputAxis = 0;
         outputAxis < rank;
         ++outputAxis) {
        const int64_t rawAxis = dims[outputAxis];
        const int64_t normalizedAxis =
            rawAxis < 0 ? rawAxis + rank : rawAxis;
        if (normalizedAxis < 0 ||
            normalizedAxis >= rank ||
            used[normalizedAxis]) {
            return ge::GRAPH_FAILED;
        }
        used[normalizedAxis] = true;
        permutation[outputAxis] =
            static_cast<uint32_t>(normalizedAxis);
        if (normalizedAxis !=
            static_cast<int64_t>(outputAxis)) {
            identity = 0;
        }
        outputDims[outputAxis] =
            inputDims[normalizedAxis];
        inputStridesForOutput[outputAxis] =
            inputStrides[normalizedAxis];
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
    const gert::StorageShape* inputStorage =
        context->GetInputShape(0);
    const gert::Tensor* inputTensor =
        context->GetInputTensor(0);
    if (inputStorage == nullptr || inputTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t outputDims[MAX_RANK] = {};
    uint64_t inputStrides[MAX_RANK] = {};
    uint32_t permutation[MAX_RANK] = {};
    uint32_t rank = 0;
    uint32_t identity = 0;
    uint64_t totalElements = 0;
    if (BuildMetadata(
            inputStorage->GetStorageShape(),
            context->GetAttrs(),
            nullptr,
            outputDims,
            inputStrides,
            permutation,
            rank,
            identity,
            totalElements) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    constexpr uint64_t ELEMENTS_PER_CORE_TARGET = 4096;
    constexpr uint64_t GATHER_ELEMENTS = 2048;
    constexpr uint32_t MAX_BLOCK_DIM = 40;
    constexpr uint32_t FAST_2D_BLOCK_DIM = 32;
    uint64_t contiguousElements = 1;
    uint64_t expectedStride = 1;
    for (int32_t axis = static_cast<int32_t>(rank) - 1;
         axis >= 0;
         --axis) {
        if (inputStrides[axis] != expectedStride) {
            break;
        }
        contiguousElements *= outputDims[axis];
        expectedStride *= outputDims[axis];
    }
    const uint64_t totalRuns =
        contiguousElements == 0
            ? 0
            : totalElements / contiguousElements;
    uint64_t desiredBlocks =
        (totalElements + ELEMENTS_PER_CORE_TARGET - 1) /
        ELEMENTS_PER_CORE_TARGET;
    uint32_t rotationPrefix = 0;
    while (rotationPrefix < rank &&
           permutation[rotationPrefix] == rotationPrefix) {
        ++rotationPrefix;
    }
    uint32_t rotationSplit = 0;
    if (identity == 0 && totalElements != 0) {
        const uint32_t suffixRank = rank - rotationPrefix;
        for (uint32_t relativeSplit = 1;
             relativeSplit < suffixRank;
             ++relativeSplit) {
            bool matches = true;
            for (uint32_t outputAxis = rotationPrefix;
                 outputAxis < rank;
                 ++outputAxis) {
                const uint32_t relativeAxis =
                    outputAxis - rotationPrefix;
                if (permutation[outputAxis] !=
                    rotationPrefix +
                        (relativeAxis + relativeSplit) %
                            suffixRank) {
                    matches = false;
                    break;
                }
            }
            if (matches) {
                rotationSplit =
                    rotationPrefix + relativeSplit;
                break;
            }
        }
    }
    uint64_t matrixBatches64 = 1;
    uint64_t matrixRows64 = 1;
    uint64_t matrixCols64 = 1;
    if (rotationSplit != 0) {
        const gert::Shape& inputShape =
            inputStorage->GetStorageShape();
        for (uint32_t axis = 0;
             axis < rotationPrefix;
             ++axis) {
            matrixBatches64 *= static_cast<uint64_t>(
                inputShape.GetDim(axis));
        }
        for (uint32_t axis = rotationPrefix;
             axis < rotationSplit;
             ++axis) {
            matrixRows64 *= static_cast<uint64_t>(
                inputShape.GetDim(axis));
        }
        matrixCols64 =
            totalElements / matrixBatches64 / matrixRows64;
    }
    const bool rotation2D =
        rotationSplit != 0 &&
        matrixRows64 <= std::numeric_limits<uint32_t>::max() &&
        matrixCols64 <= std::numeric_limits<uint32_t>::max();
    const uint32_t rows =
        rotation2D ? static_cast<uint32_t>(matrixRows64) : 0;
    const uint32_t cols =
        rotation2D ? static_cast<uint32_t>(matrixCols64) : 0;
    const bool fast2D =
        inputTensor->GetDataType() == ge::DT_FLOAT16 &&
        rotation2D &&
        rows % 16U == 0 &&
        cols % 16U == 0 &&
        rows / 16U - 1U <=
            std::numeric_limits<uint16_t>::max() &&
        cols / 16U - 1U <=
            std::numeric_limits<uint16_t>::max();
    uint32_t elementBytes = 0;
    switch (inputTensor->GetDataType()) {
        case ge::DT_INT8:
            elementBytes = 1;
            break;
        case ge::DT_FLOAT16:
            elementBytes = 2;
            break;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            elementBytes = 4;
            break;
        default:
            return ge::GRAPH_FAILED;
    }
    const uint32_t matrixTile =
        fast2D ? 16U : 32U / elementBytes;
    const uint32_t tileCols =
        rotation2D ? (cols + matrixTile - 1U) / matrixTile : 0;
    const uint64_t tileRows64 =
        rotation2D ? (rows + matrixTile - 1U) / matrixTile : 0;
    const uint64_t totalTiles64 =
        matrixBatches64 *
        tileRows64 *
        static_cast<uint64_t>(tileCols);
    const bool rotationFast =
        rotation2D &&
        totalTiles64 <= std::numeric_limits<uint32_t>::max();
    const uint32_t totalTiles =
        rotationFast ? static_cast<uint32_t>(totalTiles64) : 0;
    const uint64_t gatherLastDim64 =
        outputDims[rank - 1U];
    const uint64_t gatherInputStride64 =
        inputStrides[rank - 1U];
    const bool gatherFast =
        !rotationFast &&
        identity == 0 &&
        contiguousElements == 1 &&
        gatherLastDim64 != 0 &&
        gatherLastDim64 <= std::numeric_limits<uint32_t>::max() &&
        gatherInputStride64 > 1 &&
        gatherInputStride64 - 1U <=
            std::numeric_limits<uint32_t>::max() /
                elementBytes;
    const uint64_t gatherChunksPerRun64 =
        gatherFast
            ? (gatherLastDim64 + GATHER_ELEMENTS - 1U) /
                GATHER_ELEMENTS
            : 0;
    const uint64_t gatherRuns =
        gatherFast ? totalElements / gatherLastDim64 : 0;
    const uint64_t totalGatherTasks =
        gatherRuns * gatherChunksPerRun64;
    if (rotationFast) {
        desiredBlocks = totalTiles;
    } else if (gatherFast) {
        desiredBlocks = totalGatherTasks;
    } else if (identity == 0 &&
               contiguousElements > 1) {
        desiredBlocks = totalRuns;
    }
    if (desiredBlocks == 0) {
        desiredBlocks = 1;
    }
    const uint32_t blockLimit =
        rotationFast ? FAST_2D_BLOCK_DIM : MAX_BLOCK_DIM;
    const uint32_t blockDim = static_cast<uint32_t>(
        desiredBlocks < blockLimit
            ? desiredBlocks
            : blockLimit);

    TransposeTilingData tiling;
    tiling.set_totalElements(totalElements);
    tiling.set_baseElementsPerBlock(totalElements / blockDim);
    tiling.set_extraBlocks(static_cast<uint32_t>(
        totalElements % blockDim));
    tiling.set_rank(rank);
    tiling.set_identity(identity);
    tiling.set_fast2D(fast2D ? 1U : 0U);
    tiling.set_gatherLastDim(
        gatherFast ? static_cast<uint32_t>(gatherLastDim64) : 0);
    tiling.set_gatherInputStride(
        gatherFast
            ? static_cast<uint32_t>(gatherInputStride64)
            : 0);
    tiling.set_gatherChunksPerRun(
        gatherFast
            ? static_cast<uint32_t>(gatherChunksPerRun64)
            : 0);
    tiling.set_rows(rows);
    tiling.set_cols(cols);
    tiling.set_tileCols(tileCols);
    tiling.set_baseTilesPerBlock(
        rotationFast ? totalTiles / blockDim : 0);
    tiling.set_extraTileBlocks(
        rotationFast ? totalTiles % blockDim : 0);
    tiling.set_contiguousElements(contiguousElements);
    tiling.set_baseRunsPerBlock(
        (gatherFast ? totalGatherTasks : totalRuns) /
        blockDim);
    tiling.set_extraRunBlocks(static_cast<uint32_t>(
        (gatherFast ? totalGatherTasks : totalRuns) %
        blockDim));
    tiling.set_outputDims(outputDims);
    tiling.set_inputStrides(inputStrides);

    context->SetBlockDim(blockDim);
    context->SetTilingKey(
        fast2D
            ? (matrixBatches64 == 1 ? 2 : 5)
            : (rotationFast ? 3 : (gatherFast ? 4 : 1)));
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

    uint64_t outputDims[MAX_RANK] = {};
    uint64_t inputStrides[MAX_RANK] = {};
    uint32_t permutation[MAX_RANK] = {};
    uint32_t rank = 0;
    uint32_t identity = 0;
    uint64_t totalElements = 0;
    return BuildMetadata(
        *inputShape,
        context->GetAttrs(),
        outputShape,
        outputDims,
        inputStrides,
        permutation,
        rank,
        identity,
        totalElements);
}
}

namespace ops {
class Transpose : public OpDef {
public:
    explicit Transpose(const char* name) : OpDef(name)
    {
        this->Input("inputs")
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
        this->Attr("dims").AttrType(REQUIRED).ListInt();

        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Transpose);
}
