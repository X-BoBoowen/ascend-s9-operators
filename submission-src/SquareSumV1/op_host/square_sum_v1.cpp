#include <cstdint>
#include <limits>

#include "square_sum_v1_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr size_t MAX_RANK = 5;

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
    uint64_t outputInputStrides[MAX_RANK],
    uint64_t reduceDims[MAX_RANK],
    uint64_t reduceInputStrides[MAX_RANK],
    uint32_t& outputRank,
    uint32_t& reduceRank,
    uint32_t& fastPath,
    uint64_t& innerElements,
    uint64_t& trailingReduceElements,
    uint64_t& inputElements,
    uint64_t& outputElements,
    uint64_t& reduceElements)
{
    const uint32_t rank =
        static_cast<uint32_t>(inputShape.GetDimNum());
    if (rank == 0 || rank > MAX_RANK || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::TypedContinuousVector<int64_t>* axisAttr =
        attrs->GetListInt(0);
    if (axisAttr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const bool* keepDimsAttr = attrs->GetBool(1);
    const bool keepDims =
        keepDimsAttr == nullptr ? false : *keepDimsAttr;

    uint64_t inputDims[MAX_RANK] = {};
    uint64_t inputStrides[MAX_RANK] = {};
    bool reduced[MAX_RANK] = {};
    inputElements = 1;
    for (uint32_t axis = 0; axis < rank; ++axis) {
        const int64_t extent = inputShape.GetDim(axis);
        if (extent < 0 ||
            !SafeMultiply(
                inputElements,
                static_cast<uint64_t>(extent),
                inputElements)) {
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

    const size_t axisCount = axisAttr->GetSize();
    if (axisCount == 0) {
        for (uint32_t axis = 0; axis < rank; ++axis) {
            reduced[axis] = true;
        }
    } else {
        const int64_t* axes = axisAttr->GetData();
        for (size_t index = 0; index < axisCount; ++index) {
            const int64_t rawAxis = axes[index];
            const int64_t normalizedAxis =
                rawAxis < 0 ? rawAxis + rank : rawAxis;
            if (normalizedAxis < 0 ||
                normalizedAxis >= rank ||
                reduced[normalizedAxis]) {
                return ge::GRAPH_FAILED;
            }
            reduced[normalizedAxis] = true;
        }
    }

    outputRank = 0;
    reduceRank = 0;
    fastPath = 0;
    innerElements = 1;
    trailingReduceElements = 1;
    outputElements = 1;
    reduceElements = 1;
    for (uint32_t axis = 0; axis < rank; ++axis) {
        if (reduced[axis]) {
            reduceDims[reduceRank] = inputDims[axis];
            reduceInputStrides[reduceRank] =
                inputStrides[axis];
            if (!SafeMultiply(
                    reduceElements,
                    inputDims[axis],
                    reduceElements)) {
                return ge::GRAPH_FAILED;
            }
            ++reduceRank;
            if (keepDims) {
                outputDims[outputRank] = 1;
                outputInputStrides[outputRank] = 0;
                ++outputRank;
            }
        } else {
            outputDims[outputRank] = inputDims[axis];
            outputInputStrides[outputRank] =
                inputStrides[axis];
            if (!SafeMultiply(
                    outputElements,
                    inputDims[axis],
                    outputElements)) {
                return ge::GRAPH_FAILED;
            }
            ++outputRank;
        }
    }

    uint32_t firstReduced = rank;
    uint32_t lastReduced = 0;
    for (uint32_t axis = 0; axis < rank; ++axis) {
        if (reduced[axis]) {
            firstReduced = axis;
            break;
        }
    }
    if (firstReduced < rank) {
        for (int32_t axis = static_cast<int32_t>(rank) - 1;
             axis >= 0;
             --axis) {
            if (reduced[axis]) {
                lastReduced = static_cast<uint32_t>(axis);
                break;
            }
        }
        bool contiguousGroup = true;
        for (uint32_t axis = firstReduced;
             axis <= lastReduced;
             ++axis) {
            if (!reduced[axis]) {
                contiguousGroup = false;
                break;
            }
        }
        if (contiguousGroup) {
            fastPath =
                lastReduced == rank - 1 ? 1U : 2U;
            for (uint32_t axis = lastReduced + 1;
                 axis < rank;
                 ++axis) {
                if (!SafeMultiply(
                        innerElements,
                        inputDims[axis],
                        innerElements)) {
                    return ge::GRAPH_FAILED;
                }
            }
        } else if (reduced[rank - 1]) {
            fastPath = 3;
            for (int32_t axis =
                     static_cast<int32_t>(rank) - 1;
                 axis >= 0 && reduced[axis];
                 --axis) {
                if (!SafeMultiply(
                        trailingReduceElements,
                        inputDims[axis],
                        trailingReduceElements)) {
                    return ge::GRAPH_FAILED;
                }
            }
        } else {
            fastPath = 4;
            for (int32_t axis =
                     static_cast<int32_t>(rank) - 1;
                 axis >= 0 && !reduced[axis];
                 --axis) {
                if (!SafeMultiply(
                        innerElements,
                        inputDims[axis],
                        innerElements)) {
                    return ge::GRAPH_FAILED;
                }
            }
        }
    }

    if (outputShape != nullptr) {
        outputShape->SetDimNum(outputRank);
        for (uint32_t axis = 0; axis < outputRank; ++axis) {
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
    if (inputStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t outputDims[MAX_RANK] = {};
    uint64_t outputInputStrides[MAX_RANK] = {};
    uint64_t reduceDims[MAX_RANK] = {};
    uint64_t reduceInputStrides[MAX_RANK] = {};
    uint32_t outputRank = 0;
    uint32_t reduceRank = 0;
    uint32_t fastPath = 0;
    uint64_t innerElements = 1;
    uint64_t trailingReduceElements = 1;
    uint64_t inputElements = 0;
    uint64_t outputElements = 0;
    uint64_t reduceElements = 0;
    if (BuildMetadata(
            inputStorage->GetStorageShape(),
            context->GetAttrs(),
            nullptr,
            outputDims,
            outputInputStrides,
            reduceDims,
            reduceInputStrides,
            outputRank,
            reduceRank,
            fastPath,
            innerElements,
            trailingReduceElements,
            inputElements,
            outputElements,
            reduceElements) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    constexpr uint64_t OUTPUTS_PER_CORE_TARGET = 64;
    constexpr uint32_t MAX_BLOCK_DIM = 40;
    constexpr uint32_t SMALL_FAST_BLOCK_DIM = 32;
    constexpr uint64_t FULL_CORE_INPUT_THRESHOLD = 1U << 20U;
    uint64_t desiredBlocks = 0;
    if (fastPath == 1 || fastPath == 3) {
        const uint32_t fastBlockDim =
            inputElements >= FULL_CORE_INPUT_THRESHOLD
                ? MAX_BLOCK_DIM
                : SMALL_FAST_BLOCK_DIM;
        desiredBlocks =
            outputElements < fastBlockDim
                ? outputElements
                : fastBlockDim;
    } else {
        desiredBlocks =
            (outputElements + OUTPUTS_PER_CORE_TARGET - 1) /
            OUTPUTS_PER_CORE_TARGET;
    }
    if (desiredBlocks == 0) {
        desiredBlocks = 1;
    }
    const uint32_t blockDim = static_cast<uint32_t>(
        desiredBlocks < MAX_BLOCK_DIM
            ? desiredBlocks
            : MAX_BLOCK_DIM);

    SquareSumV1TilingData tiling;
    tiling.set_inputElements(inputElements);
    tiling.set_outputElements(outputElements);
    tiling.set_reduceElements(reduceElements);
    tiling.set_innerElements(innerElements);
    tiling.set_trailingReduceElements(
        trailingReduceElements);
    tiling.set_baseOutputsPerBlock(
        outputElements / blockDim);
    tiling.set_extraBlocks(static_cast<uint32_t>(
        outputElements % blockDim));
    tiling.set_outputRank(outputRank);
    tiling.set_reduceRank(reduceRank);
    tiling.set_fastPath(fastPath);
    tiling.set_outputDims(outputDims);
    tiling.set_outputInputStrides(outputInputStrides);
    tiling.set_reduceDims(reduceDims);
    tiling.set_reduceInputStrides(reduceInputStrides);

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

    uint64_t outputDims[MAX_RANK] = {};
    uint64_t outputInputStrides[MAX_RANK] = {};
    uint64_t reduceDims[MAX_RANK] = {};
    uint64_t reduceInputStrides[MAX_RANK] = {};
    uint32_t outputRank = 0;
    uint32_t reduceRank = 0;
    uint32_t fastPath = 0;
    uint64_t innerElements = 1;
    uint64_t trailingReduceElements = 1;
    uint64_t inputElements = 0;
    uint64_t outputElements = 0;
    uint64_t reduceElements = 0;
    return BuildMetadata(
        *inputShape,
        context->GetAttrs(),
        outputShape,
        outputDims,
        outputInputStrides,
        reduceDims,
        reduceInputStrides,
        outputRank,
        reduceRank,
        fastPath,
        innerElements,
        trailingReduceElements,
        inputElements,
        outputElements,
        reduceElements);
}
}

namespace ops {
class SquareSumV1 : public OpDef {
public:
    explicit SquareSumV1(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT})
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND})
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT})
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND})
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND});
        this->Attr("axis").AttrType(REQUIRED).ListInt();
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SquareSumV1);
}
