#include <cstdint>
#include <limits>

#include "square_sum_v1_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

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

bool SafeAdd(
    const uint64_t left,
    const uint64_t right,
    uint64_t& result)
{
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        return false;
    }
    result = left + right;
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
        bool hasSingletonGap = false;
        for (uint32_t axis = firstReduced;
             axis <= lastReduced;
             ++axis) {
            if (!reduced[axis] && inputDims[axis] != 1U) {
                contiguousGroup = false;
                break;
            }
            if (!reduced[axis]) {
                hasSingletonGap = true;
            }
        }
        if (contiguousGroup &&
            hasSingletonGap &&
            lastReduced != rank - 1U &&
            reduceElements < 8192U) {
            contiguousGroup = false;
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
            if (fastPath == 2U &&
                innerElements == 1U &&
                reduceElements > 0U) {
                fastPath = 1U;
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
    const gert::CompileTimeTensorDesc* inputDesc =
        context->GetInputDesc(0);
    if (inputStorage == nullptr || inputDesc == nullptr) {
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
    constexpr uint64_t LARGE_MIDDLE_REDUCE_THRESHOLD = 2048;
    constexpr uint64_t MIDDLE_OUTPUTS_PER_CORE_TARGET = 16;
    constexpr uint64_t WORKSPACE_REDUCE_INPUT_THRESHOLD = 1U << 18U;
    constexpr uint64_t PACKED_MIDDLE_WORKSPACE_INPUT_THRESHOLD =
        1U << 12U;
    constexpr uint64_t ALIGNED8_MIDDLE_WORKSPACE_INPUT_THRESHOLD =
        1U << 15U;
    constexpr uint64_t WORKSPACE_REDUCE_MIN_ELEMENTS = 2048;
    constexpr uint64_t NONCONTIGUOUS_SPLITK_MIN_REDUCE = 1U << 15U;
    constexpr uint64_t NONCONTIGUOUS_SPLITK_MIN_ROWS = 16U;
    constexpr uint64_t NONCONTIGUOUS_SPLITK_MIN_TAIL = 1024U;
    constexpr uint64_t NONCONTIGUOUS_SPLITK_MAX_TAIL = 16384U;
    constexpr uint64_t NONCONTIGUOUS_SPLITK_MAX_OUTPUTS = 32U;
    constexpr uint64_t NONCONTIGUOUS_LONG_SPLITK_CHUNK = 4096U;
    constexpr uint64_t NONCONTIGUOUS_LONG_SPLITK_MAX_OUTPUTS = 16U;
    constexpr uint64_t WORKSPACE_LAST_MAX_OUTPUTS = 8;
    constexpr uint64_t WORKSPACE_MIDDLE_MAX_OUTPUTS = 1024;
    constexpr uint64_t LONG_CONTIGUOUS_THRESHOLD = 8192;
    constexpr uint64_t TREE_LAST_REDUCE_THRESHOLD =
        WORKSPACE_REDUCE_MIN_ELEMENTS;
    constexpr uint64_t TREE_MIDDLE_REDUCE_THRESHOLD =
        1U << 16U;
    constexpr uint64_t FP32_LONG_TREE_MAX_OUTPUTS = 232U;
    const bool atomicReduce = false;
    uint64_t workspaceInputThreshold =
        WORKSPACE_REDUCE_INPUT_THRESHOLD;
    if (fastPath == 2U && innerElements < 8U) {
        workspaceInputThreshold =
            PACKED_MIDDLE_WORKSPACE_INPUT_THRESHOLD;
    } else if (fastPath == 2U && innerElements == 8U) {
        workspaceInputThreshold =
            ALIGNED8_MIDDLE_WORKSPACE_INPUT_THRESHOLD;
    }
    const bool contiguousWorkspaceReduce =
        inputElements >= workspaceInputThreshold &&
        reduceElements >= WORKSPACE_REDUCE_MIN_ELEMENTS &&
        ((fastPath == 1 &&
          outputElements <= WORKSPACE_LAST_MAX_OUTPUTS) ||
         (fastPath == 2 &&
          outputElements <= WORKSPACE_MIDDLE_MAX_OUTPUTS));
    bool workspaceReduce = contiguousWorkspaceReduce;
    uint32_t reduceMode =
        atomicReduce ? 1U : (contiguousWorkspaceReduce ? 2U : 0U);
    uint64_t packedScratchStride = 0U;
    if (contiguousWorkspaceReduce &&
        fastPath == 2U &&
        innerElements > 0U &&
        innerElements < 8U) {
        uint64_t phaseRows = 1U;
        while ((phaseRows * innerElements) % 8U != 0U) {
            phaseRows <<= 1U;
        }
        if (!SafeMultiply(
                phaseRows,
                innerElements,
                packedScratchStride)) {
            return ge::GRAPH_FAILED;
        }
    }
    const bool longContiguous =
        fastPath == 1 &&
        reduceElements > LONG_CONTIGUOUS_THRESHOLD;
    const bool longMiddle =
        fastPath == 2 &&
        reduceElements >= LARGE_MIDDLE_REDUCE_THRESHOLD;
    const bool longStrided =
        fastPath == 4 &&
        reduceRank > 0U &&
        reduceDims[reduceRank - 1U] >=
            LARGE_MIDDLE_REDUCE_THRESHOLD;
    const bool useLongChunk =
        longContiguous || longMiddle || longStrided;
    const bool fp32LongTreeFitsOneTile =
        inputDesc->GetDataType() != ge::DT_FLOAT ||
        !useLongChunk ||
        outputElements <= FP32_LONG_TREE_MAX_OUTPUTS;
    bool useTreeFinalize =
        contiguousWorkspaceReduce &&
        fp32LongTreeFitsOneTile &&
        ((fastPath == 1 &&
          reduceElements >= TREE_LAST_REDUCE_THRESHOLD) ||
         (fastPath == 2 &&
          reduceElements >= TREE_MIDDLE_REDUCE_THRESHOLD));
    uint64_t expectedTrailingStride = 1U;
    uint32_t firstTrailingAxis = reduceRank;
    for (int32_t axis =
             static_cast<int32_t>(reduceRank) - 1;
         axis >= 0;
         --axis) {
        if (reduceInputStrides[axis] !=
            expectedTrailingStride) {
            break;
        }
        if (!SafeMultiply(
                expectedTrailingStride,
                reduceDims[axis],
                expectedTrailingStride)) {
            return ge::GRAPH_FAILED;
        }
        firstTrailingAxis =
            static_cast<uint32_t>(axis);
    }
    uint32_t innermostOutputAxis = outputRank;
    for (int32_t axis =
             static_cast<int32_t>(outputRank) - 1;
         axis >= 0;
         --axis) {
        if (outputDims[axis] > 1U &&
            outputInputStrides[axis] != 0U) {
            innermostOutputAxis =
                static_cast<uint32_t>(axis);
            break;
        }
    }
    uint32_t innermostPhysicalOutputAxis = outputRank;
    for (int32_t axis =
             static_cast<int32_t>(outputRank) - 1;
         axis >= 0;
         --axis) {
        if (outputInputStrides[axis] != 0U) {
            innermostPhysicalOutputAxis =
                static_cast<uint32_t>(axis);
            break;
        }
    }
    const uint64_t inputTypeBytes =
        inputDesc->GetDataType() == ge::DT_FLOAT
            ? sizeof(float)
            : sizeof(uint16_t);
    const uint64_t elementsPerBlock =
        32U / inputTypeBytes;
    constexpr uint64_t STRIDED_GROUPED_MAX_WIDTH = 8U;
    constexpr uint64_t STRIDED_SPLITK_MIN_INPUT = 1U << 18U;
    constexpr uint64_t STRIDED_SPLITK_MAX_OUTPUTS = 512U;
    constexpr uint64_t NORMAL_CHUNK_ELEMENTS = 8192U;
    uint32_t stridedGroupedOutputAxis = outputRank;
    uint32_t stridedGroupedWidth = 0U;
    uint64_t stridedGroupedRowElements = 0U;
    uint64_t stridedGroupedOuterRows = 0U;
    uint64_t stridedGroupedTasks = 0U;
    bool stridedGroupedRows = false;
    bool stridedGroupedSplitK = false;
    if (fastPath == 4U &&
        reduceMode == 0U &&
        reduceRank > 0U &&
        innerElements > 0U &&
        innerElements <= 16U) {
        const uint64_t lastReduceDim =
            reduceDims[reduceRank - 1U];
        const bool powerOfTwoReduce =
            lastReduceDim > 1U &&
            lastReduceDim <= 4096U &&
            (lastReduceDim & (lastReduceDim - 1U)) == 0U;
        if (powerOfTwoReduce &&
            SafeMultiply(
                lastReduceDim,
                innerElements,
                stridedGroupedRowElements)) {
            for (int32_t axis =
                     static_cast<int32_t>(outputRank) - 1;
                 axis >= 0;
                 --axis) {
                if (outputDims[axis] > 0U &&
                    outputInputStrides[axis] ==
                        stridedGroupedRowElements) {
                    stridedGroupedOutputAxis =
                        static_cast<uint32_t>(axis);
                    break;
                }
            }
        }
        if (stridedGroupedOutputAxis < outputRank) {
            const uint64_t groupedOutputDim =
                outputDims[stridedGroupedOutputAxis];
            uint64_t groupedOutputElements = 0U;
            const uint64_t paddedInnerElements =
                (innerElements +
                 elementsPerBlock - 1U) /
                elementsPerBlock * elementsPerBlock;
            const uint64_t minimumGroupedTasks =
                innerElements <= 8U
                    ? SMALL_FAST_BLOCK_DIM
                    : 4U;
            if (SafeMultiply(
                    groupedOutputDim,
                    innerElements,
                    groupedOutputElements) &&
                groupedOutputElements > 0U &&
                outputElements % groupedOutputElements == 0U) {
                stridedGroupedOuterRows =
                    outputElements / groupedOutputElements;
                uint64_t splitKBufferRows = 0U;
                uint64_t splitKBufferElements = 0U;
                uint64_t splitKTasksPerOuter = 0U;
                uint64_t splitKTasks = 0U;
                const uint64_t outerReduceGroups =
                    reduceElements / lastReduceDim;
                if (innerElements == 2U &&
                    lastReduceDim >= 128U &&
                    outerReduceGroups >= MAX_BLOCK_DIM &&
                    inputElements >= STRIDED_SPLITK_MIN_INPUT &&
                    outputElements <=
                        STRIDED_SPLITK_MAX_OUTPUTS) {
                    for (uint64_t width =
                             STRIDED_GROUPED_MAX_WIDTH;
                         width >= 1U;
                         width /= 2U) {
                        uint64_t compactBufferElements = 0U;
                        uint64_t compactTasksPerOuter = 0U;
                        uint64_t compactTasks = 0U;
                        if (groupedOutputDim < width ||
                            !SafeMultiply(
                                width,
                                stridedGroupedRowElements,
                                compactBufferElements) ||
                            compactBufferElements >
                                NORMAL_CHUNK_ELEMENTS) {
                            continue;
                        }
                        compactTasksPerOuter =
                            (groupedOutputDim + width - 1U) /
                            width;
                        if (!SafeMultiply(
                                stridedGroupedOuterRows,
                                compactTasksPerOuter,
                                compactTasks)) {
                            continue;
                        }
                        stridedGroupedWidth =
                            static_cast<uint32_t>(width);
                        stridedGroupedTasks = compactTasks;
                        stridedGroupedRows = true;
                        stridedGroupedSplitK = true;
                        break;
                    }
                }
                if (innerElements >= 4U &&
                    !stridedGroupedSplitK &&
                    groupedOutputDim >=
                        STRIDED_GROUPED_MAX_WIDTH &&
                    outerReduceGroups >= MAX_BLOCK_DIM &&
                    inputElements >= STRIDED_SPLITK_MIN_INPUT &&
                    outputElements <=
                        STRIDED_SPLITK_MAX_OUTPUTS &&
                    SafeMultiply(
                        STRIDED_GROUPED_MAX_WIDTH,
                        lastReduceDim,
                        splitKBufferRows) &&
                    SafeMultiply(
                        splitKBufferRows,
                        paddedInnerElements,
                        splitKBufferElements) &&
                    splitKBufferElements <=
                        NORMAL_CHUNK_ELEMENTS) {
                    splitKTasksPerOuter =
                        (groupedOutputDim +
                         STRIDED_GROUPED_MAX_WIDTH - 1U) /
                        STRIDED_GROUPED_MAX_WIDTH;
                    if (SafeMultiply(
                            stridedGroupedOuterRows,
                            splitKTasksPerOuter,
                            splitKTasks)) {
                        stridedGroupedWidth =
                            static_cast<uint32_t>(
                                STRIDED_GROUPED_MAX_WIDTH);
                        stridedGroupedTasks = splitKTasks;
                        stridedGroupedRows = true;
                        stridedGroupedSplitK = true;
                    }
                }
                for (uint64_t width =
                         STRIDED_GROUPED_MAX_WIDTH;
                     !stridedGroupedSplitK && width >= 1U;
                     width /= 2U) {
                    uint64_t groupedBufferRows = 0U;
                    uint64_t groupedBufferElements = 0U;
                    uint64_t groupedTasksPerOuter = 0U;
                    uint64_t candidateTasks = 0U;
                    if (groupedOutputDim < width ||
                        !SafeMultiply(
                            width,
                            lastReduceDim,
                            groupedBufferRows) ||
                        !SafeMultiply(
                            groupedBufferRows,
                            paddedInnerElements,
                            groupedBufferElements) ||
                        groupedBufferElements >
                            NORMAL_CHUNK_ELEMENTS) {
                        continue;
                    }
                    groupedTasksPerOuter =
                        (groupedOutputDim + width - 1U) /
                        width;
                    if (!SafeMultiply(
                            stridedGroupedOuterRows,
                            groupedTasksPerOuter,
                            candidateTasks) ||
                        candidateTasks <
                            minimumGroupedTasks) {
                        continue;
                    }
                    stridedGroupedWidth =
                        static_cast<uint32_t>(width);
                    stridedGroupedTasks = candidateTasks;
                    stridedGroupedRows = true;
                    break;
                }
            }
        }
    }
    if (stridedGroupedSplitK) {
        workspaceReduce = true;
        reduceMode = 5U;
        useTreeFinalize = true;
    }
    uint64_t splitKNaturalRows = 0U;
    uint64_t splitKSourceGapBytes = 0U;
    const bool noncontiguousSplitK =
        fastPath == 3U &&
        inputElements >= WORKSPACE_REDUCE_INPUT_THRESHOLD &&
        reduceElements >= NONCONTIGUOUS_SPLITK_MIN_REDUCE &&
        outputElements > 0U &&
        outputElements <= NONCONTIGUOUS_SPLITK_MAX_OUTPUTS &&
        trailingReduceElements >= NONCONTIGUOUS_SPLITK_MIN_TAIL &&
        trailingReduceElements <= NONCONTIGUOUS_SPLITK_MAX_TAIL &&
        firstTrailingAxis > 0U &&
        firstTrailingAxis < reduceRank &&
        expectedTrailingStride == trailingReduceElements &&
        reduceDims[firstTrailingAxis - 1U] > 0U &&
        reduceInputStrides[firstTrailingAxis - 1U] >=
            trailingReduceElements &&
        SafeMultiply(
            reduceInputStrides[firstTrailingAxis - 1U] -
                trailingReduceElements,
            inputTypeBytes,
            splitKSourceGapBytes) &&
        splitKSourceGapBytes <=
            std::numeric_limits<uint32_t>::max() &&
        (splitKNaturalRows =
             reduceElements / trailingReduceElements) >=
            NONCONTIGUOUS_SPLITK_MIN_ROWS;
    if (noncontiguousSplitK) {
        workspaceReduce = true;
        reduceMode = 3U;
        useTreeFinalize = true;
    }
    const uint64_t longTailNaturalRows =
        trailingReduceElements == 0U
            ? 0U
            : reduceElements / trailingReduceElements;
    const uint64_t longTailChunksPerRow =
        trailingReduceElements == 0U
            ? 0U
            : (trailingReduceElements - 1U) /
                    NONCONTIGUOUS_LONG_SPLITK_CHUNK +
                1U;
    uint64_t longTailWorkUnits = 0U;
    const bool noncontiguousLongTailSplitK =
        fastPath == 3U &&
        inputElements >= WORKSPACE_REDUCE_INPUT_THRESHOLD &&
        reduceElements >= NONCONTIGUOUS_SPLITK_MIN_REDUCE &&
        outputElements > 0U &&
        outputElements <= NONCONTIGUOUS_LONG_SPLITK_MAX_OUTPUTS &&
        trailingReduceElements > NONCONTIGUOUS_SPLITK_MAX_TAIL &&
        firstTrailingAxis > 0U &&
        firstTrailingAxis < reduceRank &&
        expectedTrailingStride == trailingReduceElements &&
        SafeMultiply(
            longTailNaturalRows,
            longTailChunksPerRow,
            longTailWorkUnits) &&
        longTailWorkUnits >= MAX_BLOCK_DIM;
    if (noncontiguousLongTailSplitK) {
        workspaceReduce = true;
        reduceMode = 4U;
        useTreeFinalize = true;
    }
    uint64_t vectorInputElements = 0;
    uint64_t vectorSourceGapBytes = 0;
    uint64_t longVectorSourceGapBytes = 0;
    uint64_t groupedLongSourceGapBytes = 0;
    const bool lastVectorCandidate =
        fastPath == 1U &&
        reduceMode == 0U &&
        reduceElements > 4096U &&
        SafeMultiply(
            reduceElements - 1U,
            inputTypeBytes,
            longVectorSourceGapBytes) &&
        longVectorSourceGapBytes <=
            std::numeric_limits<uint32_t>::max();
    const bool lastVectorMultiStrideSafe =
        lastVectorCandidate &&
        longVectorSourceGapBytes <=
            std::numeric_limits<uint16_t>::max();
    const uint64_t lastVectorTargetBlocks =
        inputElements >= FULL_CORE_INPUT_THRESHOLD
            ? MAX_BLOCK_DIM
            : SMALL_FAST_BLOCK_DIM;
    const bool lastVector8 =
        lastVectorMultiStrideSafe &&
        outputElements >= lastVectorTargetBlocks * 8U;
    const uint64_t lastVector4Remainder =
        reduceElements % 4096U;
    const uint64_t lastVector4AlignedRemainder =
        (lastVector4Remainder + elementsPerBlock - 1U) /
        elementsPerBlock * elementsPerBlock;
    const bool lastVector4StrideSafe =
        inputTypeBytes != sizeof(float) ||
        lastVector4Remainder == 0U ||
        lastVector4AlignedRemainder >= 2056U;
    const bool lastVector4 =
        lastVectorMultiStrideSafe &&
        !lastVector8 &&
        outputElements >= lastVectorTargetBlocks * 4U &&
        lastVector4StrideSafe;
    const bool lastVector2 =
        lastVectorMultiStrideSafe &&
        !lastVector8 &&
        !lastVector4 &&
        reduceElements <= 8192U &&
        outputElements >= lastVectorTargetBlocks * 2U &&
        lastVector4StrideSafe;
    const bool lastVectorSmall1 =
        fastPath == 1U &&
        reduceMode == 0U &&
        reduceElements > 64U &&
        reduceElements <= 4096U &&
        outputElements > 0U &&
        outputElements < SMALL_FAST_BLOCK_DIM;
    const bool lastVector1 =
        (lastVectorCandidate &&
         !lastVector8 &&
         !lastVector4 &&
         !lastVector2) ||
        lastVectorSmall1;
    const uint64_t lastVectorWidth =
        lastVector8
            ? 8U
            : (lastVector4
                ? 4U
                : (lastVector2 ? 2U : (lastVector1 ? 1U : 0U)));
    const bool validGroupedVectorWidth = SafeMultiply(
        trailingReduceElements,
        8U,
        vectorInputElements);
    const uint64_t groupedPaddedVector =
        validGroupedVectorWidth
            ? (vectorInputElements + elementsPerBlock - 1U) /
                elementsPerBlock * elementsPerBlock
            : 0U;
    const bool groupedShortTail =
        trailingReduceElements > 0U &&
        trailingReduceElements <= 64U &&
        validGroupedVectorWidth &&
        vectorInputElements <= 8192U;
    const bool groupedMediumTail =
        trailingReduceElements > 64U &&
        trailingReduceElements < 1024U &&
        validGroupedVectorWidth &&
        groupedPaddedVector / 8U <= 255U;
    const uint64_t groupedPaddedTail =
        (trailingReduceElements + elementsPerBlock - 1U) /
        elementsPerBlock * elementsPerBlock;
    const uint64_t groupedLongChunks =
        (trailingReduceElements + 1023U) / 1024U;
    const uint64_t legacyGroupedRows =
        groupedPaddedTail == 0U
            ? 0U
            : 8192U / groupedPaddedTail;
    const bool groupedLongDmaSafe =
        groupedPaddedTail > 8192U ||
        groupedLongChunks * legacyGroupedRows <= 8U;
    const bool groupedLongTail =
        trailingReduceElements >= 1024U &&
        trailingReduceElements <=
            static_cast<uint64_t>(
                std::numeric_limits<uint32_t>::max()) &&
        validGroupedVectorWidth &&
        SafeMultiply(
            trailingReduceElements - 1U,
            inputTypeBytes,
            groupedLongSourceGapBytes) &&
        groupedLongSourceGapBytes <=
            std::numeric_limits<uint32_t>::max() &&
        groupedLongDmaSafe;
    bool groupedVectorCandidate =
        fastPath == 3U &&
        reduceMode == 0U &&
        firstTrailingAxis > 0U &&
        firstTrailingAxis < reduceRank &&
        expectedTrailingStride == trailingReduceElements &&
        innermostOutputAxis < outputRank &&
        trailingReduceElements > 0U &&
        trailingReduceElements <=
            static_cast<uint64_t>(
                std::numeric_limits<int32_t>::max()) &&
        outputDims[innermostOutputAxis] >= 8U &&
        outputInputStrides[innermostOutputAxis] ==
            trailingReduceElements &&
        (groupedShortTail || groupedMediumTail ||
         groupedLongTail);
    bool groupedVector8 =
        groupedVectorCandidate &&
        outputElements % 8U == 0U &&
        outputDims[innermostOutputAxis] % 8U == 0U;
    bool groupedVectorAdaptive =
        groupedVectorCandidate && !groupedVector8 &&
        !groupedMediumTail;
    uint64_t adaptiveVectorTasks = 0U;
    if (groupedVectorCandidate) {
        const uint32_t batchAxis =
            firstTrailingAxis - 1U;
        const bool flatVectorCopy =
            groupedVector8 &&
            trailingReduceElements <= 64U &&
            trailingReduceElements % elementsPerBlock == 0U;
        const uint64_t copiedRowElements =
            flatVectorCopy || groupedMediumTail ||
                    groupedLongTail
                ? vectorInputElements
                : trailingReduceElements;
        const bool validVectorLayout =
            reduceDims[batchAxis] > 0U &&
            reduceInputStrides[batchAxis] >=
                vectorInputElements &&
            SafeMultiply(
                reduceInputStrides[batchAxis] -
                    copiedRowElements,
                inputTypeBytes,
                vectorSourceGapBytes) &&
            vectorSourceGapBytes <=
                std::numeric_limits<uint32_t>::max();
        groupedVector8 =
            groupedVector8 && validVectorLayout;
        groupedVectorAdaptive =
            groupedVectorAdaptive && validVectorLayout;
        if (groupedMediumTail && groupedVector8) {
            const uint64_t paddedVector =
                groupedPaddedVector;
            const uint64_t vectorRows =
                paddedVector == 0U
                    ? 0U
                    : 8192U / paddedVector;
            const uint64_t vectorDma =
                vectorRows == 0U
                    ? std::numeric_limits<uint64_t>::max()
                    : (reduceDims[batchAxis] + vectorRows - 1U) /
                        vectorRows;
            const uint64_t legacyDma =
                legacyGroupedRows == 0U
                    ? 0U
                    : 8U *
                        ((reduceDims[batchAxis] +
                          legacyGroupedRows - 1U) /
                         legacyGroupedRows);
            groupedVector8 =
                vectorRows > 0U &&
                vectorDma <= legacyDma;
        }
        if (groupedVectorAdaptive) {
            const uint64_t innerOutputDim =
                outputDims[innermostOutputAxis];
            const uint64_t outputRows =
                outputElements / innerOutputDim;
            const uint64_t tasksPerRow =
                (innerOutputDim + 7U) / 8U;
            groupedVectorAdaptive = SafeMultiply(
                outputRows,
                tasksPerRow,
                adaptiveVectorTasks);
        }
    }
    uint32_t groupedShortVectorWidth = 0U;
    if (!groupedVector8 &&
        !groupedVectorAdaptive &&
        fastPath == 3U &&
        reduceMode == 0U &&
        firstTrailingAxis > 0U &&
        firstTrailingAxis < reduceRank &&
        expectedTrailingStride == trailingReduceElements &&
        innermostPhysicalOutputAxis < outputRank &&
        outputDims[innermostPhysicalOutputAxis] > 0U &&
        outputDims[innermostPhysicalOutputAxis] < 8U &&
        trailingReduceElements > 0U &&
        trailingReduceElements <= 64U &&
        outputInputStrides[innermostPhysicalOutputAxis] ==
            trailingReduceElements) {
        const uint32_t batchAxis = firstTrailingAxis - 1U;
        const uint64_t batchDim = reduceDims[batchAxis];
        for (uint64_t width = 4U;
             width >= 1U;
             width /= 2U) {
            if (outputElements % width != 0U ||
                outputDims[innermostPhysicalOutputAxis] % width != 0U) {
                continue;
            }
            uint64_t vectorElements = 0U;
            uint64_t sourceGapBytes = 0U;
            if (!SafeMultiply(
                    trailingReduceElements,
                    width,
                    vectorElements) ||
                reduceInputStrides[batchAxis] < vectorElements ||
                !SafeMultiply(
                    reduceInputStrides[batchAxis] - vectorElements,
                    inputTypeBytes,
                    sourceGapBytes) ||
                sourceGapBytes >
                    std::numeric_limits<uint32_t>::max()) {
                continue;
            }
            const uint64_t alignedVector =
                (vectorElements + elementsPerBlock - 1U) /
                elementsPerBlock * elementsPerBlock;
            const uint64_t paddedVector =
                alignedVector < 64U ? 64U : alignedVector;
            uint64_t vectorRows = 0U;
            if (trailingReduceElements % elementsPerBlock == 0U) {
                vectorRows = 8192U / vectorElements;
                if (vectorRows > 31U) {
                    vectorRows = 31U;
                }
            } else {
                vectorRows = 8192U / paddedVector;
            }
            if (vectorRows == 0U || legacyGroupedRows == 0U) {
                continue;
            }
            const uint64_t vectorDma =
                (batchDim + vectorRows - 1U) / vectorRows;
            const uint64_t legacyDma =
                width *
                ((batchDim + legacyGroupedRows - 1U) /
                 legacyGroupedRows);
            if (vectorDma <= legacyDma) {
                groupedShortVectorWidth =
                    static_cast<uint32_t>(width);
                break;
            }
        }
    }
    uint32_t groupedMediumVectorWidth = 0U;
    if (!groupedVector8 &&
        fastPath == 3U &&
        reduceMode == 0U &&
        firstTrailingAxis > 0U &&
        firstTrailingAxis < reduceRank &&
        expectedTrailingStride == trailingReduceElements &&
        innermostPhysicalOutputAxis < outputRank &&
        trailingReduceElements > 64U &&
        trailingReduceElements < 1024U &&
        outputInputStrides[innermostPhysicalOutputAxis] ==
            trailingReduceElements) {
        const uint32_t batchAxis = firstTrailingAxis - 1U;
        const uint64_t batchDim = reduceDims[batchAxis];
        const uint64_t maxWindows =
            (trailingReduceElements + 70U) / 64U;
        for (uint64_t width = 4U;
             width >= 1U;
             width /= 2U) {
            uint64_t mediumInputElements = 0U;
            uint64_t mediumSourceGapBytes = 0U;
            uint64_t partialBufferElements = 0U;
            if (outputElements % width != 0U ||
                outputDims[innermostPhysicalOutputAxis] < width ||
                outputDims[innermostPhysicalOutputAxis] % width != 0U ||
                !SafeMultiply(
                    trailingReduceElements,
                    width,
                    mediumInputElements)) {
                continue;
            }
            const uint64_t paddedVector =
                (mediumInputElements + elementsPerBlock - 1U) /
                elementsPerBlock * elementsPerBlock;
            if (paddedVector == 0U ||
                paddedVector > 8192U ||
                paddedVector / 8U > 255U ||
                reduceInputStrides[batchAxis] <
                    mediumInputElements ||
                !SafeMultiply(
                    reduceInputStrides[batchAxis] -
                        mediumInputElements,
                    inputTypeBytes,
                    mediumSourceGapBytes) ||
                mediumSourceGapBytes >
                    std::numeric_limits<uint32_t>::max()) {
                continue;
            }
            const uint64_t vectorRows = 8192U / paddedVector;
            const uint64_t partialsPerOutput =
                vectorRows * maxWindows;
            const uint64_t paddedPartialsPerOutput =
                (partialsPerOutput + 63U) / 64U * 64U;
            if (!SafeMultiply(
                    width,
                    paddedPartialsPerOutput,
                    partialBufferElements) ||
                partialBufferElements > 8192U ||
                paddedPartialsPerOutput / 64U > 64U) {
                continue;
            }
            const uint64_t vectorDma =
                (batchDim + vectorRows - 1U) / vectorRows;
            const uint64_t legacyDma =
                width *
                ((batchDim + legacyGroupedRows - 1U) /
                 legacyGroupedRows);
            if (vectorDma <= legacyDma) {
                groupedMediumVectorWidth =
                    static_cast<uint32_t>(width);
                break;
            }
        }
    }
    uint32_t groupedLongVectorWidth = 0U;
    if (!groupedVector8 &&
        !groupedVectorAdaptive &&
        fastPath == 3U &&
        reduceMode == 0U &&
        firstTrailingAxis > 0U &&
        firstTrailingAxis < reduceRank &&
        expectedTrailingStride == trailingReduceElements &&
        innermostPhysicalOutputAxis < outputRank &&
        outputDims[innermostPhysicalOutputAxis] > 0U &&
        outputDims[innermostPhysicalOutputAxis] < 8U &&
        trailingReduceElements >= 1024U &&
        trailingReduceElements <=
            static_cast<uint64_t>(
                std::numeric_limits<int32_t>::max()) &&
        outputInputStrides[innermostPhysicalOutputAxis] ==
            trailingReduceElements &&
        SafeMultiply(
            trailingReduceElements - 1U,
            inputTypeBytes,
            groupedLongSourceGapBytes) &&
        groupedLongSourceGapBytes <=
            std::numeric_limits<uint32_t>::max()) {
        const uint32_t batchAxis = firstTrailingAxis - 1U;
        const uint64_t batchDim = reduceDims[batchAxis];
        const uint64_t legacyDmaPerOutput =
            groupedPaddedTail <= 8192U &&
                    legacyGroupedRows > 0U
                ? (batchDim + legacyGroupedRows - 1U) /
                    legacyGroupedRows
                : batchDim *
                    ((trailingReduceElements + 8191U) / 8192U);
        for (uint64_t width = 4U;
             width >= 1U;
             width /= 2U) {
            if (outputElements % width != 0U ||
                outputDims[innermostPhysicalOutputAxis] % width != 0U) {
                continue;
            }
            uint64_t copiedRowElements = 0U;
            uint64_t sourceGapBytes = 0U;
            if (!SafeMultiply(
                    trailingReduceElements,
                    width,
                    copiedRowElements) ||
                reduceInputStrides[batchAxis] <
                    copiedRowElements ||
                !SafeMultiply(
                    reduceInputStrides[batchAxis] -
                        copiedRowElements,
                    inputTypeBytes,
                    sourceGapBytes) ||
                sourceGapBytes >
                    std::numeric_limits<uint32_t>::max()) {
                continue;
            }
            const uint64_t rowChunk = 8192U / width;
            const uint64_t remainder =
                trailingReduceElements % rowChunk;
            const uint64_t alignedRemainder =
                (remainder + elementsPerBlock - 1U) /
                elementsPerBlock * elementsPerBlock;
            const uint64_t destinationStrideBlocks =
                width == 1U || remainder == 0U
                    ? 0U
                    : (rowChunk - alignedRemainder) *
                        inputTypeBytes / 32U;
            if (destinationStrideBlocks > 255U) {
                continue;
            }
            const uint64_t vectorDmaPerTask =
                batchDim *
                ((trailingReduceElements + rowChunk - 1U) /
                 rowChunk);
            uint64_t legacyDmaForWidth = 0U;
            if (!SafeMultiply(
                    width,
                    legacyDmaPerOutput,
                    legacyDmaForWidth) ||
                vectorDmaPerTask > legacyDmaForWidth) {
                continue;
            }
            groupedLongVectorWidth =
                static_cast<uint32_t>(width);
            break;
        }
    }
    const uint32_t groupedFixedVectorWidth =
        groupedShortVectorWidth != 0U
            ? groupedShortVectorWidth
            : (groupedLongVectorWidth != 0U
                ? groupedLongVectorWidth
                : groupedMediumVectorWidth);
    uint64_t desiredBlocks = 0;
    if (reduceMode != 0U) {
        desiredBlocks = MAX_BLOCK_DIM;
    } else if (stridedGroupedRows) {
        desiredBlocks =
            stridedGroupedTasks < MAX_BLOCK_DIM
                ? stridedGroupedTasks
                : MAX_BLOCK_DIM;
    } else if (lastVectorWidth != 0U) {
        const uint64_t vectorTasks =
            (outputElements + lastVectorWidth - 1U) /
            lastVectorWidth;
        desiredBlocks =
            vectorTasks < MAX_BLOCK_DIM
                ? vectorTasks
                : MAX_BLOCK_DIM;
    } else if (groupedVector8) {
        const uint64_t vectorTasks =
            outputElements / 8U;
        desiredBlocks =
            vectorTasks < MAX_BLOCK_DIM
                ? vectorTasks
                : MAX_BLOCK_DIM;
    } else if (groupedVectorAdaptive) {
        desiredBlocks =
            adaptiveVectorTasks < MAX_BLOCK_DIM
                ? adaptiveVectorTasks
                : MAX_BLOCK_DIM;
    } else if (groupedFixedVectorWidth != 0U) {
        const uint64_t vectorTasks =
            outputElements / groupedFixedVectorWidth;
        desiredBlocks =
            vectorTasks < MAX_BLOCK_DIM
                ? vectorTasks
                : MAX_BLOCK_DIM;
    } else if (
        fastPath == 1 ||
        fastPath == 3 ||
        (fastPath == 4 &&
         innerElements <= 16U)) {
        const uint32_t fastBlockDim =
            inputElements >= FULL_CORE_INPUT_THRESHOLD
                ? MAX_BLOCK_DIM
                : SMALL_FAST_BLOCK_DIM;
        desiredBlocks =
            outputElements < fastBlockDim
                ? outputElements
                : fastBlockDim;
    } else if (
        fastPath == 2 &&
        reduceElements >= LARGE_MIDDLE_REDUCE_THRESHOLD) {
        desiredBlocks =
            (outputElements +
             MIDDLE_OUTPUTS_PER_CORE_TARGET - 1) /
            MIDDLE_OUTPUTS_PER_CORE_TARGET;
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
    const uint64_t scheduledOutputs =
        stridedGroupedRows
            ? stridedGroupedTasks
            : (lastVectorWidth != 0U
            ? (outputElements + lastVectorWidth - 1U) /
                lastVectorWidth
            : (groupedVector8
                ? outputElements / 8U
                : (groupedVectorAdaptive
                    ? adaptiveVectorTasks
                    : (groupedFixedVectorWidth != 0U
                        ? outputElements /
                            groupedFixedVectorWidth
                        : outputElements))));
    tiling.set_baseOutputsPerBlock(
        scheduledOutputs / blockDim);
    tiling.set_extraBlocks(static_cast<uint32_t>(
        scheduledOutputs % blockDim));
    tiling.set_outputRank(outputRank);
    tiling.set_reduceRank(reduceRank);
    tiling.set_fastPath(fastPath);
    tiling.set_reduceMode(reduceMode);
    tiling.set_outputDims(outputDims);
    tiling.set_outputInputStrides(outputInputStrides);
    tiling.set_reduceDims(reduceDims);
    tiling.set_reduceInputStrides(reduceInputStrides);

    context->SetTilingKey(
        reduceElements == 0U
            ? 13U
            : (stridedGroupedRows
            ? (stridedGroupedWidth == 8U
                ? 5U
                : (stridedGroupedWidth == 4U
                    ? 9U
                    : (stridedGroupedWidth == 2U
                        ? 10U
                        : 11U)))
            : (noncontiguousSplitK || noncontiguousLongTailSplitK
            ? 4U
            : (lastVector8
            ? 7U
            : (lastVector4
                ? 8U
                : (lastVector2
                    ? 10U
                    : (lastVector1
                        ? (reduceElements <= 8192U ? 11U : 12U)
                        : (groupedVector8
                            ? 5U
                            : (groupedVectorAdaptive
                                ? 6U
                                : (groupedFixedVectorWidth == 4U
                                    ? 9U
                                    : (groupedFixedVectorWidth == 2U
                                        ? 10U
                                        : (groupedFixedVectorWidth == 1U
                                            ? 11U
                                             : (useTreeFinalize
                                                 ? (useLongChunk ? 4U : 3U)
                                                  : (useLongChunk ? 2U : 1U))))))))))))));
    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (atomicReduce || workspaceReduce) {
        const uint64_t partialStride =
            (outputElements + 7U) / 8U * 8U;
        const uint64_t partialBlocks =
            workspaceReduce ? blockDim : 1U;
        uint64_t partialElements = 0;
        uint64_t scratchElements = 0;
        uint64_t workspaceElements = 0;
        uint64_t userWorkspaceBytes = 0;
        if (!SafeMultiply(
                partialStride,
                partialBlocks,
                partialElements) ||
            !SafeMultiply(
                packedScratchStride,
                partialBlocks,
                scratchElements) ||
            !SafeAdd(
                partialElements,
                scratchElements,
                workspaceElements) ||
            !SafeMultiply(
                workspaceElements,
                sizeof(float),
                userWorkspaceBytes)) {
            return ge::GRAPH_FAILED;
        }
        const platform_ascendc::PlatformAscendC ascendcPlatform(
            context->GetPlatformInfo());
        const size_t systemWorkspace =
            ascendcPlatform.GetLibApiWorkSpaceSize();
        uint64_t totalWorkspaceBytes = 0;
        if (!SafeAdd(
                userWorkspaceBytes,
                static_cast<uint64_t>(systemWorkspace),
                totalWorkspaceBytes) ||
            totalWorkspaceBytes >
                std::numeric_limits<size_t>::max()) {
            return ge::GRAPH_FAILED;
        }
        workspace[0] =
            static_cast<size_t>(totalWorkspaceBytes);
    } else {
        workspace[0] = 0;
    }
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
