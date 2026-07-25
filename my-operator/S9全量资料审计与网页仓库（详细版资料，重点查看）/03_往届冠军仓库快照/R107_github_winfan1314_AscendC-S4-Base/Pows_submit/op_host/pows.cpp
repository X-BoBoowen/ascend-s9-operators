#include "pows_tiling.h"
#include "host_inc.h"

constexpr uint64_t MINI_BATCH_SIZE = 32;

#define OPTI_SHAPES 1
template<class Tp>
uint32_t set_remove(Tp *x1, Tp *x2, Tp *y, const uint32_t size, uint32_t idx, const uint32_t count, const Tp &val) {
    if (idx >= size) return size;
    x1[idx] = val;
    x2[idx] = val;
    y[idx] = val;
    idx += count;
    while(idx < size) {
        x1[idx - count + 1] = x1[idx];
        x2[idx - count + 1] = x2[idx];
        y[idx - count + 1] = y[idx];
        idx++;
    }
    return size - count + 1;
}
template<class Tp>
std::tuple<bool, uint32_t, uint32_t, Tp> check_arrs(const Tp *x1, const Tp *x2, const uint32_t size) {
    for(uint32_t i = 0; i < size - 1; i++) 
        if(x1[i] == x2[i] && x1[i + 1] == x2[i + 1])
            return {true, i, 2, x1[i] * x1[i + 1]};
    return { false, 0, 0, Tp{} };
}

namespace optiling {
    constexpr uint32_t BUFFER_NUM = 2;
    constexpr uint32_t BUFFER_COUNT_32 = 2;
    static graphStatus TilingFunc(gert::TilingContext* context) {
        auto dev_info = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());              // AscendC Platform Info
        uint32_t core_num = dev_info.GetCoreNum();                                                  // AI Core Number
        context->SetBlockDim(core_num);                                                 
        uint64_t ub_size; dev_info.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);      // Unified Buffer Size
        // get runtime shapes
        const auto x1_shape = context->GetInputShape(0);
        const auto x2_shape = context->GetInputShape(1);
        const auto y_shape = context->GetOutputShape(0);
        bool need_broadcast = *x1_shape != *x2_shape; 
        if(!need_broadcast) {
            PowsTilingDataNoBC tiling;
            context->SetTilingKey(1);
            uint32_t type_sz = GetSizeByDataType(context->GetInputTensor(0)->GetDataType());
            uint32_t mini_batch = MINI_BATCH_SIZE / type_sz;
            uint32_t ub_size_per_it;
            switch(context->GetInputTensor(0)->GetDataType()) {
                case DT_BF16:
                    ub_size_per_it = 2 * 4 + BUFFER_NUM * 3 * type_sz;
                    break;
                case DT_FLOAT:
                    ub_size_per_it = BUFFER_NUM * BUFFER_COUNT_32 * type_sz;
                    break;
                case DT_FLOAT16:
                    ub_size_per_it = 2 * 4 + BUFFER_NUM * 3 * type_sz;
                    break;
                default:
                    return GRAPH_FAILED;    // error
            }
            uint32_t tileLength = FLOOR(ub_size / ub_size_per_it, 512 / type_sz);
            // uint32_t tileLength = 8192;
            uint32_t yLength = 1;
            for (int i = 0; i < y_shape->GetStorageShape().GetDimNum(); i++) 
                yLength  *= y_shape->GetStorageShape().GetDim(i);
            uint32_t tileNumber = yLength / tileLength;
            uint32_t maxLength = tileNumber * tileLength;
            uint32_t reminder = CEIL(yLength % tileLength, mini_batch);
            SET(tileLength);
            SET(reminder);
            SET(maxLength);
#if DEBUG_OUTPUT
            printf("TilingFunc: Using Non-broadcast Kernel\n");
            printf("TilingFunc: tileLength: %u, tileNum: %u, reminder: %u\n", tileLength, tileNumber, reminder);
            printf("TilingFunc: Core Number: %u, UB Size: %u\n", core_num, ub_size);
            printf("TilingFunc: TK=%lu\n", context->GetTilingKey());
#endif // #if DEBUG_OUTPUT
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
            return GRAPH_SUCCESS;
        }
        uint32_t *x1Shape = new uint32_t[MAX_SHAPE_SIZE], *x2Shape = new uint32_t[MAX_SHAPE_SIZE], *yShape = new uint32_t[MAX_SHAPE_SIZE];
        uint32_t *x1ShapeRSum = new uint32_t[MAX_SHAPE_SIZE + 1], *x2ShapeRSum = new uint32_t[MAX_SHAPE_SIZE + 1], *yShapeRSum = new uint32_t[MAX_SHAPE_SIZE + 1];
        uint32_t x1ShapeSize = x1_shape->GetStorageShape().GetDimNum();
        uint32_t x2ShapeSize = x2_shape->GetStorageShape().GetDimNum();
        uint32_t yShapeSize = y_shape->GetStorageShape().GetDimNum();
        uint32_t shapeSize = max(x1ShapeSize, x2ShapeSize);
        for (int i = x1ShapeSize - 1; i >= 0; i--) 
            x1Shape[x1ShapeSize - 1 - i] = x1_shape->GetStorageShape().GetDim(i);
        for (int i = x2ShapeSize - 1; i >= 0; i--) 
            x2Shape[x2ShapeSize - 1 - i] = x2_shape->GetStorageShape().GetDim(i);
        if (x1ShapeSize < x2ShapeSize) {
            for(uint32_t i = x1ShapeSize; i < x2ShapeSize; i++)
                x1Shape[i] = 1;
            x1ShapeSize = x2ShapeSize;
        } else if (x1ShapeSize > x2ShapeSize) {
            for(uint32_t i = x2ShapeSize; i < x1ShapeSize; i++)
                x2Shape[i] = 1;
            x2ShapeSize = x1ShapeSize;
        }
        for (uint32_t i = 0; i < shapeSize; i++) 
            yShape[i] = max(x1Shape[i], x2Shape[i]);
        int32_t x1Length = 1, x2Length = 1, yLength = 1;
        x1ShapeRSum[0] = x2ShapeRSum[0] = yShapeRSum[0] = 1;
        for (int i = 1; i <= shapeSize; i++) {
            x1Length *= x1Shape[i - 1];
            x2Length *= x2Shape[i - 1];
            yLength  *= yShape[i - 1];
            x1ShapeRSum[i] = x1Length;
            x2ShapeRSum[i] = x2Length;
            yShapeRSum[i] = yLength;
        }
#if !OPTI_SHAPES && DEBUG_OUTPUT
        printf("TilingFunc:  x1  Shape: [");
        for(uint32_t i = shapeSize - 1; i > 0; i--) 
            printf("%u, ", x1Shape[i]);
        printf("%u], size: %u\n", x1Shape[0], shapeSize);
        printf("TilingFunc:  x2  Shape: [");
        for(uint32_t i = shapeSize - 1; i > 0; i--) 
            printf("%u, ", x2Shape[i]);
        printf("%u], size: %u\n", x2Shape[0], shapeSize);
        printf("TilingFunc:  ^y  Shape: [");
        for(uint32_t i = shapeSize - 1; i > 0; i--) 
            printf("%u, ", yShape[i]);
        printf("%u], size: %u\n", yShape[0], shapeSize);
        printf("TilingFunc:   y  Shape: [");
        for(uint32_t i = 0; i < yShapeSize - 1; i++) 
            printf("%lu, ", y_shape->GetStorageShape().GetDim(i));
        printf("%lu], size: %u\n", y_shape->GetStorageShape().GetDim(yShapeSize - 1), yShapeSize);
        printf("TilingFunc:  x1  Shape RSum: [");
        for(uint32_t i = shapeSize; i > 0; i--) 
            printf("%u, ", x1ShapeRSum[i]);
        printf("%u]\n", x1ShapeRSum[0]);
        printf("TilingFunc:  x2  Shape RSum: [");
        for(uint32_t i = shapeSize; i > 0; i--) 
            printf("%u, ", x2ShapeRSum[i]);
        printf("%u]\n", x2ShapeRSum[0]);
        printf("TilingFunc:   y  Shape RSum: [");
        for(uint32_t i = shapeSize; i > 0; i--) 
            printf("%u, ", yShapeRSum[i]);
        printf("%u]\n", yShapeRSum[0]);
#endif // #if !OPTI_SHAPES && DEBUG_OUTPUT
#if OPTI_SHAPES
        while(true) {
            const auto [flag, idx, count, val] = check_arrs(x1Shape, x2Shape, shapeSize);
            if(!flag) break;
            shapeSize = set_remove(x1Shape, x2Shape, yShape, shapeSize, idx, count, val);
        }
        x1Length = 1, x2Length = 1, yLength = 1;
        x1ShapeRSum[0] = x2ShapeRSum[0] = yShapeRSum[0] = 1;
        for (int i = 1; i <= shapeSize; i++) {
            x1Length *= x1Shape[i - 1];
            x2Length *= x2Shape[i - 1];
            yLength  *= yShape[i - 1];
            x1ShapeRSum[i] = x1Length;
            x2ShapeRSum[i] = x2Length;
            yShapeRSum[i] = yLength;
        }
#if DEBUG_OUTPUT
        printf("TilingFunc: Optimized shapes:\n");
        printf("TilingFunc:  x1  Shape: [");
        for(uint32_t i = shapeSize - 1; i > 0; i--) 
            printf("%u, ", x1Shape[i]);
        printf("%u], size: %u\n", x1Shape[0], shapeSize);
        printf("TilingFunc:  x2  Shape: [");
        for(uint32_t i = shapeSize - 1; i > 0; i--) 
            printf("%u, ", x2Shape[i]);
        printf("%u], size: %u\n", x2Shape[0], shapeSize);
        printf("TilingFunc:   y  Shape: [");
        for(uint32_t i = shapeSize - 1; i > 0; i--) 
            printf("%u, ", yShape[i]);
        printf("%u], size: %u\n", yShape[0], shapeSize);
        printf("TilingFunc:  x1  Shape RSum: [");
        for(uint32_t i = shapeSize; i > 0; i--) 
            printf("%u, ", x1ShapeRSum[i]);
        printf("%u]\n", x1ShapeRSum[0]);
        printf("TilingFunc:  x2  Shape RSum: [");
        for(uint32_t i = shapeSize; i > 0; i--) 
            printf("%u, ", x2ShapeRSum[i]);
        printf("%u]\n", x2ShapeRSum[0]);
        printf("TilingFunc:   y  Shape RSum: [");
        for(uint32_t i = shapeSize; i > 0; i--) 
            printf("%u, ", yShapeRSum[i]);
        printf("%u]\n", yShapeRSum[0]);
#endif // #if DEBUG_OUTPUT
#endif // #if OPTI_SHAPES
        uint32_t dataLength = yShapeRSum[shapeSize];
        if(shapeSize > 1 && x1Shape[0] != 1 && x2Shape[0] != 1) { // minibatch specialization
            PowsTilingDataBCWithMiniBatch tiling;
            ub_size -= 1024; // reserved for reminder handling
            uint32_t type_sz = GetSizeByDataType(context->GetInputTensor(0)->GetDataType());
            uint32_t mini_batch = MINI_BATCH_SIZE / type_sz;
            uint32_t ub_size_per_it;
            switch(context->GetInputTensor(0)->GetDataType()) {
                case DT_BF16:
                    ub_size_per_it = 2 * 4 + BUFFER_NUM * 3 * type_sz;
                    break;
                case DT_FLOAT:
                    ub_size_per_it = BUFFER_NUM * 3 * type_sz;
                    break;
                case DT_FLOAT16:
                    ub_size_per_it = 2 * 4 + BUFFER_NUM * 3 * type_sz;
                    break;
                default:
                    return GRAPH_FAILED;    // error
            }
            uint32_t tileLength = FLOOR(ub_size / ub_size_per_it, mini_batch);
            uint32_t tileNumber = yShape[0] / tileLength;
            uint32_t maxLength = tileNumber * tileLength;
            uint32_t reminder = CEIL(yShape[0] % tileLength, MINI_BATCH_SIZE / type_sz);
            uint32_t bigBatch = dataLength / yShape[0];
            SET(tileLength);
            SET(maxLength);
            SET(reminder);
            SET(bigBatch);
            SET(x1Shape);   
            SET(x2Shape);   
            SET(yShape);
            SET(shapeSize);
            SET(x1ShapeRSum);   
            SET(x2ShapeRSum);    
            SET(yShapeRSum);
            context->SetBlockDim(1);
            context->SetTilingKey(4);
#if DEBUG_OUTPUT
            printf("TilingFunc: Using MiniBatch Kernel\n");
            printf("TilingFunc: bigBatch: %u, tileLength: %u, tileNum: %u, reminder: %u\n", bigBatch, tileLength, tileNumber, reminder);
#endif
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        } else { // normal broadcast ver
            PowsTilingDataBC tiling;
            SET(x1Shape);   
            SET(x2Shape);   
            SET(yShape);
            SET(shapeSize);
            SET(x1ShapeRSum);   
            SET(x2ShapeRSum);    
            SET(yShapeRSum);
            SET(dataLength);
            context->SetBlockDim(1);
            if(GetSizeByDataType(context->GetInputTensor(0)->GetDataType()) == 2) context->SetTilingKey(2);
            else context->SetTilingKey(3);
#if DEBUG_OUTPUT
            printf("TilingFunc: Using Plain Broadcast Kernel\n");
#endif
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        }
#if DEBUG_OUTPUT
        printf("TilingFunc: Core Number: %u, UB Size: %u\n", core_num, ub_size);
        printf("TilingFunc: TK=%lu\n", context->GetTilingKey());
#endif
        delete[] x1Shape;
        delete[] x2Shape;
        delete[] yShape;
        delete[] x1ShapeRSum;
        delete[] x2ShapeRSum;
        delete[] yShapeRSum;
        return GRAPH_SUCCESS;
    }
}


namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context) {
        printf("Start to infer shape...\n");
        const auto *x1_shape = context->GetInputShape(0);
        const auto *x2_shape = context->GetInputShape(1);
        if(*x1_shape == *x2_shape) { // no need to broadcast
            *context->GetOutputShape(0) = *x1_shape;
            return GRAPH_SUCCESS;
        }
        auto y_dim = x1_shape->GetDimNum();
        if(x1_shape->GetDimNum() == x2_shape->GetDimNum()) { // same dim count
            context->GetOutputShape(0)->SetDimNum(y_dim);
            for(int i = 0; i < y_dim; i++) {
                if( x1_shape->GetDim(i) != x2_shape->GetDim(i) &&
                    x1_shape->GetDim(i) != 1 && 
                    x2_shape->GetDim(i) != 1) return GRAPH_FAILED; // Can NOT broadcast
                context->GetOutputShape(0)->SetDim(i, max(x1_shape->GetDim(i), x2_shape->GetDim(i)));
            }
            return GRAPH_SUCCESS;
        }
        gert::Shape broadcast_shape{};
        broadcast_shape.SetDimNum(max(x1_shape->GetDimNum(), x2_shape->GetDimNum())); // different Dim count
        for(int x1 = x1_shape->GetDimNum() - 1, x2 = x2_shape->GetDimNum() - 1, y = broadcast_shape.GetDimNum() - 1;
            y >= 0;
            x1--, x2--, y--) {
                if(x1 < 0) broadcast_shape.SetDim(y, x2_shape->GetDim(x2));
                else if(x2 < 0) broadcast_shape.SetDim(y, x1_shape->GetDim(x1));
                if( x1_shape->GetDim(x1) != x2_shape->GetDim(x2) &&
                    x1_shape->GetDim(x1) != 1 && 
                    x2_shape->GetDim(x2) != 1) return GRAPH_FAILED; // Can NOT broadcast
                broadcast_shape.SetDim(y, max(x1_shape->GetDim(x1), x2_shape->GetDim(x2)));
        }
        *context->GetOutputShape(0) = broadcast_shape;
        return GRAPH_SUCCESS;
    }
}


namespace ops {
class Pows : public OpDef {
public:
    explicit Pows(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Pows);
}
