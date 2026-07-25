#include "select_v2_tiling.h"
#include "host_inc.h"

constexpr uint64_t MINI_BATCH_SIZE = 256;

#define OPTI_SHAPES 1
template<class Tp>
uint32_t set_remove(Tp *cond, Tp *x1, Tp *x2, Tp *y, const uint32_t size, uint32_t idx, const uint32_t count, const Tp &val) {
    if (idx >= size) return size;
    cond[idx] = x1[idx] = x2[idx] = y[idx] = val;
    idx += count;
    while(idx < size) {
        cond[idx - count + 1] = cond[idx];
        x1[idx - count + 1] = x1[idx];
        x2[idx - count + 1] = x2[idx];
        y[idx - count + 1] = y[idx];
        idx++;
    }
    return size - count + 1;
}
template<class Tp>
std::tuple<bool, uint32_t, uint32_t, Tp> check_arrs(const Tp *cond, const Tp *x1, const Tp *x2, const uint32_t size) {
    for(uint32_t i = 0; i < size - 1; i++) 
        if( x1[i] == x2[i] && cond[i] == x2[i] && 
            x1[i + 1] == x2[i + 1] && cond[i + 1] == x2[i + 1])
            return {true, i, 2, x1[i] * x1[i + 1]};
    return { false, 0, 0, Tp{} };
}

namespace optiling {
    constexpr uint32_t BUFFER_NUM = 2;
    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        auto dev_info = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());              // AscendC Platform Info
        context->SetBlockDim(1);
        const auto cond_shape = context->GetInputShape(0)->GetStorageShape();
        const auto x1_shape = context->GetInputShape(1)->GetStorageShape();
        const auto x2_shape = context->GetInputShape(2)->GetStorageShape();
        const auto y_shape = context->GetOutputShape(0)->GetStorageShape();
        bool need_broadcast = y_shape != x2_shape || y_shape != cond_shape || y_shape != x1_shape;
        if(!need_broadcast) {
            SelectV2TilingDataNoBC tiling;
            uint64_t ub_size; dev_info.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);      // Unified Buffer Size
            uint32_t type_sz = GetSizeByDataType(context->GetInputTensor(1)->GetDataType());
            uint32_t mini_batch = MINI_BATCH_SIZE / type_sz;
            uint32_t ub_size_per_it;
            switch(context->GetInputTensor(1)->GetDataType()) {
                case DT_INT8:
                    ub_size_per_it = 3 + 4 + BUFFER_NUM * sizeof(bool) + 3 * BUFFER_NUM * sizeof(int8_t);
                    break;
                case DT_FLOAT:
                case DT_INT32:
                    ub_size_per_it = 3 + BUFFER_NUM * sizeof(bool) + 3 * BUFFER_NUM * sizeof(int32_t);
                    break;
                case DT_FLOAT16:
                    ub_size_per_it = 3 + BUFFER_NUM * sizeof(bool) + 3 * BUFFER_NUM * sizeof(int16_t);
                    break;
                default:
                    ub_size_per_it = 0;     // error
                    break;
            }
            uint32_t tileLength = FLOOR(ub_size / ub_size_per_it, 512 / type_sz);
            SET(tileLength);
            context->SetTilingKey(1);
            uint32_t yLength = 1;
            for (int i = 0; i < y_shape.GetDimNum(); i++) 
                yLength  *= y_shape.GetDim(i);
            uint32_t tileNumber = yLength / tileLength;
            uint32_t reminder = CEIL(yLength % tileLength, mini_batch);
            SET(tileNumber);
            SET(reminder);
#if DEBUG_OUTPUT
            printf("TilingFunc: tileLength: %u, tileNum: %u, reminder: %u\n", tileLength, tileNumber, reminder);
#endif
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
            return GRAPH_SUCCESS;
        }                                                        
        uint32_t condShape[MAX_SHAPE_SIZE]{0}, x1Shape[MAX_SHAPE_SIZE]{0}, x2Shape[MAX_SHAPE_SIZE]{0}, yShape[MAX_SHAPE_SIZE]{0};
        uint32_t condShapeRSum[MAX_SHAPE_SIZE + 1]{0}, x1ShapeRSum[MAX_SHAPE_SIZE + 1]{0}, x2ShapeRSum[MAX_SHAPE_SIZE + 1]{0}, yShapeRSum[MAX_SHAPE_SIZE + 1]{0};
        uint32_t condShapeSize = cond_shape.GetDimNum();
        uint32_t x1ShapeSize = x1_shape.GetDimNum();
        uint32_t x2ShapeSize = x2_shape.GetDimNum();
        uint32_t yShapeSize = y_shape.GetDimNum();
        uint32_t shapeSize = max(yShapeSize, max(x1ShapeSize, max(x2ShapeSize, condShapeSize)));
        condShapeRSum[0] = x1ShapeRSum[0] = x2ShapeRSum[0] = yShapeRSum[0] = 1;
        for (int i = 0; i < shapeSize; i++) {
            if (condShapeSize > i) condShape[condShapeSize - i - 1] = cond_shape.GetDim(i);
            else condShape[i] = 1;
            if (x1ShapeSize > i) x1Shape[x1ShapeSize - i - 1] = x1_shape.GetDim(i);
            else x1Shape[i] = 1;
            if (x2ShapeSize > i) x2Shape[x2ShapeSize - i - 1] = x2_shape.GetDim(i);
            else x2Shape[i] = 1;
            if (yShapeSize > i) yShape[yShapeSize - i - 1] = y_shape.GetDim(i);
            else yShape[i] = 1;
        }
        while(true) {
            const auto [flag, idx, count, val] = check_arrs(condShape, x1Shape, x2Shape, shapeSize);
            if(!flag) break;
            shapeSize = set_remove(condShape, x1Shape, x2Shape, yShape, shapeSize, idx, count, val);
        }
        uint32_t condLength = 1, x1Length = 1, x2Length = 1, yLength = 1;
        condShapeRSum[0] = x1ShapeRSum[0] = x2ShapeRSum[0] = yShapeRSum[0] = 1;
        for (int i = 1; i <= shapeSize; i++) {
            condLength *= condShape[i - 1];
            x1Length *= x1Shape[i - 1];
            x2Length *= x2Shape[i - 1];
            yLength  *= yShape[i - 1];
            condShapeRSum[i] = condLength;
            x1ShapeRSum[i] = x1Length;
            x2ShapeRSum[i] = x2Length;
            yShapeRSum[i] = yLength;
        }
        uint32_t dataLength = yShapeRSum[shapeSize];
        if(shapeSize > 1 && x1Shape[0] != 1 && x2Shape[0] != 1 && condShape[0] != 1) { // minibatch specialization
            uint64_t ub_size; dev_info.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);      // Unified Buffer Size
            uint32_t type_sz = GetSizeByDataType(context->GetInputTensor(1)->GetDataType());
            uint32_t mini_batch = MINI_BATCH_SIZE / type_sz;
            uint32_t ub_size_per_it;
            ub_size -= 10240;
            switch(context->GetInputTensor(1)->GetDataType()) {
                case DT_INT8:
                    ub_size_per_it = 3 + 4 + BUFFER_NUM * sizeof(bool) + 3 * BUFFER_NUM * sizeof(int8_t);
                    break;
                case DT_FLOAT:
                case DT_INT32:
                    ub_size_per_it = 3 + BUFFER_NUM * sizeof(bool) + 3 * BUFFER_NUM * sizeof(int32_t);
                    break;
                case DT_FLOAT16:
                    ub_size_per_it = 3 + BUFFER_NUM * sizeof(bool) + 3 * BUFFER_NUM * sizeof(int16_t);
                    break;
                default:
                    ub_size_per_it = 0;     // error
                    break;
            }
            uint32_t tileLength = FLOOR(ub_size / ub_size_per_it, 512 / type_sz);
            uint32_t tileNumber = yShape[0] / tileLength;
            uint32_t reminder = CEIL(yShape[0] % tileLength, mini_batch);
            uint32_t bigBatch = dataLength / yShape[0];
            SelectV2TilingDataBCWithMiniBatch tiling;
            context->SetTilingKey(3);      
            SET(condShape);
            SET(condShapeRSum);
            SET(x1Shape);   
            SET(x1ShapeRSum);
            SET(x2Shape);   
            SET(x2ShapeRSum);
            SET(yShape);    
            SET(yShapeRSum);
            SET(shapeSize);
            SET(tileLength);
            SET(tileNumber);
            SET(reminder);
            SET(bigBatch);
#if DEBUG_OUTPUT
            printf("TilingFunc: bigBatch: %u, tileLength: %u, tileNum: %u, reminder: %u\n", bigBatch, tileLength, tileNumber, reminder);
#endif
            context->SetBlockDim(1);
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        } else {
            SelectV2TilingDataBC  tiling;
            context->SetTilingKey(2);      
            SET(condShape);
            SET(condShapeRSum);
            SET(x1Shape);   
            SET(x1ShapeRSum);
            SET(x2Shape);   
            SET(x2ShapeRSum);
            SET(yShape);    
            SET(yShapeRSum);
            SET(shapeSize);
            SET(dataLength);
            context->SetBlockDim(1);
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        }
#if DEBUG_OUTPUT
        printf("TilingFunc: cond Shape: [");
        for(uint32_t i = shapeSize - 1; i > 0; i--) 
            printf("%u, ", condShape[i]);
        printf("%u], size: %u\n", condShape[0], shapeSize);
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
        printf("TilingFunc: cond Shape RSum: [");
        for(uint32_t i = shapeSize; i > 0; i--) 
            printf("%u, ", condShapeRSum[i]);
        printf("%u]\n", condShapeRSum[0]);
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
        printf("TK=%lu\n", context->GetTilingKey());
#endif
        return GRAPH_SUCCESS;
    }
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class SelectV2 : public OpDef {
public:
    explicit SelectV2(const char* name) : OpDef(name)
    {
        this->Input("condition")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(SelectV2);
}
