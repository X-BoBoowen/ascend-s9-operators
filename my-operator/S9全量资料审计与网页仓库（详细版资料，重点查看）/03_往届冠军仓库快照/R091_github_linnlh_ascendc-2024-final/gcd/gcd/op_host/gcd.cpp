
#include "gcd_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

using namespace platform_ascendc;

constexpr int BUFFER_NUM = 1;


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto platform = PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize;
    platform.GetCoreMemSize(CoreMemType::UB, ubSize);
    auto coreNum = platform.GetCoreNum();
    uint32_t dtypeSize = 0;
    ge::TypeUtils::GetDataTypeLength(
        context->GetInputDesc(0)->GetDataType(),
        dtypeSize
    );


    GcdTilingData tiling;
    auto x1Shape = context->GetInputShape(0)->GetStorageShape();
    auto x2Shape = context->GetInputShape(1)->GetStorageShape();
    int64_t x1ShapePtr[10];
    int64_t x2ShapePtr[10];
    int64_t yShapePtr[10];
    int64_t dimNum1 = x1Shape.GetDimNum();
    int64_t dimNum2 = x2Shape.GetDimNum();
    int64_t dimNum = std::max(dimNum1, dimNum2);
    if (dimNum1 >= dimNum2) {
        for (int i = 0; i < (dimNum1 - dimNum2); i++) {
            x1ShapePtr[i] = x1Shape[i];
            x2ShapePtr[i] = 1;
            yShapePtr[i] = x1ShapePtr[i];
        }
        for (int i = (dimNum1 - dimNum2); i < dimNum1; i++) {
            x1ShapePtr[i] = x1Shape[i];
            x2ShapePtr[i] = x2Shape[i - dimNum1 + dimNum2];
            yShapePtr[i] = std::max(x1ShapePtr[i], x2ShapePtr[i]);
        }
    }
    else {
        for (int i = 0; i < (dimNum2 - dimNum1); i++) {
            x1ShapePtr[i] = 1;
            x2ShapePtr[i] = x2Shape[i];
            yShapePtr[i] = x2ShapePtr[i];
        }
        for (int i = (dimNum2 - dimNum1); i < dimNum2; i++) {
            x1ShapePtr[i] = x1Shape[i - dimNum2 + dimNum1];
            x2ShapePtr[i] = x2Shape[i];
            yShapePtr[i] = std::max(x1ShapePtr[i], x2ShapePtr[i]);
        }
    }

    uint32_t totalLen = 1;
    for (int i = 0; i < dimNum; i++) {
        totalLen *= yShapePtr[i];
    }

    // 切分大小核数据
    if (totalLen < coreNum) {
        coreNum = totalLen;
    }
    

    uint32_t tileLen = ubSize / coreNum / dtypeSize;
    uint32_t smallCoreLen = totalLen / coreNum;
    uint32_t smallCoreTileLen = tileLen;
    if (smallCoreTileLen > smallCoreLen) {
        smallCoreTileLen = smallCoreLen;
    }
    uint32_t smallCoreTileNum = smallCoreLen / smallCoreTileLen;
    uint32_t smallCoreTailTileLen = smallCoreLen % smallCoreTileLen;

    
    uint32_t remainLen = totalLen % coreNum;
    uint32_t bigCoreLen = smallCoreLen + 1;
    uint32_t bigCoreNum = remainLen;
    if (remainLen == 0) {
        bigCoreLen -= 1;
        bigCoreNum = coreNum;
    }
    uint32_t bigCoreTileLen = tileLen;
    if (bigCoreTileLen > bigCoreLen) {
        bigCoreTileLen = bigCoreLen;
    }
    uint32_t bigCoreTileNum = bigCoreLen / bigCoreTileLen;
    uint32_t bigCoreTailTileLen = bigCoreLen % bigCoreTileLen;

    tiling.set_x1Shape(x1ShapePtr);
    tiling.set_x2Shape(x2ShapePtr);
    tiling.set_yShape(yShapePtr);
    tiling.set_shapeDim(dimNum);

    tiling.set_smallCoreLen(smallCoreLen);
    tiling.set_smallCoreTileLen(smallCoreTileLen);
    tiling.set_smallCoreTileNum(smallCoreTileNum);
    tiling.set_smallCoreTailTileLen(smallCoreTailTileLen);
    tiling.set_bigCoreLen(bigCoreLen);
    tiling.set_bigCoreTileLen(bigCoreTileLen);
    tiling.set_bigCoreTileNum(bigCoreTileNum);
    tiling.set_bigCoreTailTileLen(bigCoreTailTileLen);
    tiling.set_bigCoreNum(bigCoreNum);

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity()
    );
    context->SetBlockDim(coreNum);
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
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
class Gcd : public OpDef {
public:
    explicit Gcd(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Gcd);
}
