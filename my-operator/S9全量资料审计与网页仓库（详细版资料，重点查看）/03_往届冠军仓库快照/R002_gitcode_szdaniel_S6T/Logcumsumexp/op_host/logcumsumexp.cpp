
#include "logcumsumexp_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define BUFFER_NUM 2
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#include <iostream>
#include <fstream>

#define ENABLE_LOG 0 // 1

// return new dimaxis
static inline int mergeShape(const gert::StorageShape *x_shape, int &newShape0, uint32_t dim, uint32_t &ADimLength,
                             uint32_t &PDimLength)
{
    int xShape[4];
    int dimnum = x_shape->GetStorageShape().GetDimNum();

    for (int i = 0; i < dimnum; i++) {
        xShape[i] = x_shape->GetStorageShape().GetDim(i);
    }

#if ENABLE_LOG
    std::cout << "dimnum:" << dimnum << std::endl;
    for (int i = 0; i < dimnum; i++) {
        std::cout << "xShape[" << i << "]:" << xShape[i] << std::endl;
    }
    std::cout << "dim:" << dim << std::endl;
#endif

    if (dimnum == 1) {
        newShape0 = xShape[0];
        ADimLength = xShape[0];
        PDimLength = 1;
    } else if (dim > 0) {
        newShape0 = 1;
        for (int i = 0; i < dim; i++) {
            newShape0 *= xShape[i];
        }
        ADimLength = xShape[dim];

        PDimLength = 1;
        for (int i = dim + 1; i < dimnum; i++) {
            PDimLength *= xShape[i];
        }
    } else {
        newShape0 = xShape[0];
        ADimLength = xShape[0];
        PDimLength = 1;
        for (int i = 1; i < dimnum; i++) {
            PDimLength *= xShape[i];
        }
    }

    if (dim == 0) {
        return 0;
    } else {
        return 1;
    }
}

static inline void cut_1_dim(int totalLen, int maxTileLen, uint32_t &tileNum, uint32_t &tileLength,
                             uint32_t &lastTileLength)
{
    tileLength = MIN(maxTileLen, totalLen);
    tileNum = CEIL_DIV(totalLen, tileLength);
    lastTileLength = totalLen - (tileNum - 1) * tileLength;
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    const int64_t *dim = attrs->GetInt(0);
    int64_t dim_value = *dim;

    uint32_t formerNum;
    uint32_t formerLength;
    uint32_t formerTileNum;
    uint32_t formerTileLength;
    uint32_t formerLastTileLength;

    uint32_t tailNum;
    uint32_t tailLength;
    uint32_t tailTileNum;
    uint32_t tailTileLength;
    uint32_t tailLastTileLength;

    uint32_t ADimLength;
    uint32_t PDimLength;

    LogcumsumexpTilingData tiling;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int block_num = ascendcPlatform.GetCoreNumAiv();
#if ENABLE_LOG
    std::cout << "block_num:" << block_num << std::endl;
#endif


    int tilingKey = 0;

    const gert::StorageShape *shape = context->GetInputShape(0);

    int newShape0 = 0;
    int dimaxis = mergeShape(context->GetInputShape(0), newShape0, dim_value, ADimLength, PDimLength);
#if ENABLE_LOG
    std::cout << "dimaxis:" << dimaxis << std::endl;
    std::cout << "newShape0:" << newShape0 << std::endl;
    std::cout << "ADimLength:" << ADimLength << std::endl;
    std::cout << "PDimLength:" << PDimLength << std::endl;
#endif

    int max_tile_len = 2048;
    if (dimaxis == 0) {
        block_num = 1;
        formerNum = 1;
        tailNum = 0;
        tailLength = 0;
        formerLength = 1;
    } else {
        int block_size = newShape0 / block_num;
        formerNum = (newShape0 % block_num) == 0 ? block_num : (newShape0 % block_num);
        tailNum = block_num - formerNum;
        tailLength = block_size;
        formerLength = block_size + (tailNum == 0 ? 0 : 1);
    }


    cut_1_dim(PDimLength, max_tile_len, formerTileNum, formerTileLength, formerLastTileLength);
    cut_1_dim(PDimLength, max_tile_len, tailTileNum, tailTileLength, tailLastTileLength);


    tilingKey = 0;
    if (PDimLength == 1) {
        tilingKey = 1;
    }

    tiling.set_ADimLength(ADimLength);
    tiling.set_PDimLength(PDimLength);

    tiling.set_formerNum(formerNum);
    tiling.set_formerLength(formerLength);
    tiling.set_formerTileNum(formerTileNum);
    tiling.set_formerTileLength(formerTileLength);
    tiling.set_formerLastTileLength(formerLastTileLength);

    tiling.set_tailNum(tailNum);
    tiling.set_tailLength(tailLength);
    tiling.set_tailTileNum(tailTileNum);
    tiling.set_tailTileLength(tailTileLength);
    tiling.set_tailLastTileLength(tailLastTileLength);

    context->SetTilingKey(tilingKey);
#if ENABLE_LOG
    std::cout << "final block_num:" << block_num << std::endl;
    std::cout << "tilingKey: " << tilingKey << std::endl;
    std::cout << "formerNum:" << formerNum << ", formerLength:" << formerLength << std::endl;
    std::cout << "tailNum:" << tailNum << ", tailLength:" << tailLength << std::endl;
    std::cout << "formerTileNum:" << formerTileNum << std::endl;
    std::cout << "formerTileLength:" << formerTileLength << ", formerLastTileLength:" << formerLastTileLength
              << std::endl;
    std::cout << "tailTileNum:" << tailTileNum << std::endl;
    std::cout << "tailTileLength:" << tailTileLength << ", tailLastTileLength:" << tailLastTileLength << std::endl;
#endif

    context->SetBlockDim(block_num); //
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());


    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge


namespace ops {
class Logcumsumexp : public OpDef {
public:
    explicit Logcumsumexp(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Logcumsumexp);
} // namespace ops
