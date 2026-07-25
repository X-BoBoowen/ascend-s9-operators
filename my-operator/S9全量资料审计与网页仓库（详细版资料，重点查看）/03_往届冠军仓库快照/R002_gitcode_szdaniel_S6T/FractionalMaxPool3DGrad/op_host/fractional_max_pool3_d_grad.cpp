

#include "fractional_max_pool3_d_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define BUFFER_NUM 2
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define ROUND_UP(a, b) (((a) + (b) - 1) / (b) * (b))
#define ROUND_DOWN(a, b) ((a) / (b) * (b))
#include <iostream>
#include <fstream>

#define ENABLE_LOG 0 // 0

static inline uint32_t find_former_len(uint32_t block_byte, uint32_t NC)
{
    uint32_t a = 1;
    while ((block_byte % 2 == 0) && (a < 64)) {
        a *= 2;
        block_byte /= 2;
        // std::cout << "a: " << a << ",b:" << block_byte << std::endl;
    }
    a = 64 / a;
    if (a > NC || a == 64)
        return NC;
    else
        return a;
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    FractionalMaxPool3DGradTilingData tiling;
    const gert::StorageShape *x1_shape = context->GetInputShape(0);
    const gert::StorageShape *x2_shape = context->GetInputShape(1);

    uint32_t N, C, TO, HO, WO, TI, HI, WI;
    int dimnum = x1_shape->GetStorageShape().GetDimNum();
    if (dimnum == 5) {
        N = x1_shape->GetStorageShape().GetDim(0);
        C = x1_shape->GetStorageShape().GetDim(1);
        TO = x1_shape->GetStorageShape().GetDim(2);
        HO = x1_shape->GetStorageShape().GetDim(3);
        WO = x1_shape->GetStorageShape().GetDim(4);

        TI = x2_shape->GetStorageShape().GetDim(2);
        HI = x2_shape->GetStorageShape().GetDim(3);
        WI = x2_shape->GetStorageShape().GetDim(4);
    } else {
        N = 1;
        C = x1_shape->GetStorageShape().GetDim(0);
        TO = x1_shape->GetStorageShape().GetDim(1);
        HO = x1_shape->GetStorageShape().GetDim(2);
        WO = x1_shape->GetStorageShape().GetDim(3);

        TI = x2_shape->GetStorageShape().GetDim(1);
        HI = x2_shape->GetStorageShape().GetDim(2);
        WI = x2_shape->GetStorageShape().GetDim(3);
    }


    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int block_num = ascendcPlatform.GetCoreNumAiv();
    if (N * C < block_num)
        block_num = N * C;


    int block_size = N * C / block_num;
    int formerNum = (N * C % block_num) == 0 ? block_num : (N * C % block_num);
    int tailNum = block_num - formerNum;
    int tailLength = block_size;
    int formerLength = block_size + (tailNum == 0 ? 0 : 1);


    // auto dt = context->GetInputTensor(0)->GetDataType();
    // uint32_t dataWidth = 4;
    // if ((dt == ge::DT_BF16) || (dt == ge::DT_FLOAT16)) {
    //     dataWidth = 2;
    // } else if ((dt == ge::DT_FLOAT)) {
    //     dataWidth = 4;
    // }
    // if (TI * HI * WI * dataWidth % 64 != 0) {
    //     int NC = N * C;
    //     formerLength = find_former_len(TI * HI * WI * dataWidth, NC);
    //     formerNum = NC / formerLength;
    //     tailLength = NC - formerLength * formerNum;
    //     tailNum = tailLength > 0 ? 1 : 0;
    //     block_num = formerNum + tailNum;
    // }


    int tilingKey = 0;

#if ENABLE_LOG
    std::cout << "N: " << N << std::endl;
    std::cout << "C: " << C << std::endl;
    std::cout << "TO: " << TO << std::endl;
    std::cout << "HO: " << HO << std::endl;
    std::cout << "WO: " << WO << std::endl;
    std::cout << "TI: " << TI << std::endl;
    std::cout << "HI: " << HI << std::endl;
    std::cout << "WI: " << WI << std::endl;
    std::cout << "block_num: " << block_num << std::endl;
    std::cout << "tilingKey: " << tilingKey << std::endl;
    std::cout << "formerLength: " << formerLength << std::endl;
    std::cout << "formerNum: " << formerNum << std::endl;
    std::cout << "tailLength: " << tailLength << std::endl;
    std::cout << "tailNum: " << tailNum << std::endl;
#endif

    tiling.set_N(N);
    tiling.set_C(C);
    tiling.set_TO(TO);
    tiling.set_HO(HO);
    tiling.set_WO(WO);
    tiling.set_TI(TI);
    tiling.set_HI(HI);
    tiling.set_WI(WI);

    tiling.set_formerLength(formerLength);
    tiling.set_formerNum(formerNum);
    tiling.set_tailLength(tailLength);
    tiling.set_tailNum(tailNum);

    context->SetBlockDim(block_num);
    context->SetTilingKey(tilingKey);


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
class FractionalMaxPool3DGrad : public OpDef {
public:
    explicit FractionalMaxPool3DGrad(const char *name) : OpDef(name)
    {
        this->Input("grad_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("random_sample")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("kernel_size").ListInt();
        this->Attr("output_size").AttrType(OPTIONAL).ListInt({});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(FractionalMaxPool3DGrad);
} // namespace ops
