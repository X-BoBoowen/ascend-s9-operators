
#include "fractional_max_pool3_d_tiling.h"
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

static void calcStep(int iT, int iH, int iW, int oT, int oH, int oW, int poolSizeT, int poolSizeH, int poolSizeW,
                     uint32_t &tStep, uint32_t &hStep, uint32_t &wStep)
{
    tStep = 1;
    hStep = 1;
    wStep = oW;
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    FractionalMaxPool3DTilingData tiling;
    const gert::StorageShape *x1_shape = context->GetInputShape(0);
    const gert::StorageShape *z_shape = context->GetOutputShape(0);

    // const int64_t *pkernel_size = context->GetAttrs()->GetInt(0);
    auto *pkernel_sizecv = context->GetAttrs()->GetListInt(0);
    const int64_t *pkernel_size = pkernel_sizecv->GetData();
    int kernel_size_list_size = pkernel_sizecv->GetSize();
    int kernel_size[3];
    if (kernel_size_list_size != 3) {
        kernel_size[0] = pkernel_size[0];
        kernel_size[1] = pkernel_size[0];
        kernel_size[2] = pkernel_size[0];
    } else {
        kernel_size[0] = pkernel_size[0];
        kernel_size[1] = pkernel_size[1];
        kernel_size[2] = pkernel_size[2];
    }

    auto *poutput_sizecv = context->GetAttrs()->GetListInt(1);
    const int64_t *poutput_size = poutput_sizecv->GetData();

    const float *poutput_ratio_t = context->GetAttrs()->GetFloat(2);
    const float *poutput_ratio_h = context->GetAttrs()->GetFloat(3);
    const float *poutput_ratio_w = context->GetAttrs()->GetFloat(4);
    const bool *preturn_indices = context->GetAttrs()->GetBool(5);

    uint32_t N, C, T, H, W;
    int dimnum = x1_shape->GetStorageShape().GetDimNum();
    if (dimnum == 5) {
        N = x1_shape->GetStorageShape().GetDim(0);
        C = x1_shape->GetStorageShape().GetDim(1);
        T = x1_shape->GetStorageShape().GetDim(2);
        H = x1_shape->GetStorageShape().GetDim(3);
        W = x1_shape->GetStorageShape().GetDim(4);
    } else {
        N = 1;
        C = x1_shape->GetStorageShape().GetDim(0);
        T = x1_shape->GetStorageShape().GetDim(1);
        H = x1_shape->GetStorageShape().GetDim(2);
        W = x1_shape->GetStorageShape().GetDim(3);
    }

#if ENABLE_LOG
    std::cout << "N: " << N << " C: " << C << " T: " << T << " H: " << H << " W: " << W << std::endl;
    std::cout << "list size: " << kernel_size_list_size << " :kernel_size: " << pkernel_size[0] << " "
              << pkernel_size[1] << " " << pkernel_size[2] << std::endl;
    std::cout << "output_size: " << poutput_size[0] << " " << poutput_size[1] << " " << poutput_size[2] << std::endl;
    std::cout << "output_ratio: " << *poutput_ratio_t << " " << *poutput_ratio_h << " " << *poutput_ratio_w
              << std::endl;
    std::cout << "return_indices: " << *preturn_indices << std::endl;
#endif

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int block_num = ascendcPlatform.GetCoreNumAiv();
    int tilingKey = 0;
    if (*preturn_indices == false) {
        tilingKey = 1;
    }

    uint32_t poolSizeT = kernel_size[0];
    uint32_t poolSizeH = kernel_size[1];
    uint32_t poolSizeW = kernel_size[2];

    uint32_t outputT;
    uint32_t outputH;
    uint32_t outputW;
    // if(*poutput_ratio_t==0.0){
    //     outputT = poutput_size[0];
    //     outputH = poutput_size[1];
    //     outputW = poutput_size[2];
    // }else{
    //     outputT = T * *poutput_ratio_t;
    //     outputH = H * *poutput_ratio_h;
    //     outputW = W * *poutput_ratio_w;
    // }
    {
        int odimnum = z_shape->GetStorageShape().GetDimNum();
        if (odimnum == 5) {
            outputT = z_shape->GetStorageShape().GetDim(2);
            outputH = z_shape->GetStorageShape().GetDim(3);
            outputW = z_shape->GetStorageShape().GetDim(4);
        } else {
            outputT = z_shape->GetStorageShape().GetDim(1);
            outputH = z_shape->GetStorageShape().GetDim(2);
            outputW = z_shape->GetStorageShape().GetDim(3);
        }
    }


    uint32_t formerNum;
    uint32_t formerLength;
    uint32_t tailNum;
    uint32_t tailLength;
    {
        int total_block = N * C;
        if (total_block < block_num) {
            block_num = total_block;

            formerNum = block_num;
            formerLength = 1;
            tailNum = 0;
            tailLength = 0;
        } else {
            int block_size = total_block / block_num;

            formerNum = (total_block % block_num) == 0 ? block_num : (total_block % block_num);
            tailNum = block_num - formerNum;
            tailLength = block_size;
            formerLength = block_size + (tailNum == 0 ? 0 : 1);
        }
    }


    uint32_t tStep;
    uint32_t hStep;
    uint32_t wStep;

    // TODO: 计算tStep, hStep, wStep
    calcStep(T, H, W, outputT, outputH, outputW, poolSizeT, poolSizeH, poolSizeW, tStep, hStep, wStep);

    tiling.set_formerNum(formerNum);
    tiling.set_formerLength(formerLength);
    tiling.set_tailNum(tailNum);
    tiling.set_tailLength(tailLength);

    tiling.set_tStep(tStep);
    tiling.set_hStep(hStep);
    tiling.set_wStep(wStep);

    tiling.set_poolSizeT(poolSizeT);
    tiling.set_poolSizeH(poolSizeH);
    tiling.set_poolSizeW(poolSizeW);

    tiling.set_outputT(outputT);
    tiling.set_outputH(outputH);
    tiling.set_outputW(outputW);

    tiling.set_inputT(T);
    tiling.set_inputH(H);
    tiling.set_inputW(W);

#if ENABLE_LOG
    std::cout << "block_num: " << block_num << std::endl;
    std::cout << "tilingKey: " << tilingKey << std::endl;
    std::cout << "poolSizeT: " << poolSizeT << " poolSizeH: " << poolSizeH << " poolSizeW: " << poolSizeW << std::endl;
    std::cout << "outputT: " << outputT << " outputH: " << outputH << " outputW: " << outputW << std::endl;
    std::cout << "inputT: " << T << " inputH: " << H << " inputW: " << W << std::endl;
    std::cout << "formerNum: " << formerNum << " formerLength: " << formerLength << " tailNum: " << tailNum
              << " tailLength: " << tailLength << std::endl;
    std::cout << "tStep: " << tStep << " hStep: " << hStep << " wStep: " << wStep << std::endl;
#endif

    //   tiling.set_size(data_sz);
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
    gert::Shape *z_shape = context->GetOutputShape(0);
    // *z_shape = *x1_shape; //TODO:
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
class FractionalMaxPool3D : public OpDef {
public:
    explicit FractionalMaxPool3D(const char *name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
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
        this->Output("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("kernel_size").ListInt();
        this->Attr("output_size").AttrType(OPTIONAL).ListInt({});
        this->Attr("output_ratio_t").AttrType(OPTIONAL).Float(0.0);
        this->Attr("output_ratio_h").AttrType(OPTIONAL).Float(0.0);
        this->Attr("output_ratio_w").AttrType(OPTIONAL).Float(0.0);
        this->Attr("return_indices").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(FractionalMaxPool3D);
} // namespace ops
