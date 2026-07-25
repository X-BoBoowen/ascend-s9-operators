
#include "glu_grad_jvp_tiling.h"
#include "register/op_def_registry.h"
#include <iostream>
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    GluGradJvpTilingData tiling;
    const auto runtime_attrs = context->GetAttrs();
    int dim = *(runtime_attrs->GetInt(0));

    // 完整的输入形状
    const auto inputShape = context->GetInputTensor(0)->GetOriginShape();

    uint16_t mmInputDims[8];
    int nInputDims = inputShape.GetDimNum();
    int inputSize = 1;
    for (int i = 0; i < inputShape.GetDimNum(); i++) {
        mmInputDims[i] = inputShape.GetDim(i);
        inputSize *= mmInputDims[i];
    }
    if (dim < 0) {
        dim += nInputDims;
    }
    int KS = 1;
    for (int i = nInputDims - 1; i > dim; i--) {
        KS *= mmInputDims[i];
    }
    int HI = inputSize / KS / mmInputDims[dim];
    int J = mmInputDims[dim];

    tiling.set_HI(HI);
    tiling.set_KS(KS);
    tiling.set_J(J);

    // 以y_grad大小为基准
    uint32_t size = context->GetInputTensor(1)->GetShapeSize();
    uint32_t aivNum = 40; // Ascend910B
    auto dt = context->GetInputTensor(0)->GetDataType(); //  ge::DT_FLOAT
    int DataTypeSize = 0;
    if (dt == ge::DT_FLOAT) {
        DataTypeSize = 4;
    } else {
        DataTypeSize = 2;
    }
    // blockSize以byte为单位
    uint32_t blockSize = J * KS / 2 * DataTypeSize; // 交错排放
    uint32_t blockNum = (size * DataTypeSize + blockSize - 1) / blockSize;
    aivNum = std::min(aivNum, blockNum);
    // std::cout << "GluGradJvpTiling: HI=" << HI << ", J=" << J << ", KS=" << KS
    //       << ", size=" << size << ", blockSize=" << blockSize
    //       << ", blockNum=" << blockNum << ", aivNum=" << aivNum
    //       << ", DataTypeSize=" << DataTypeSize << std::endl;
    uint32_t smallSize = blockNum / aivNum * blockSize / DataTypeSize;
    uint32_t incSize = blockSize / DataTypeSize;
    uint16_t formerNum = blockNum % aivNum;
    tiling.set_smallSize(smallSize);
    tiling.set_incSize(incSize);
    tiling.set_formerNum(formerNum);

    if (dt == ge::DT_FLOAT) {
        uint32_t hi_block_size = J * KS / 2;
        if (hi_block_size <= 2 * 1024) {
            context->SetTilingKey(3);
        } else {
            context->SetTilingKey(2);
        }
        context->SetBlockDim(aivNum);
    } else if (dt == ge::DT_BF16 || dt == ge::DT_FLOAT16) {
        context->SetTilingKey(1);
        context->SetBlockDim(aivNum);
    }

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class GluGradJvp : public OpDef {
public:
    explicit GluGradJvp(const char* name) : OpDef(name) {
        this->Input("x_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("v_y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("v_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("jvp_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(-1);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(GluGradJvp);
} // namespace ops
