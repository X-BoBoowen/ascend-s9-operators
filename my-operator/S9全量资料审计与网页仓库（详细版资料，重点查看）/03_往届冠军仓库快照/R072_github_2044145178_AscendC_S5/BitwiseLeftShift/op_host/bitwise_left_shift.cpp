
#include "bitwise_left_shift_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    BitwiseLeftShiftTilingData tiling;
    const auto shape0 = context->GetInputTensor(0)->GetOriginShape();
    const auto shape1 = context->GetInputTensor(1)->GetOriginShape();

    uint8_t nOutputDims = std::max(shape0.GetDimNum(), shape1.GetDimNum());
    uint16_t arr0[8];
    for (int i = 0; i < nOutputDims; i++) {
        if (shape0.GetDimNum() + i >= nOutputDims) {
            arr0[i] = shape0.GetDim(shape0.GetDimNum() - nOutputDims + i);
        } else {
            arr0[i] = 1;
        }
    }
    tiling.set_mmInputDims(arr0);
    uint16_t arr1[8];
    for (int i = 0; i < nOutputDims; i++) {
        if (shape1.GetDimNum() + i >= nOutputDims) {
            arr1[i] = shape1.GetDim(shape1.GetDimNum() - nOutputDims + i);
        } else {
            arr1[i] = 1;
        }
    }
    tiling.set_mmOtherDims(arr1);
    uint16_t arr[8];
    uint32_t outputSize = 1;
    bool isBroadcast = false;
    for (int i = 0; i < nOutputDims; i++) {
        if (arr0[i] != arr1[i]) {
            isBroadcast = true;
        }
        arr[i] = std::max(arr0[i], arr1[i]);
        outputSize *= arr[i];
    }
    tiling.set_totalSize(outputSize);
    tiling.set_nOutputDims(nOutputDims);
    tiling.set_mmOutputDims(arr);

    uint32_t size = outputSize;
    uint32_t aivNum = 40; // Ascend910B
    auto dt = context->GetInputTensor(0)->GetDataType(); //  ge::DT_FLOAT
    int DataTypeSize = 0;
    if (dt == ge::DT_INT64) {
        DataTypeSize = 8;
    } else if (dt == ge::DT_INT32) {
        DataTypeSize = 4;
    } else if (dt == ge::DT_INT16) {
        DataTypeSize = 2;
    } else if (dt == ge::DT_INT8) {
        DataTypeSize = 1;
    }

    if (!isBroadcast) {
        // 非广播模式

        context->SetTilingKey(1);

        // if (size <= 128 * 1024) {
        //     aivNum = 20;
        // }
    } else {
        // 广播模式
        int inputSize = 1;
        int otherSize = 1;
        int totalSize = outputSize;
        for (int i = 0; i < nOutputDims; i++) {
            inputSize *= arr0[i];
            otherSize *= arr1[i];
        }
        if (inputSize != totalSize && inputSize > 5 * 1024) {
            context->SetTilingKey(2);
        } else if (otherSize != totalSize && otherSize > 5 * 1024) {
            context->SetTilingKey(2);
        } else if (DataTypeSize != 8 || nOutputDims != 2 || totalSize < 32*1024) {
            context->SetTilingKey(2);
        } else {
            context->SetTilingKey(3);
        }
    }

    // blockSize以byte为单位
    uint32_t blockSize = 1024;
    uint32_t blockNum = (size * DataTypeSize + blockSize - 1) / blockSize;
    aivNum = std::min(aivNum, blockNum);

    uint32_t smallSize = blockNum / aivNum * blockSize / DataTypeSize;
    uint32_t incSize = blockSize / DataTypeSize;
    uint16_t formerNum = blockNum % aivNum;
    tiling.set_smallSize(smallSize);
    tiling.set_incSize(incSize);
    tiling.set_formerNum(formerNum);
    context->SetBlockDim(aivNum);

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
class BitwiseLeftShift : public OpDef {
public:
    explicit BitwiseLeftShift(const char* name) : OpDef(name) {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("other")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(BitwiseLeftShift);
} // namespace ops
