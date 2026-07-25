
#include "copysign_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <iostream>
#include <vector>


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    CopysignTilingData tiling;
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    const gert::StorageShape* x2_shape = context->GetInputShape(1);
    auto dim1 = x1_shape->GetStorageShape().GetDimNum();
    auto dim2 = x2_shape->GetStorageShape().GetDimNum();
    
    int n1[MAX_DIM_NUMBER], n2[MAX_DIM_NUMBER], shape[MAX_DIM_NUMBER];
    for (int i = 0; i < MAX_DIM_NUMBER; ++i) n1[i] = n2[i] = 1;
    for (int i = 0; i < dim1; ++i) {
        n1[i + (MAX_DIM_NUMBER - dim1)] = x1_shape->GetStorageShape().GetDim(i);
    }
    for (int i = 0; i < dim2; ++i) {
        n2[i + (MAX_DIM_NUMBER - dim2)] = x2_shape->GetStorageShape().GetDim(i);
    }
    int32_t size = 1, size1 = 1, size2 = 1;
    for (int i = 0; i < MAX_DIM_NUMBER; ++i) {
        shape[i] = std::max(n1[i], n2[i]);
        size *= std::max(n1[i], n2[i]);
        size1 *= n1[i]; size2 *= n2[i];
    }
    tiling.set_size(size);
    tiling.set_shape(shape);
    
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    int32_t sizeofdatatype = 1;
    auto dt = context->GetInputTensor(0)->GetDataType();
    if (dt == ge::DT_INT16) sizeofdatatype = 2;
    else if (dt == ge::DT_INT32) sizeofdatatype = 4;
    else if (dt == ge::DT_INT64) sizeofdatatype = 8;
    auto num_cores = aivNum;
    if (size1 == size2)
    {
        if (size <= 10000)
        {
            context->SetTilingKey(1);
            const int32_t alignment = 64 / sizeofdatatype;
            unsigned length = (size - 1) / num_cores + 1;
            while (length % alignment != 0) length += 1;
            tiling.set_length(length);
        }
        else
        {
            context->SetTilingKey(2);
            const int32_t alignment = 32 / sizeofdatatype;
            unsigned length = (size - 1) / num_cores + 1;
            while (length % alignment != 0) length += 1;
            tiling.set_length(length);
        }
    }
    else if (n1[MAX_DIM_NUMBER - 1] == n2[MAX_DIM_NUMBER - 1] && n1[MAX_DIM_NUMBER - 1] <= 3000)
    {
        context->SetTilingKey(3);
        auto sizetemp = size / n1[MAX_DIM_NUMBER - 1];
        int length = (sizetemp - 1) / num_cores + 1;
        length *= n1[MAX_DIM_NUMBER - 1];
        tiling.set_length(length);
    }
    else
    {
        context->SetTilingKey(4);
        const int32_t alignment = 64 / sizeofdatatype;
        unsigned length = (size - 1) / num_cores + 1;
        while (length % alignment != 0) length += 1;
        tiling.set_length(length);
    }

    context->SetBlockDim(num_cores);

    for (int i = MAX_DIM_NUMBER - 2; i >= 0; i--)
    {
        n1[i] *= n1[i+1];
        n2[i] *= n2[i+1];
    }
    for (int i = MAX_DIM_NUMBER - 1; i > 0; i--)
    {
        if (n1[i-1] == n1[i]) n1[i] = 0;
        if (n2[i-1] == n2[i]) n2[i] = 0;
    }
    for (int i = 0; i < MAX_DIM_NUMBER - 1; i++) n1[i] = n1[i+1], n2[i] = n2[i+1];
    n1[MAX_DIM_NUMBER - 1] = n2[MAX_DIM_NUMBER - 1] = 1;
    tiling.set_n1(n1);
    tiling.set_n2(n2);
    // uint64_t ub_size;
    // ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    // printf("%d\n", ub_size);
    // ub:196352 Byte

    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
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
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class Copysign : public OpDef {
public:
    explicit Copysign(const char* name) : OpDef(name)
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

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Copysign);
}
