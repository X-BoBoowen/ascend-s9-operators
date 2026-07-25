
#include "lcm_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <iostream>
#include <algorithm>



namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    LcmTilingData tiling;
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    auto dim1 = x1_shape->GetStorageShape().GetDimNum();
    int32_t n1[3] = {1, 1, 1};
    for (int i = 0; i < dim1; ++i) {
        n1[i] = x1_shape->GetStorageShape().GetDim(i);
    }
    const gert::StorageShape* x2_shape = context->GetInputShape(1);
    auto dim2 = x2_shape->GetStorageShape().GetDimNum();
    int32_t n2[3] = {1, 1, 1};
    for (int i = 0; i < dim2; ++i) {
        n2[i + (dim1 - dim2)] = x2_shape->GetStorageShape().GetDim(i);
    }
    int32_t size = 1;
    for (int i = 0; i < 3; ++i) {
        size *= std::max(n1[i], n2[i]);
    }
    tiling.set_size(size);
    tiling.set_n1(n1);
    tiling.set_n2(n2);
    
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    auto num_cores = aivNum;
    //std::cout << "num_cores:" << num_cores << ' ' << ascendcPlatform.GetCoreNumAic() << ' ' << ascendcPlatform.GetCoreNumAiv() << std::endl;
    int32_t sizeofdatatype = 1;
    auto dt = context->GetInputTensor(0)->GetDataType();
    if (dt == ge::DT_INT16) sizeofdatatype = 2;
    else if (dt == ge::DT_INT32) sizeofdatatype = 4;
    else if (dt == ge::DT_INT64) sizeofdatatype = 8;
    const int32_t alignment = 64 / sizeofdatatype;
    unsigned length = (size - 1) / num_cores + 1;
    while (length % alignment != 0) length += 1;
    tiling.set_length(length);

    context->SetBlockDim(num_cores);

    int tag1 = 0, tag2 = 0;
    for (int i = 0; i < 3; i++)
    {
        tag1 = tag1 * 2 + (n1[i] != std::max(n1[i], n2[i]));
        tag2 = tag2 * 2 + (n2[i] != std::max(n1[i], n2[i]));
    }
    tiling.set_tag1(tag1);
    tiling.set_tag2(tag2);
    uint64_t ub_size;
    // ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    // printf("%d\n", ub_size);
    //ub:196352 Byte
    // if (sizeofdatatype == 2 || sizeofdatatype == 4)
    // {
    //     context->SetTilingKey(1);
    // }
    // else if (sizeofdatatype == 1)
    // {
    //     context->SetTilingKey(2);
    // }
    // else if (tag1 == 0 && tag2 == 2){
    //     context->SetTilingKey(3);
    // }
    // else context->SetTilingKey(4);

    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // size_t userWorkspaceSize = 102400; // queue depth * aivnum * message size
    // size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    // size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    // printf("%d %d %d\n",systemWorkspaceSize, currentWorkspace[0],currentWorkspace[1]);
    // currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;

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
class Lcm : public OpDef {
public:
    explicit Lcm(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Lcm);
}
