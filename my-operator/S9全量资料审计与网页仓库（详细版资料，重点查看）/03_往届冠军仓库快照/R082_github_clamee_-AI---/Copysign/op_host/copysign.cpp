
#include "copysign_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <iostream>
#include <algorithm>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    CopysignTilingData tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto num_cores = ascendcPlatform.GetCoreNum();
    uint32_t sizeofdatatype = 1;
    auto dt = context->GetInputTensor(0)->GetDataType();
    int status = 2;
    if (dt == ge::DT_FLOAT) {
    	sizeofdatatype = 4;
    }
    else {
        sizeofdatatype = 2;
    }
    const uint32_t alignment = 64 / sizeofdatatype;

    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    auto dim1 = x1_shape->GetStorageShape().GetDimNum();
    uint32_t n1[4] = {1, 1, 1, 1};
    for (int i = 0; i < dim1; ++i) {
        n1[i] = x1_shape->GetStorageShape().GetDim(i);
    }
    const gert::StorageShape* x2_shape = context->GetInputShape(1);
    auto dim2 = x2_shape->GetStorageShape().GetDimNum();
    uint32_t n2[4] = {1, 1, 1, 1};
    for (int i = 0; i < dim2; ++i) {
        n2[i + (dim1 - dim2)] = x2_shape->GetStorageShape().GetDim(i);
    }
    int dim = std::max(dim1, dim2);
    //uint32_t ny[3] = {1, 1, 1};
    uint64_t size = 1;
    for (int i = 0; i < dim; ++i) {
    //    ny[i] = std::max(n1[i], n2[i]);
        size *= n1[i];
    }
    tiling.set_n1(n1);
    tiling.set_n2(n2);
    //std::cout << "ny: " << ny[0] << " " << ny[1] << " " << ny[2] << " " << std::endl;
    for (int i = 0; i <4; ++i) {
        if (n1[i] != n2[i]) {
            status = 1;
        }
    }
    if(size <= num_cores * alignment) status = 0;
    //std::cout << status << std::endl;
    tiling.set_status(status);
    tiling.set_size(size);
    uint64_t length = (size - 1) / num_cores + 1;
    //while (length % alignment != 0) length += 1;
    length+=((64-alignment)&(alignment-1));
    tiling.set_length(length);
    if (status == 0) {
        context->SetBlockDim(1);
    }
    else {
    //    std::cout << "Multicore" << std::endl;
        context->SetBlockDim(num_cores);
    }
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

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Copysign);
}
