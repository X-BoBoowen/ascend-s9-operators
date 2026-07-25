
#include "trunc_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TruncTilingData tiling;
 
    int32_t NUM = 12;
    int32_t aivNum = 1;
    uint32_t sizeofdatatype; 
    uint64_t ub_size; 
    uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto dt = context->GetInputDesc(0)->GetDataType();
    if(dt == ge::DT_INT8 || dt == ge::DT_UINT8){
        sizeofdatatype = 1;
        NUM = 4; 
    }else if(dt == ge::DT_FLOAT16 || dt == ge::DT_BF16){
        sizeofdatatype = 2;
        NUM = 5; 
    }else if(dt == ge::DT_FLOAT){
        NUM = 3;
        sizeofdatatype = 4; 
    } else {
        NUM = 3;
        sizeofdatatype = 4; 
    }
    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype; // 8
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;   
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8; 
    uint32_t block_size = tiling_size * ALIGN_NUM; 
    uint32_t core_size = totalLength + (totalLength % ALIGN_NUM ? ALIGN_NUM - totalLength % ALIGN_NUM : 0); // aligned core
    uint32_t tile_num = (core_size + block_size - 1) / block_size;
    uint32_t final_length = core_size - ((tile_num - 1) * block_size);

    tiling.set_block_size(block_size); 
    tiling.set_core_size(core_size); 
    tiling.set_tile_num(tile_num);
    tiling.set_final_length(final_length); 

    context->SetBlockDim(aivNum); 
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
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
class Trunc : public OpDef {
public:
    explicit Trunc(const char* name) : OpDef(name)
    {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output_y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Trunc);
}
