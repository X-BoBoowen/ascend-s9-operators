

#include "register/op_def_registry.h"  
#include "tiling/platform/platform_ascendc.h"  
#include "tiling/tiling_api.h"
#include "fresnel_cos_tiling.h"


namespace optiling {  
const uint32_t BLOCK_SIZE = 32;  
static const uint32_t DT_SIZE1 = 1;  
static const uint32_t DT_SIZE2 = 2;  
static const uint32_t DT_SIZE4 = 4;  


static const uint32_t UB_DIVISION_FACTOR = 2;  
static const uint32_t ALIGNMENT_THRESHOLD = 8;  
static const uint32_t CORE_ALIGNMENT_FACTOR = 8;  
static const uint32_t MIN_AIV_NUM = 1;  

static ge::graphStatus TilingFunc(gert::TilingContext* context) {  
    FresnelCosTilingData tiling;  
    int32_t NUM = 24;
    uint32_t sizeofdatatype;  
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());  
    auto socVersion = ascendcPlatform.GetSocVersion();  
    uint64_t ub_size;  
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);  
    auto aivNum = ascendcPlatform.GetCoreNum();  
    uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();  
    auto dt = context->GetInputDesc(0)->GetDataType();  

    // 确定数据类型大小/
    if(dt == ge::DT_INT8){  
        sizeofdatatype = DT_SIZE1;  
    } else if(dt == ge::DT_FLOAT16 || dt == ge::DT_BF16){  
        sizeofdatatype = DT_SIZE2;  
    } else { // DT_FLOAT, DT_INT32, DT_UINT32 等
        sizeofdatatype = DT_SIZE4;  
    }  
    // sizeofdatatype = DT_SIZE4;  
    if (sizeofdatatype == 0) {  
        return ge::GRAPH_FAILED;  
    }  

    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;  

    if (BLOCK_SIZE == 0 || NUM == 0) {  
        return ge::GRAPH_FAILED;  
    }  

    // 计算每个核心处理的数据量 (以 BLOCK_SIZE 为单位)
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / UB_DIVISION_FACTOR) / NUM;  
    // 对 tiling_size 进行对齐
    tiling_size = tiling_size <= ALIGNMENT_THRESHOLD ?   
                tiling_size :   
                (tiling_size / ALIGNMENT_THRESHOLD) * ALIGNMENT_THRESHOLD;  

    uint32_t block_size = tiling_size * ALIGN_NUM;  

    // 确定使用的核心数量 (aivNum)
    if (block_size == 0) {  
        aivNum = MIN_AIV_NUM;  
    } else {  
        aivNum = (aivNum < totalLength / block_size) ? aivNum : (totalLength / block_size);  
        aivNum = aivNum >= MIN_AIV_NUM ? aivNum : MIN_AIV_NUM;  
    }  

    uint32_t core_size = 0;  
    uint32_t core_remain = 0;  

    if (aivNum == 0 || ALIGN_NUM == 0) {  
        core_size = 0;  
        core_remain = totalLength;  
    } else {  
        uint32_t align_factor = ALIGN_NUM * CORE_ALIGNMENT_FACTOR;  
        core_size = (totalLength / aivNum) / align_factor * align_factor;  
        core_remain = totalLength - aivNum * core_size;  
    }  

    // 填充 tiling 数据结构
    tiling.set_totalLength(totalLength);  
    tiling.set_ALIGN_NUM(ALIGN_NUM);  
    tiling.set_tiling_size(tiling_size);  
    tiling.set_block_size(block_size);  
    tiling.set_aivNum(aivNum);  
    tiling.set_core_size(core_size);  
    tiling.set_core_remain(core_remain);  

    context->SetBlockDim(aivNum);  
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());  
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());  

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);  
    currentWorkspace[0] = 0; // 本算子不需要额外的 workspace  

    return ge::GRAPH_SUCCESS;  
}  
}  

namespace ge {  
static ge::graphStatus InferShape(gert::InferShapeContext* context) {  
    const gert::Shape* x_shape = context->GetInputShape(0);  
    gert::Shape* y_shape = context->GetOutputShape(0);  
    *y_shape = *x_shape; // 输出形状与输入相同
    return GRAPH_SUCCESS;  
}  
}  


namespace ops {
class FresnelCos : public OpDef {
public:
    explicit FresnelCos(const char* name) : OpDef(name)
    {
        this->Input("x")
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
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(FresnelCos);
}
