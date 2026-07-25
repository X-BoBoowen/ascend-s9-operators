
#include "select_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cstdio>

inline uint32_t align128U(uint32_t n,uint32_t DataType){
    n *= DataType;
    return ((n + 127) & ~127) / DataType;
}

inline uint32_t align128D(uint32_t n, uint32_t DataType) {
    n *= DataType;
    return (n & ~127) / DataType;
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  SelectV2TilingData tiling;
  uint64_t sizeofdatatype;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();

    uint64_t totalLength = context->GetInputShape(1)->GetStorageShape().GetShapeSize(); 
    auto dt = context->GetInputDesc(1)->GetDataType();
    if(dt == ge::DT_INT8){
        sizeofdatatype = 1;
    }else if(dt== ge::DT_FLOAT16 || dt == ge::DT_BF16){
        sizeofdatatype = 2;
    }else{
        sizeofdatatype = 4;
    }
    uint32_t cSize  = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t x1Size = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    uint32_t x2Size = context->GetInputShape(2)->GetStorageShape().GetShapeSize();
    if (x1Size != x2Size || cSize != x1Size || cSize != x2Size){  
        context->SetTilingKey(2);
        uint32_t y_ndarray[20], c_ndarray[20], x1_ndarray[20], x2_ndarray[20];
        uint32_t y_dimensional, c_dimensional, x1_dimensional, x2_dimensional;
        
        auto shape_y  = context->GetOutputShape(0)->GetOriginShape();
        auto shape_c = context->GetInputTensor(0)->GetOriginShape();
        auto shape_x1 = context->GetInputTensor(1)->GetOriginShape();
        auto shape_x2 = context->GetInputTensor(2)->GetOriginShape();
        
        y_dimensional = shape_y.GetDimNum();
        c_dimensional = shape_c.GetDimNum();
        x1_dimensional = shape_x1.GetDimNum();
        x2_dimensional = shape_x2.GetDimNum();

        uint32_t max_dimensional = y_dimensional;
        if (c_dimensional > max_dimensional)
            max_dimensional = c_dimensional;
        if (x1_dimensional > max_dimensional)
            max_dimensional = x1_dimensional;
        if (x2_dimensional > max_dimensional)
            max_dimensional = x2_dimensional;
     
        for (uint32_t i = 0; i < max_dimensional; i++) {
            if (i < y_dimensional) {
                y_ndarray[y_dimensional - i - 1] = shape_y.GetDim(i);
            } else {
                y_ndarray[i] = 1;
            }
            if (i <c_dimensional) {
                c_ndarray[c_dimensional - i - 1] = shape_c.GetDim(i);
            } else {
                c_ndarray[i] = 1;
            }
            if (i < x1_dimensional) {
                x1_ndarray[x1_dimensional - i - 1] = shape_x1.GetDim(i);
            } else {
                x1_ndarray[i] = 1;
            }
            if (i < x2_dimensional) {
                x2_ndarray[x2_dimensional - i - 1] = shape_x2.GetDim(i);
            } else {
                x2_ndarray[i] = 1;
            }
        }

        tiling.set_y_dimensional(max_dimensional);
        tiling.set_y_ndarray(y_ndarray);
        tiling.set_c_ndarray(c_ndarray);
        tiling.set_x1_ndarray(x1_ndarray);
        tiling.set_x2_ndarray(x2_ndarray);
        
        uint32_t y_sumndarray[20], c_sumndarray[20],x1_sumndarray[20], x2_sumndarray[20];
        y_sumndarray[0] = 1;
        c_sumndarray[0] = 1;
        x1_sumndarray[0] = 1;
        x2_sumndarray[0] = 1;
        for (uint32_t i = 1; i <= max_dimensional; i++){
            y_sumndarray[i] = y_sumndarray[i - 1] * y_ndarray[i - 1];
            c_sumndarray[i] = c_sumndarray[i - 1] * c_ndarray[i - 1];
            x1_sumndarray[i] = x1_sumndarray[i - 1] * x1_ndarray[i - 1];
            x2_sumndarray[i] = x2_sumndarray[i - 1] * x2_ndarray[i - 1];
        }
        tiling.set_y_sumndarray(y_sumndarray);
        tiling.set_c_sumndarray(c_sumndarray);
        tiling.set_x1_sumndarray(x1_sumndarray);
        tiling.set_x2_sumndarray(x2_sumndarray);
        totalLength = align128U(y_sumndarray[max_dimensional],sizeofdatatype); 
        
        tiling.set_cSize(align128U(cSize,sizeofdatatype));
        tiling.set_x1Size(align128U(x1Size,sizeofdatatype));
        tiling.set_x2Size(align128U(x2Size,sizeofdatatype));
        tiling.set_totalLength(totalLength);

        uint32_t tileLength;
        if(dt == ge::DT_FLOAT16){        // fp16 
            tileLength = align128D(ub_size / 14,sizeofdatatype);
        }else if(dt == ge::DT_FLOAT){    // fp32
            tileLength = align128D(ub_size / 24,sizeofdatatype);
        }else if(dt == ge::DT_INT32){     //  int32 填充成fp32
            tileLength = align128D(ub_size / 36,sizeofdatatype);
        }else if(dt == ge::DT_INT8){      // int8 填充成fp16
            tileLength = align128D(ub_size / 20,sizeofdatatype);
        }
        tileLength = std::min((int)tileLength,int(totalLength));
    
        uint32_t loopCount = totalLength / tileLength;
        uint32_t leftNum = align128U(totalLength % tileLength,sizeofdatatype);
        tiling.set_tileLength(tileLength);
        tiling.set_loopCount(loopCount);
        tiling.set_leftNum(leftNum);
        
        // printf("y_dimensional: %d\n", max_dimensional);
        // printf("cSize %d\n",cSize);
        // printf("x1Size: %d\n", x1Size);
        // printf("x2Size: %d\n", x2Size);
        // printf("totalLength: %d\n", totalLength);
        // printf("tileLength: %d\n", tileLength);
        // printf("loopCount: %d\n", loopCount);
        // printf("leftNum: %d\n", leftNum);
        // printf("ubsize: %d\n",ub_size);

    }else{
        context->SetTilingKey(1);
        totalLength = align128U(totalLength,sizeofdatatype);
        uint32_t tileLength;
        if(dt == ge::DT_FLOAT16){        // fp16 
            tileLength = align128D(ub_size / 18,sizeofdatatype);
        }else if(dt == ge::DT_FLOAT){    // fp32
            tileLength = align128D(ub_size / 30,sizeofdatatype);
        }else if(dt == ge::DT_INT32){     //  int32 填充成fp32
            tileLength = align128D(ub_size / 42,sizeofdatatype);
        }else if(dt == ge::DT_INT8){      // int8 填充成fp16
            tileLength = align128D(ub_size / 24,sizeofdatatype);
        }
        
        tileLength = std::min((int)tileLength,int(totalLength));
        
        uint32_t loopCount = totalLength / tileLength;
        uint32_t leftNum = align128U(totalLength % tileLength,sizeofdatatype);
        tiling.set_tileLength(tileLength);
        tiling.set_loopCount(loopCount);
        tiling.set_leftNum(leftNum);
        tiling.set_totalLength(totalLength); // Gm总地址32B对齐
        
        // printf("totalLength: %d\n", totalLength);
        // printf("tileLength: %d\n", tileLength);
        // printf("loopCount: %d\n", loopCount);
        // printf("leftNum: %d\n", leftNum);
        // printf("ubsize: %d\n",ub_size);
    }
    context->SetBlockDim(1);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class SelectV2 : public OpDef {
public:
    explicit SelectV2(const char* name) : OpDef(name)
    {
        this->Input("condition")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(SelectV2);
}
