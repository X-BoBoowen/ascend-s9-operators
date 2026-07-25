 #include "hypot_tiling.h"
 #include "register/op_def_registry.h"
 #include "tiling/platform/platform_ascendc.h"

 namespace optiling {
 
 
 static void EncodeShape(const gert::StorageShape *shape, uint32_t arr[8]) {
    
    const auto &origin_shape = shape->GetOriginShape();
    
    size_t nd = origin_shape.GetDimNum();
    arr[0] = nd > 7 ? 7 : nd;
    
    for (size_t i = 0; i < arr[0]; ++i) {
        arr[i+1] = static_cast<uint32_t>(origin_shape.GetDim(i));
    }
    
    for (size_t i = arr[0] + 1; i < 8; ++i) arr[i] = 1;
}

static uint32_t GetEncodedShapeSize(const uint32_t* arr)
{
    if (arr == nullptr) {
        return 0;
    }
    uint32_t ndim = arr[0];
    if (ndim == 0) {
         return 1; 
    }

    uint64_t size = 1; 
    for (uint32_t i = 1; i <= ndim; ++i) {
        uint32_t dim = arr[i];
         if (dim == 0) { 
             return 0;
         }
         if (size > UINT64_MAX / static_cast<uint64_t>(dim) && dim > 1) {
              if (size > UINT32_MAX / static_cast<uint64_t>(dim) && dim > 1) {
                   return 0; 
              }
         }
        size *= static_cast<uint64_t>(dim);
    }

    if (size > UINT32_MAX) {
         return 0; 
    }

    return static_cast<uint32_t>(size);
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    TilingData tiling;
    uint32_t dt ;
    auto dtype=context->GetInputDesc(0)->GetDataType();
    if(dtype==ge::DT_FLOAT16)dt= 1;
    else if(dtype==ge::DT_FLOAT)dt=0;
    else dt=2;
    uint32_t arrInput[8], arrValues[8], arrOut[8];

    EncodeShape(context->GetInputShape(0), arrInput);
    EncodeShape(context->GetInputShape(1), arrValues);
    EncodeShape(context->GetOutputShape(0), arrOut);
    tiling.set_dt(dt);
    uint32_t x_length = GetEncodedShapeSize(arrInput);
    uint32_t value_length = GetEncodedShapeSize(arrValues);
    tiling.set_input_shape_info(arrInput);
    tiling.set_values_shape_info(arrValues);
    tiling.set_output_shape_info(arrOut);
    tiling.set_xlength(x_length);
    tiling.set_vlength(value_length);

    context->SetBlockDim(aivNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

 } 
 
 namespace ge {
 static graphStatus InferShape(gert::InferShapeContext *context)
 {
     
     const gert::Shape *x1_shape = context->GetInputShape(0);
     const gert::Shape *x2_shape = context->GetInputShape(1);
     gert::Shape *y_shape = context->GetOutputShape(0);
 
     
     size_t ndim1 = x1_shape->GetDimNum();
     size_t ndim2 = x2_shape->GetDimNum();
     size_t ndim_out = std::max(ndim1, ndim2);
     std::vector<int64_t> out_shape;
     for (size_t i=0; i<ndim_out; ++i) {
         int64_t d1 = (i < ndim_out - ndim1) ? 1 : x1_shape->GetDim(i - (ndim_out - ndim1));
         int64_t d2 = (i < ndim_out - ndim2) ? 1 : x2_shape->GetDim(i - (ndim_out - ndim2));
         if (d1 != d2 && d1 != 1 && d2 != 1) return GRAPH_FAILED; 
         out_shape.push_back(std::max(d1, d2));
     }
    
     for (auto d : out_shape) y_shape->AppendDim(d);
     return GRAPH_SUCCESS;
 }
 
 static graphStatus InferDataType(gert::InferDataTypeContext *context)
 {
     const auto inputDataType = context->GetInputDataType(0);
     context->SetOutputDataType(0, inputDataType);
     return ge::GRAPH_SUCCESS;
 }
 } 
 
 namespace ops {
 class Hypot : public OpDef {
 public:
     explicit Hypot(const char *name) : OpDef(name)
     {
         this->Input("input")
             .ParamType(REQUIRED)
             .DataType({ge::DT_FLOAT16, ge::DT_FLOAT,ge::DT_BF16})
             .Format({ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND});
         this->Input("values")
             .ParamType(REQUIRED)
             .DataType({ge::DT_FLOAT16, ge::DT_FLOAT,ge::DT_BF16})
             .Format({ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND});
         this->Output("out")
             .ParamType(REQUIRED)
             .DataType({ge::DT_FLOAT16, ge::DT_FLOAT,ge::DT_BF16})
             .Format({ge::FORMAT_ND, ge::FORMAT_ND,ge::FORMAT_ND});
 
         this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
         this->AICore()
             .SetTiling(optiling::TilingFunc)
             .AddConfig("ascend910")
             .AddConfig("ascend310p")
             .AddConfig("ascend310b")
             .AddConfig("ascend910b");
     }
 };
 OP_ADD(Hypot);
 } 
 