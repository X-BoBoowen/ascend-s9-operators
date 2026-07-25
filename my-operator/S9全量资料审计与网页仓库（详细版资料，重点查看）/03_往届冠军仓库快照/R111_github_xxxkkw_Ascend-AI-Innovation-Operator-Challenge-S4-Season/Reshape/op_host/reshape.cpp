
#include "reshape_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdio>

inline uint32_t align32U(uint32_t n,uint32_t DataType){
    n *= DataType;
    return ((n + 31) & ~31) / DataType;
}

inline uint32_t align32D(uint32_t n, uint32_t DataType) {
    n *= DataType;
    return (n & ~31) / DataType;
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  ReshapeTilingData tiling;
  uint64_t sizeofdatatype;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto socVersion = ascendcPlatform.GetSocVersion();
  uint64_t ub_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  auto aivNum = ascendcPlatform.GetCoreNum();
  
  uint64_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
  uint64_t shapeLength = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
  auto dt = context->GetInputDesc(0)->GetDataType();
  if(dt == ge::DT_INT8 || dt == ge::DT_UINT8){
      sizeofdatatype = 1;
  }else if(dt== ge::DT_FLOAT16 || dt == ge::DT_INT16 || dt == ge::DT_UINT16){
      sizeofdatatype = 2;
  }else if(dt == ge::DT_FLOAT || dt == ge::DT_INT32 || dt== ge::DT_UINT32){
      sizeofdatatype = 4;
  }else if(dt == ge::DT_INT64 || dt == ge::DT_UINT64){
      sizeofdatatype = 8;
  }
  
  totalLength = align32U(totalLength,sizeofdatatype);
  shapeLength = align32U(shapeLength,sizeofdatatype);
  uint32_t tileLength;
  tileLength = align32D(ub_size / (sizeofdatatype * 4) ,sizeofdatatype);
  tileLength = std::min((int)tileLength,int(totalLength));
      
  const int* axis = context->GetAttrs()->GetAttrPointer<int>(0);
  const int* num_axes = context->GetAttrs()->GetAttrPointer<int>(1);
  
  uint32_t loopCount = totalLength / tileLength;
  uint32_t leftNum = align32U(totalLength % tileLength,sizeofdatatype);
  tiling.set_tileLength(tileLength);
  tiling.set_loopCount(loopCount);
  tiling.set_leftNum(leftNum);
  tiling.set_totalLength(totalLength); // Gm总地址32B对齐
  tiling.set_shapeLength(shapeLength); // shape地址32B对齐
  tiling.set_axis(*axis);
  tiling.set_num_axes(*num_axes);
  
  printf("shapeLength: %d\n", shapeLength);
  printf("totalLength: %d\n", totalLength);
  printf("tileLength: %d\n", tileLength);
  printf("loopCount: %d\n", loopCount);
  printf("leftNum: %d\n", leftNum);
  printf("ubsize: %d\n",ub_size);
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
    const gert::Shape* x_shape = context->GetInputShape(0);
    int64_t x_rank = x_shape->GetDimNum();

    // 2. 获取 shape tensor，需在 OpDef 中对该输入做 ValueDepend
    const gert::Tensor* shape_tensor = context->GetInputTensor(1);
    if (shape_tensor == nullptr) {
        // 未标记为数据依赖或非 Const 输入
        return ge::GRAPH_PARAM_INVALID;
    }
    int64_t shape_len = shape_tensor->GetShapeSize();
    ge::DataType dt = shape_tensor->GetDataType();

    // 3. 将 shape values 读到 vector
    std::vector<int64_t> shape_vals;
    shape_vals.reserve(shape_len);
    if (dt == ge::DT_INT32) {
        const int32_t* data = shape_tensor->GetData<int32_t>();
        for (int64_t i = 0; i < shape_len; ++i) {
            shape_vals.push_back(static_cast<int64_t>(data[i]));
        }
    } else if (dt == ge::DT_INT64) {
        const int64_t* data = shape_tensor->GetData<int64_t>();
        for (int64_t i = 0; i < shape_len; ++i) {
            shape_vals.push_back(data[i]);
        }
    } else {
        // Unsupported dtype
        return ge::GRAPH_PARAM_INVALID;
    }

    // 4. 读取 axis 和 num_axes 属性
    const int* axis_ptr = context->GetAttrs()->GetAttrPointer<int>(0);
    const int* num_axes_ptr = context->GetAttrs()->GetAttrPointer<int>(1);
    int axis = *axis_ptr;
    int num_axes = *num_axes_ptr;
    if (num_axes == -1) {
        num_axes = static_cast<int>(x_rank) - axis;
    }

    std::vector<int64_t> y_dims;
    // dims before axis
    for (int i = 0; i < axis; ++i) {
        y_dims.push_back(x_shape->GetDim(i));
    }
    // dims from shape_vals, 处理 -1 推断
    int64_t infer_idx = -1;
    int64_t known_prod = 1;
    for (int64_t i = 0; i < shape_len; ++i) {
        int64_t v = shape_vals[i];
        if (v == -1) {
            if (infer_idx != -1) {
                return ge::GRAPH_PARAM_INVALID; // 只允许一个 -1
            }
            infer_idx = static_cast<int64_t>(y_dims.size());
            y_dims.push_back(1);
        } else {
            y_dims.push_back(v);
            known_prod *= v;
        }
    }
    // dims after replaced axes
    for (int i = axis + num_axes; i < x_rank; ++i) {
        y_dims.push_back(x_shape->GetDim(i));
    }

    // 6. 处理 -1 推断
    if (infer_idx >= 0) {
        int64_t total_in = 1;
        for (int i = 0; i < x_rank; ++i) total_in *= x_shape->GetDim(i);
        int64_t inferred = total_in / known_prod;
        y_dims[infer_idx] = inferred;
    }

    // 7. 最终校验元素总数一致
    int64_t prod_in = 1;
    for (int i = 0; i < x_rank; ++i) prod_in *= x_shape->GetDim(i);
    int64_t prod_out = 1;
    for (auto d : y_dims) prod_out *= d;
    if (prod_in != prod_out) {
        return ge::GRAPH_PARAM_INVALID;
    }

    // 8. 写回输出 shape
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->SetDimNum(static_cast<int>(y_dims.size()));
    for (size_t i = 0; i < y_dims.size(); ++i) {
        y_shape->SetDim(i, y_dims[i]);
    }

    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class Reshape : public OpDef {
public:
    explicit Reshape(const char* name) : OpDef(name)
    {
        //   this->Input("x")
        //     .ParamType(REQUIRED)
        //     .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64})
        //     .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        //     .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        // this->Input("shape")
        //     .ParamType(REQUIRED)
        //     .DataType({ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64})
        //     .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        //     .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        //     .ValueDepend(OpParamDef::REQUIRED, DependScope::ALL);
        // this->Output("y")
        //     .ParamType(REQUIRED)
        //     .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64})
        //     .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        //     .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("shape")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
            //.ValueDepend(REQUIRED);
        this->Output("y")   
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("axis").AttrType(OPTIONAL).Int(0);
        this->Attr("num_axes").AttrType(OPTIONAL).Int(-1);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Reshape);
}
