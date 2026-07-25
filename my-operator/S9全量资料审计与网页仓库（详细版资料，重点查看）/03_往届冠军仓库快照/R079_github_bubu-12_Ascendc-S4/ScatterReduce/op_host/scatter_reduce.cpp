
#include "scatter_reduce_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/attr_value.h" 


namespace optiling {

static void EncodeShape(const gert::StorageShape *shape, uint32_t arr[SHAPE_INFO_DIM]) {
    if (shape == nullptr || arr == nullptr) return;
    const auto &storage_shape = shape->GetStorageShape();
    size_t nd = storage_shape.GetDimNum();
    arr[0] = nd > (SHAPE_INFO_DIM - 1) ? (SHAPE_INFO_DIM - 1) : static_cast<uint32_t>(nd); // Store actual ndim, capped at 7
    for (size_t i = 0; i < arr[0]; ++i) {
        arr[i + 1] = static_cast<uint32_t>(storage_shape.GetDim(i));
    }
    for (size_t i = arr[0] + 1; i < SHAPE_INFO_DIM; ++i) {
        arr[i] = 1;
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // --- 获取核数 (保持不变) ---
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    // 移除空指针检查
    // if (aivNum == 0) { aivNum = 1; }
    // context->SetBlockDim(1);
    context->SetBlockDim(aivNum);

    ScatterReduceTilingData tiling;

    // --- 获取 RuntimeAttrs 对象 ---
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    // 移除空指针检查
    // if (attrs == nullptr) { return ge::GRAPH_FAILED; } // 移除检查

    // --- 变量定义 ---
    int64_t dim_attr = 0;
    std::string reduce_attr_str;
    bool include_self_attr = true; // 默认值先设置好

    // --- 修改点 1: 使用索引从 attrs 获取属性指针 ---
    // 假设属性索引基于 OpDef 中的定义顺序: dim=0, reduce=1, include_self=2

    // 获取 'dim' (Int) - index 0
    const int64_t *dim_ptr = attrs->GetInt(0);
    // 移除空指针检查和错误处理
    // if (dim_ptr == nullptr) { return ge::GRAPH_FAILED; } // 移除检查
    dim_attr = *dim_ptr; // 直接解引用获取值

    // 获取 'reduce' (String) - index 1
    const char *reduce_ptr = attrs->GetStr(1); // GetStr 通常返回 const char*
    // 移除空指针检查和错误处理
    // if (reduce_ptr == nullptr) { return ge::GRAPH_FAILED; } // 移除检查
    reduce_attr_str = reduce_ptr; // 从 const char* 构造 std::string

    // 获取 'include_self' (Bool, Optional) - index 2
    const bool *include_self_ptr = attrs->GetBool(2);
    // 对于可选属性，如果指针非空，则使用其值，否则保持默认值 true
    if (include_self_ptr != nullptr) {
        include_self_attr = *include_self_ptr;
    }
    // else { // 属性不存在或获取失败，保持 include_self_attr = true; }

    // --- 移除点 2: 移除所有 ge::AttrValue 相关代码 ---
    // ge::AttrValue dim_value;
    // ge::AttrValue reduce_value;
    // ge::AttrValue include_self_value;
    // 以及所有相关的 GetAttr 和 GetValue 调用和检查

    // --- ReduceType 处理 (保持不变) ---
    ReduceType reduce_code = REDUCE_UNKNOWN;
    if (reduce_attr_str == "sum") { reduce_code = REDUCE_SUM; }
    else if (reduce_attr_str == "prod") { reduce_code = REDUCE_PROD; }
    else if (reduce_attr_str == "mean") { reduce_code = REDUCE_MEAN; }
    else if (reduce_attr_str == "amax") { reduce_code = REDUCE_AMAX; }
    else if (reduce_attr_str == "amin") { reduce_code = REDUCE_AMIN; }
    // 移除错误处理

    // --- 设置 TilingData (保持不变) ---
    tiling.set_dim(static_cast<uint32_t>(dim_attr));
    tiling.set_reduce_code(static_cast<uint32_t>(reduce_code));
    tiling.set_include_self_flag(include_self_attr ? 1 : 0);

    ge::DataType tensor_dtype = context->GetInputDesc(0)->GetDataType();
    uint32_t dtype_flag = (tensor_dtype == ge::DT_FLOAT16) ? 1 : 0;
    tiling.set_dtype_flag(dtype_flag);

    uint32_t self_shape_arr[SHAPE_INFO_DIM];
    uint32_t index_shape_arr[SHAPE_INFO_DIM];
    uint32_t src_shape_arr[SHAPE_INFO_DIM];
    uint32_t y_shape_arr[SHAPE_INFO_DIM];

    EncodeShape(context->GetInputShape(0), self_shape_arr); // self
    EncodeShape(context->GetInputShape(1), index_shape_arr); // index
    EncodeShape(context->GetInputShape(2), src_shape_arr); // src
    EncodeShape(context->GetOutputShape(0), y_shape_arr); // y

    tiling.set_self_shape_info(self_shape_arr);
    tiling.set_index_shape_info(index_shape_arr);
    tiling.set_src_shape_info(src_shape_arr);
    tiling.set_y_shape_info(y_shape_arr);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    //std::cout << aivNum << std::endl;
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
class ScatterReduce : public OpDef {
public:
    explicit ScatterReduce(const char* name) : OpDef(name)
    {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("src")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").Int();
        this->Attr("reduce").String();
        this->Attr("include_self").AttrType(OPTIONAL).Bool(true);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(ScatterReduce);
}

