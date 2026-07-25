#include "hypot_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

static std::vector<int> get_canon_dims(const gert::Shape &shape)
{
    std::vector<int> dims{1, 1, 1, 1};
    for (auto i = 0; i < shape.GetDimNum(); i++)
        dims[i] = shape[shape.GetDimNum() - i - 1];
    while (!dims.empty() && dims.back() == 1)
        dims.pop_back();
    return dims;
}

template<typename T>
constexpr T ceil_div(T x, T y)
{
    return (x - 1) / y + 1;
}

template<typename T>
constexpr T ceil_round(T x, T y)
{
    return ceil_div(x, y) * y;
}

static int get_dim(const gert::Shape &shape, int dim)
{
    if (dim < shape.GetDimNum())
        return shape[shape.GetDimNum() - dim - 1];
    else
        return 1;
}

namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        auto &input = context->GetInputShape(0)->GetStorageShape();
        auto &other = context->GetInputShape(1)->GetStorageShape();
        auto input_dims = get_canon_dims(input);
        auto other_dims = get_canon_dims(other);
        if (input_dims.size() != other_dims.size())
        {
            auto min_size = std::min(input_dims.size(), other_dims.size());
            auto max_size = std::max(input_dims.size(), other_dims.size());
            auto &max_shape = input_dims.size() > other_dims.size() ? input_dims : other_dims;
            auto high_dim = 1;
            for (auto i = min_size; i < max_size; i++)
                high_dim *= max_shape[i];
            max_shape.resize(min_size + 1);
            max_shape[min_size] = high_dim;
        }
        input_dims.resize(4, 1);
        other_dims.resize(4, 1);
        std::vector<int> out_dims(4);
        auto size = 1;
        for (auto i = 0; i < 4; i++)
        {
            out_dims[i] = std::max(input_dims[i], other_dims[i]);
            size *= out_dims[i];
        }
        auto broadcast_count = 0;
        for (auto i = 0; i < 4; i++)
        {
            if (input_dims[i] != other_dims[i])
                broadcast_count++;
        }
        auto low_size = 1;
        auto high_size = 1;
        bool swap_input_other = false;
        auto dtype = context->GetInputTensor(0)->GetDataType();
        auto stype = 0;
        if (broadcast_count > 1)
            stype = 1;
        else if (broadcast_count == 1)
        {
            for (auto i = 0; i < 4; i++)
            {
                if (input_dims[i] != other_dims[i])
                {
                    swap_input_other = other_dims[i] == 1;
                    for (auto j = 0; j < i; j++)
                        low_size *= out_dims[j];
                    for (auto j = i + 1; j < 4; j++)
                        high_size *= out_dims[j];
                }
            }
            if (ge::GetSizeInBytes(low_size, dtype) % 32 != 0)
                stype = 1;
            else
                stype = 2;
        }
        if (stype > 1)
            size /= low_size * high_size;
        std::vector<long> broadcast_offset(2);
        auto single_size = ceil_round(ge::GetSizeInBytes(size, dtype), 512L);
        long next_offset = 0;
        if (input_dims == out_dims)
            broadcast_offset[0] = -1;
        else
        {
            broadcast_offset[0] = next_offset;
            next_offset += single_size;
        }
        if (other_dims == out_dims)
            broadcast_offset[1] = -1;
        else
        {
            broadcast_offset[1] = next_offset;
            next_offset += single_size;
        }

        HypotTilingData tiling;
        tiling.set_size(size);
        tiling.set_low_size(low_size);
        tiling.set_high_size(high_size);
        tiling.set_swap_input_other(swap_input_other);
        tiling.set_input_n(input_dims.data());
        tiling.set_other_n(other_dims.data());
        tiling.set_out_n(out_dims.data());
        tiling.set_broadcast_offset(broadcast_offset.data());
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        auto plaform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto block_dim = plaform.GetCoreNumAiv();
        context->SetBlockDim(block_dim);
        context->SetTilingKey(dtype + stype * 100);
        if (stype == 1)
        {
            auto workspace_sizes = context->GetWorkspaceSizes(1);
            workspace_sizes[0] = plaform.GetLibApiWorkSpaceSize() + next_offset;
        }

        return ge::GRAPH_SUCCESS;
    }
}


namespace ge
{
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *input_shape = context->GetInputShape(0);
        const gert::Shape *other_shape = context->GetInputShape(1);
        gert::Shape *out_shape = context->GetOutputShape(0);
        out_shape->SetDimNum(std::max(input_shape->GetDimNum(), other_shape->GetDimNum()));
        for (auto i = 0; i < out_shape->GetDimNum(); i++)
            (*out_shape)[out_shape->GetDimNum() - i - 1] = std::max(get_dim(*input_shape, i), get_dim(*other_shape, i));
        return GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return ge::GRAPH_SUCCESS;
    }
}


namespace ops
{
    class Hypot : public OpDef
    {
    public:
        explicit Hypot(const char *name) : OpDef(name)
        {
            this->Input("input")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("other")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("out")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

            this->AICore()
                    .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(Hypot);
}
