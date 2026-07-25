#include "eye_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        auto &y = context->GetInputTensor(0)->GetShape().GetStorageShape();
        auto dim_num = y.GetDimNum();
        auto batch_size = 1;
        for (auto i = 0; i < dim_num - 2; i++)
            batch_size *= y[i];
        auto num_rows = y[dim_num - 2];
        auto num_columns = y[dim_num - 1];

        EyeTilingData tiling;
        tiling.set_num_rows(num_rows);
        tiling.set_num_columns(num_columns);
        tiling.set_batch_size(batch_size);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        auto plaform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto block_dim = plaform.GetCoreNumAiv();
        context->SetBlockDim(block_dim);
        auto dtype = *context->GetAttrs()->GetInt(3);
        context->SetTilingKey(dtype);

        return ge::GRAPH_SUCCESS;
    }
}


namespace ge
{
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *y1_shape = context->GetInputShape(0);
        gert::Shape *y2_shape = context->GetOutputShape(0);
        *y2_shape = *y1_shape;
        return GRAPH_SUCCESS;
    }
}


namespace ops
{
    class Eye : public OpDef
    {
    public:
        explicit Eye(const char *name) : OpDef(name)
        {
            this->Input("y")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_DOUBLE})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_DOUBLE})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("num_rows").Int();
            this->Attr("num_columns").AttrType(OPTIONAL).Int(0);
            this->Attr("batch_shape").ListInt();
            this->Attr("dtype").AttrType(OPTIONAL).Int(0);

            this->SetInferShape(ge::InferShape);

            this->AICore()
                    .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(Eye);
}
