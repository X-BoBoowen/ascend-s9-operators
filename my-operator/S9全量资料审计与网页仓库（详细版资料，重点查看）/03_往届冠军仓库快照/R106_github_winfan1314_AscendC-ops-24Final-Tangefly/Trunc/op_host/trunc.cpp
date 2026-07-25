
#include "trunc_tiling.h"
#define DEBUG_OUTPUT 1
#include "host_inc.h" 
 

namespace optiling {
    constexpr auto BUFFER_NUM = 2;
    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        context->SetBlockDim(1);
        printDebugInfo<1, 1>(context);
        const auto dt = context->GetInputDesc(0)->GetDataType();
        const auto ubSize = getUBSize(context);
        const auto [y_shape, yLength] = getOutputShape(context, 0);
        const auto type_sz = getDTSize(context, 0);
        if(dt == DT_FLOAT || dt == DT_BF16 || dt == DT_FLOAT16) { // cast
            context->SetTilingKey(1);
            TruncTIBasic tiling;
            auto SizePerIt = BUFFER_NUM * 2 * type_sz;
            if(dt != DT_FLOAT) SizePerIt += 4;
            const auto [tileLength, tileNumber, reminder] = getTilingInfo(ubSize / SizePerIt, yLength, type_sz, 512, 32);
            SET(tileLength);
            SET(tileNumber);
            SET(reminder);
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        } else { // copy
            context->SetTilingKey(2);
            TruncTIBasic2 tiling;
            const auto SizePerIt = BUFFER_NUM * type_sz;
            const auto [tileLength, tileNumber, reminder] = getTilingInfo(ubSize / SizePerIt, yLength, type_sz, 512, 32);
            SET(tileLength);
            SET(tileNumber);
            SET(reminder);
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        }
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
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output_y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8, ge::DT_UINT8})
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
