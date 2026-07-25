#include "lin_space_tiling.h"
#define DEBUG_OUTPUT 1
#include "host_inc.h"


namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        context->SetBlockDim(1);
        printDebugInfo<3, 1>(context);
        const auto [y_shape, yLength] = getOutputShape(context, 0);
        const auto dt = context->GetInputDesc(0)->GetDataType();
        auto ubSize = getUBSize(context);
        

        if (true) {
            LinSpaceTIBasic tiling;
            context->SetTilingKey(1);
            uint64_t buffSize = 0;
            if(dt == DT_INT16 
                || dt == DT_INT32 
                || dt == DT_INT8) {                                 // i8 i16 i32
                const auto SizePerIt = 2 * getDTSize(context, 0);
                buffSize = ubSize / SizePerIt;
            } else if (dt == DT_FLOAT) {                            // f32
                const auto SizePerIt = 4 * getDTSize(context, 0);
                buffSize = ubSize / SizePerIt;
            } else if (dt == DT_FLOAT16 || dt == DT_BF16) {         // f16 b16
                const auto SizePerIt = 2 * getDTSize(context, 0) + 4;
                buffSize = (ubSize - 512) / SizePerIt;
            } else if (dt == DT_UINT8) {                            // u8
                const auto SizePerIt = 8 * getDTSize(context, 0);
                buffSize = ubSize / SizePerIt;
            }
            
            
            const auto [tileLength, tileNumber, reminder] = getTilingInfo(buffSize, yLength, getDTSize(context, 0), 512, 32);
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
class LinSpace : public OpDef {
public:
    explicit LinSpace(const char* name) : OpDef(name)
    {
        this->Input("start")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_INT32, ge::DT_INT8, ge::DT_INT16, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("stop")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_INT32, ge::DT_INT8, ge::DT_INT16, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("num_axes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_INT32, ge::DT_INT8, ge::DT_INT16, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(LinSpace);
}
