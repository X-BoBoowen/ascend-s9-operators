#include "bitwise_left_shift_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

constexpr int64_t BLOCK_SIZE(256);
constexpr int64_t BUFFER_NUM(2);
constexpr uint64_t PRESERVE_UB(8 * 1024);
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {

    BitwiseLeftShiftTilingData tiling;

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNum();
    // if (coreNum < 3) coreNum = 3;

    const gert::Shape input_shape =
        context->GetInputShape(0)->GetStorageShape();
    const gert::Shape other_shape =
        context->GetInputShape(1)->GetStorageShape();
    const gert::Shape out_shape = context->GetOutputShape(0)->GetStorageShape();

    int64_t input_dim_num = input_shape.GetDimNum();
    int64_t other_dim_num = other_shape.GetDimNum();
    int64_t out_dim_num = out_shape.GetDimNum();

    uint8_t board_cast = 0;
    int64_t out_dims[3];

    for (int64_t i = 0; i < 3; i++) {

        int64_t input_dim = 1, other_dim = 1, out_dim = 1;
        if (i + input_dim_num - 3 >= 0)
            input_dim = input_shape.GetDim(i + input_dim_num - 3);
        if (i + other_dim_num - 3 >= 0)
            other_dim = other_shape.GetDim(i + other_dim_num - 3);
        if (i + out_dim_num - 3 >= 0)
            out_dim = out_shape.GetDim(i + out_dim_num - 3);

        if (input_dim < out_dim) board_cast |= (uint8_t(1) << i);
        if (other_dim < out_dim) board_cast |= (uint8_t(1) << (i + 3));
        out_dims[i] = out_dim;
    }

    tiling.set_board_cast(board_cast);

    int64_t total_length = 1;
    for (int64_t i = 0; i < out_dim_num; i++) {
        total_length *= out_shape.GetDim(i);
    }

    const ge::DataType data_type = context->GetInputTensor(0)->GetDataType();
    int64_t data_type_size = ge::GetSizeByDataType(data_type);

    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    ub_size -= PRESERVE_UB;

    int64_t ub_per_length[ge::DataType::DT_MAX];
    ub_per_length[ge::DataType::DT_INT8] = (1 * 3);
    ub_per_length[ge::DataType::DT_INT16] = (2 * 3 + 1);
    ub_per_length[ge::DataType::DT_INT32] = (4 * 3 + 1);
    ub_per_length[ge::DataType::DT_INT64] = (8 * 3);

    if (board_cast) {

        int64_t length_per_block = BLOCK_SIZE / data_type_size;
        int64_t total_block =
            (total_length + length_per_block - 1) / length_per_block;
        int64_t tail_block = total_block / coreNum;
        int64_t former_block = tail_block + 1;
        int64_t former_num = total_block % coreNum;

        int64_t ub_per_block = length_per_block * ub_per_length[data_type];
        int64_t tile_block = ub_size / ub_per_block / BUFFER_NUM;

        tiling.set_tile_length(tile_block * length_per_block);
        tiling.set_former_length(former_block * length_per_block);
        tiling.set_tail_length(tail_block * length_per_block);
        tiling.set_former_num(former_num);

        tiling.set_out_dim1(out_dims[0]);
        tiling.set_out_dim2(out_dims[1]);
        tiling.set_out_dim3(out_dims[2]);

    } else {

        int64_t length_per_block = BLOCK_SIZE / data_type_size;
        int64_t total_block =
            (total_length + length_per_block - 1) / length_per_block;
        int64_t tail_block = total_block / coreNum;
        int64_t former_block = tail_block + 1;
        int64_t former_num = total_block % coreNum;

        int64_t ub_per_block = length_per_block * ub_per_length[data_type];
        int64_t tile_block = ub_size / ub_per_block / BUFFER_NUM;

        tiling.set_tile_length(tile_block * length_per_block);
        tiling.set_former_length(former_block * length_per_block);
        tiling.set_tail_length(tail_block * length_per_block);
        tiling.set_former_num(former_num);
    }

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class BitwiseLeftShift : public OpDef {
  public:
    explicit BitwiseLeftShift(const char *name) : OpDef(name) {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("other")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(BitwiseLeftShift);
} // namespace ops

