
#include "lcm_tiling.h"
#include "register/op_def_registry.h"

#define CORENUM 40
#define BLOCK_BYTES_SIZE 32
#define MAXDIM 8

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  // printf("into tilingfunc\n");

  const gert::StorageShape *x1_shape = context->GetInputShape(0);
  const gert::StorageShape *x2_shape = context->GetInputShape(1);

  uint32_t x1dimNum, x2dimNum;
  x1dimNum = x1_shape->GetStorageShape().GetDimNum();
  x2dimNum = x2_shape->GetStorageShape().GetDimNum();
  int maxDim = x1dimNum > x2dimNum ? x1dimNum : x2dimNum;

  uint32_t shapeInf1[MAXDIM] = {};
  uint32_t shapeInf2[MAXDIM] = {};

  uint32_t zipInf1[MAXDIM] = {};
  uint32_t zipInf2[MAXDIM] = {};

  for (int i = 0; i < maxDim; i++) {
    shapeInf1[i] = 1;
    shapeInf2[i] = 1;
    zipInf1[i] = 1;
    zipInf2[i] = 1;
  }
  int dimOffset = maxDim - x1dimNum;
  for (int i = 0; i < x1dimNum; i++) {
    uint64_t tmp = x1_shape->GetStorageShape().GetDim(i);
    shapeInf1[i + dimOffset] = tmp;
  }
  dimOffset = maxDim - x2dimNum;
  for (int i = 0; i < x2dimNum; i++) {
    uint64_t tmp = x2_shape->GetStorageShape().GetDim(i);
    shapeInf2[i + dimOffset] = tmp;
  }

  int tag = 0;
  if (shapeInf1[0] > shapeInf2[0]) {
    tag = 1;
  } else if (shapeInf1[0] < shapeInf2[0]) {
    tag = -1;
  }
  zipInf1[0] = shapeInf1[0];
  zipInf2[0] = shapeInf2[0];
  int p = 0;
  for (int i = 1; i < maxDim; i++) {
    int tmpTag = 0;
    if (shapeInf1[i] > shapeInf2[i]) {
      tmpTag = 1;
    } else if (shapeInf1[i] < shapeInf2[i]) {
      tmpTag = -1;
    }
    if (tag == tmpTag) {
      zipInf1[p] *= shapeInf1[i];
      zipInf2[p] *= shapeInf2[i];
    } else {
      p++;
      zipInf1[p] = shapeInf1[i];
      zipInf2[p] = shapeInf2[i];
      tag = tmpTag;
    }
  }
  int zipDimNum = p + 1;

  uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(0)->GetDataType());
  int blockSize = BLOCK_BYTES_SIZE / inputBytes;


  bool allEqual = (zipDimNum == 1 && zipInf1[0] == zipInf2[0]) ? 1 : 0;

  if (allEqual) {
    if (inputBytes == 4) {
      context->SetTilingKey(0);

      LcmTilingData_0 tiling;
      int totalNum = zipInf1[0];
      tiling.set_totalNum(totalNum);
      int totalBlockNum = (totalNum + blockSize - 1) / blockSize;
      int coreNum = totalBlockNum < CORENUM ? totalBlockNum : CORENUM;

      context->SetBlockDim(coreNum);

      tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                          context->GetRawTilingData()->GetCapacity());
      context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    } else {
      context->SetTilingKey(1);

      LcmTilingData_1 tiling;
      int totalNum = zipInf1[0];
      tiling.set_totalNum(totalNum);
      int totalBlockNum = (totalNum + blockSize - 1) / blockSize;
      int coreNum = totalBlockNum < CORENUM ? totalBlockNum : CORENUM;

      context->SetBlockDim(coreNum);

      tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                          context->GetRawTilingData()->GetCapacity());
      context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    }
  } else if (zipDimNum == 3 && zipInf1[1] != zipInf2[1] && zipInf1[0] < 40) {
    context->SetTilingKey(3);

    LcmTilingData_3 tiling;
    uint16_t shapeInf[3] = {zipInf1[0], zipInf1[1], zipInf1[2]};
    tiling.set_shapeInf(shapeInf);

    int coreNum = zipInf1[1] < CORENUM ? zipInf1[1] : CORENUM;

    context->SetBlockDim(coreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  } else if (zipDimNum == 3 && zipInf1[1] != zipInf2[1]) {
    // context->SetTilingKey(2);

    // LcmTilingData_2 tiling;
    // uint16_t shapeInf[3] = {zipInf1[0],zipInf1[1],zipInf1[2]};
    // tiling.set_shapeInf(shapeInf);

    // int coreNum = zipInf1[0] < CORENUM ? zipInf1[0] : CORENUM;

    // context->SetBlockDim(coreNum);

    // tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
    // context->GetRawTilingData()->GetCapacity());
    // context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  } else {
    context->SetTilingKey(4);

    LcmTilingData tiling;
    tiling.set_maxDim(maxDim);
    tiling.set_shapeInf1(shapeInf1);
    tiling.set_shapeInf2(shapeInf2);
    int maxCol = shapeInf1[0] > shapeInf2[0] ? shapeInf1[0] : shapeInf2[0];
    maxCol = (maxCol + blockSize - 1) / blockSize;
    maxCol = maxCol < CORENUM ? maxCol : CORENUM;

    context->SetBlockDim(maxCol);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  }

  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

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
}  // namespace ge

namespace ops {
class Lcm : public OpDef {
 public:
  explicit Lcm(const char *name) : OpDef(name) {
    this->Input("x1")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("x2")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(Lcm);
}  // namespace ops
