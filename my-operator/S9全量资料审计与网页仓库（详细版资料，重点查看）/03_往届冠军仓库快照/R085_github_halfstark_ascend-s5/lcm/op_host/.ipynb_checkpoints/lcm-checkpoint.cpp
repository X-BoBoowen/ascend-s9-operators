
#include "lcm_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define BLOCK_SIZE 256


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  LcmTilingData tiling;
  uint64_t ubSize;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); //获取硬件平台存储空间 UB 的内存大小
  const gert::StorageShape* input = context->GetInputShape(0);
  const gert::StorageShape* other = context->GetInputShape(1);
  // ubSize = 32;
  uint32_t dims = context->GetInputShape(0)->GetStorageShape().GetDimNum();
  uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
  uint32_t typeSize = GetSizeByDataType(context->GetInputDesc(0)->GetDataType()); //输入类型
  uint32_t totalBytes = context->GetInputTensor(0)->GetShapeSize();
  uint64_t inputNum = input->GetStorageShape().GetShapeSize(); //输入数量
  uint64_t otherNum = other->GetStorageShape().GetShapeSize(); //输入数量
  uint64_t inputLength = typeSize * inputNum; //输入长度
  uint64_t otherLength = typeSize * otherNum; //输入长度
  uint32_t ubDataNumber = 6;
    if (inputLength == otherLength) {
        ubDataNumber = 10;
    }
    auto dt = context->GetInputTensor(2)->GetDataType();
  uint32_t tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber; //每个ub段可用的空间块数
    // tileBlockNum = 2;
  uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeSize; //每次处理的数据量

  int coreNum = 40;
  
  if (inputLength == otherLength) {
      // return -1;
    // printf("inputLength == otherLength\n");
    context->SetTilingKey(0);
    if (typeSize != 4) {
      context->SetTilingKey(1);
    }
    uint32_t alignLen = (inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    uint32_t totalBlock = alignLen / BLOCK_SIZE;// 输入数据需要多少空间块    
    if (coreNum > totalBlock) {
        coreNum = totalBlock;
    }
    uint16_t nAcores = totalBlock % coreNum;
    uint16_t nBcores = coreNum - nAcores;
    // uint32_t CoreDataNum = totalBlock * BLOCK_SIZE / typeSize; //对齐空间后的输入数量
    // uint32_t TileNum = totalBlock / tileBlockNum;
    // uint32_t finalTileNum = (totalBlock % tileBlockNum) == 0 ? TileNum : TileNum + 1; //需要循环处理几次
    // uint32_t tailDataNum = inputLength - (tileDataNum * TileNum);
    // tailDataNum = tailDataNum == 0 ? tileDataNum : tailDataNum; // 最后一次处理的数据量
    tiling.set_blockPerCore(totalBlock / coreNum);
    tiling.set_maxBlockPerIter(tileBlockNum);
    tiling.set_nAcores(nAcores);
    tiling.set_nBcores(nBcores);
  } else {  
    // for (int i = 0; i < )
    context->SetTilingKey(2);
    auto other_shape = context->GetInputShape(1)->GetStorageShape();
    auto input_shape = context->GetInputShape(0)->GetStorageShape();
    // return -1;
    int cnt = 0;
    int pre = 1;
    int mid = 1, last=1;
    int batch = 1;
    // if (inputNum < otherNum) return -1;
    int lenin = input_shape.GetDimNum(),j=0;
    int lenother = other_shape.GetDimNum();
    bool found = false;
    for (int i = 0; i < other_shape.GetDimNum(); i++) {
        if (input_shape.GetDim(lenin - other_shape.GetDimNum() + i) != other_shape.GetDim(i)) {
            j = lenin - other_shape.GetDimNum() + i;
            mid = input_shape.GetDim(lenin - other_shape.GetDimNum() + i);
            found = true;
            break;
        }
    }
      // printf("j :%d\n", j);
        for (int i = 0; i < j; i++) {
          pre *= other_shape.GetDim(i);
        }
      // if (pre < 10) return -1;
      if (found) j++;
      for (int i = j; i < other_shape.GetDimNum(); i++) {
          last *= other_shape.GetDim(i);
      }            // if (cnt > 1) return -1;
      // periter = last;
      batch = 1;
      for (int i = 0; i < input_shape.GetDimNum() -  other_shape.GetDimNum(); i++) {
          batch *= input_shape.GetDim(i);
      }
      if (lenin == lenother) {
          batch = mid;
      }
      tiling.set_mid(mid);
      tiling.set_pre(pre);
      tiling.set_last(last);
        // coreNum = 2;
      if (coreNum > batch) coreNum = batch;

      // smallBatch = batch/coreNum;
      tiling.set_nAcores(batch%coreNum);
      tiling.set_smallBatch(batch/coreNum);
  }
  // context->SetTilingKey(0);

  // const gert::StorageShape* x1_shape = context->GetInputShape(0);
//   int32_t data_sz = 1;
//   for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
//     data_sz *= x1_shape->GetStorageShape().GetDim(i);
//   tiling.set_size(data_sz);
  context->SetBlockDim(coreNum);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

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
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class Lcm : public OpDef {
public:
    explicit Lcm(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("other")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Lcm);
}
