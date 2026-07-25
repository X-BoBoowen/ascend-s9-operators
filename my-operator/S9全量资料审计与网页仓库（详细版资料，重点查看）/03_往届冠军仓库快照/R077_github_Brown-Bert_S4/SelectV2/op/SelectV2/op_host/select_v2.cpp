
#include "select_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    SelectV2TilingData tiling;
    // 通过context获取用户传入的数据大小
    uint64_t dim0 = context->GetInputShape(1)->GetStorageShape().GetDim(0);
    uint64_t dim1 = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint64_t interval = context->GetInputShape(2)->GetStorageShape().GetDim(1) * context->GetInputShape(2)->GetStorageShape().GetDim(2);
    // 获取数据类型
    auto dt = context->GetInputTensor(1)->GetDataType();
    if (dt == 3){
        tiling.set_isBroadCast(1);
        tiling.set_rows(dim0);
        tiling.set_interval(interval);
    }

    // auto dt1 = context->GetInputTensor(2)->GetStorageShape().GetShapeSize();
    // auto dt2 = context->GetInputTensor(0)->GetStorageShape().GetShapeSize();
    // if (dt1 != dt2){
    //     tiling.set_flag(1);
    // }else{
    //     tiling.set_flag(0);
    // }

    // 在compute接口中，每次计算需要同时消耗多少个LocalTensor
    uint64_t ubDataNum = 8; // 是根据具体的代码逻辑设定的，不是通过算出来的

    uint64_t inputNum = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    if (dt == 3){
        inputNum = context->GetInputShape(2)->GetStorageShape().GetShapeSize();
    }

    // 获取UB内存大小
    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  //   std::cout << "ubsize = " << ubSize << std::endl;
  
    // 获取AiCore的物理核数
    auto coreNum = ascendcPlatform.GetCoreNum();
  
    // 用户输入数据的类型大小（以字节为单位）
    uint32_t inputBytes = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(1)->GetDataType(), inputBytes);
  
    // 数据的总大小
    uint64_t inputLength = inputNum * inputBytes;
  
  
    // aicore每次最多处理多少block块
    uint64_t blockNum = ubSize / 32 / 2 / ubDataNum;
  
    // 根据每次最多处理的block块，转换成每次最多处理的数据个数
    uint64_t dataNum = blockNum * 32 / inputBytes;
  
    // 根据32字节对齐计算数据的字节数
    uint64_t inputLengthAlgin32 = (inputLength + 32 - 1) / 32 * 32;
  //   inputLengthAlgin32 = 320; // warng answer
  
    // 计算数据需要几个core去执行，如果数据量太小就不需要全部的core去执行
    coreNum = (coreNum < inputLengthAlgin32 / 32) ? coreNum : inputLengthAlgin32 / 32;
    coreNum = (coreNum >= 1) ? coreNum : 1;
  
    // 计算每个core需要处理多少个block
    uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / 32 / coreNum;
  //   everyCoreInputBlockNum = 10; // warng answer
  
    // 上面一行不是整除的话，会剩余几个block
    uint64_t tailBlockNum = inputLengthAlgin32 / 32 % coreNum;
  
    /*
        小核
    */
  
    // 计算小核要处理的数据量
    uint64_t smallCoreDataNum = everyCoreInputBlockNum * 32 / inputBytes;
  
    // 计算小核，根据每个小核要处理多少block，但是一次只能处理多少个block，计算出需要计算多少次
    uint64_t smallCoreCount = everyCoreInputBlockNum / blockNum;
    uint64_t smallCoreCarryNum = (everyCoreInputBlockNum % blockNum == 0) ? smallCoreCount : (smallCoreCount + 1);
  
    // 计算小核最后一次需要处理多少数据
    uint64_t smallCoreFinallDealNum = smallCoreDataNum - (dataNum * smallCoreCount);
    smallCoreFinallDealNum = (smallCoreFinallDealNum == 0) ? dataNum : smallCoreFinallDealNum;
  
  
  //   std::cout << "smallCoreDataNum = " << smallCoreDataNum << std::endl;
  //   std::cout << "smallCoreCarryNum = " << smallCoreCarryNum << std::endl;
  //   std::cout << "smallCoreFinallDealNum = " << smallCoreFinallDealNum << std::endl;
  //   std::cout << "smallCoreCount = " << smallCoreCount << std::endl;
  
  
  
    /**
        大核
     */
    everyCoreInputBlockNum++;
    uint64_t bigCoreDataNum = everyCoreInputBlockNum * 32 / inputBytes;
  
    uint64_t bigCoreCount = everyCoreInputBlockNum / blockNum;
    uint64_t bigCoreCarryNum = (everyCoreInputBlockNum % blockNum == 0) ? bigCoreCount : (bigCoreCount + 1);
  
    uint64_t bigCoreFinallDealNum = bigCoreDataNum - (dataNum * bigCoreCount);
    bigCoreFinallDealNum = (bigCoreFinallDealNum == 0) ? dataNum : bigCoreFinallDealNum;
  
    // 将上述计算的值全部回填到tiling中
    tiling.set_smallCoreDataNum(smallCoreDataNum);
    tiling.set_bigCoreDataNum(bigCoreDataNum);
    tiling.set_smallCoreCarryNum(smallCoreCarryNum);
    tiling.set_bigCoreCarryNum(bigCoreCarryNum);
    tiling.set_tileDataNum(dataNum);
    tiling.set_smallCoreFinallDealNum(smallCoreFinallDealNum);
    tiling.set_bigCoreFinallDealNum(bigCoreFinallDealNum);
    tiling.set_bigCoreNum(tailBlockNum);
    tiling.set_dataType(dt);
  
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    // size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    // currentWorkspace[0] = 0;
    context->SetBlockDim(coreNum);
  
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
class SelectV2 : public OpDef {
public:
    explicit SelectV2(const char* name) : OpDef(name)
    {
        this->Input("condition")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(SelectV2);
}
