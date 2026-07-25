
#include "glu_jvp_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define BLOCK_SIZE 32

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    GluJvpTilingData tiling;
    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); //获取硬件平台存储空间 UB 的内存大小

    // const gert::StorageShape* x1_shape = context->GetInputShape(0);
    // int32_t data_sz = 1;
//   for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
//     data_sz *= x1_shape->GetStorageShape().GetDim(i);

    // 获取dim参数
    const int64_t *axesVal = context->GetAttrs()->GetInt(0);
    int64_t axes = *axesVal;
    uint32_t dims = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint32_t totalLength = context->GetInputTensor(1)->GetShapeSize();
    uint32_t typeSize = GetSizeByDataType(context->GetInputDesc(1)->GetDataType()); //输入类型
    uint32_t totalBytes = context->GetInputTensor(1)->GetShapeSize();
    auto dt = context->GetInputTensor(1)->GetDataType();
    uint32_t ubDataNumber = 5*2 + 4;
    if (dt == ge::DT_FLOAT16) {
        ubDataNumber = 23;
    }  else if(dt == ge::DT_FLOAT){
        ubDataNumber = 8;
    } else {
        ubDataNumber = 23;
    }
    // ubDataNumber *= 2;
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber; //每个ub段可用的空间块数
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeSize; //每次处理的数据量
    // uint32_t totalBytesAlgin32 = (((totalBytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE); //输入长度 对齐处理
    uint32_t stride = 1;
    uint32_t shapeDim[12];
    if(axes<0) axes += dims;
    // if (axes==dims-1) return -1;

    for(int i=0;i<dims;i++) {
        shapeDim[i] = context->GetInputShape(1)->GetStorageShape().GetDim(i);
        // printf("shape i:%d %d\n", i, shapeDim[i]);
        // if (shapeDim[i] != context->GetInputShape(2)->GetStorageShape().GetDim(i)) {
        //     return -1;
        // }
    }
    for(int i=axes+1;i<dims;i++) stride *= shapeDim[i];
    //确定操作Iter数
    uint32_t iterStep = stride * shapeDim[axes];
    uint32_t nIter = 1;
    for(int i=0;i<axes;i++) nIter *= shapeDim[i];
    // 结果直接对半折
    // uint32_t resLength = stride * nIter/2;
    // 每次需要处理iterstep/2数量的数据
    uint32_t iterStepBytes = iterStep*typeSize/2;
    uint32_t iterStepBytesAlgin32 = (((iterStepBytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE); //输入长度 对齐处理
    uint32_t everyCoreInputBlockNum = iterStepBytesAlgin32 / BLOCK_SIZE;// 输入数据需要多少空间块    
    uint32_t CoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeSize; //对齐空间后的输入数量
    uint32_t TileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? TileNum : TileNum + 1; //需要循环处理几次
    uint32_t tailDataNum = CoreDataNum - (tileDataNum * TileNum);
    uint32_t padtimes = 1;
    if (2*everyCoreInputBlockNum < tileBlockNum) {
        padtimes = tileBlockNum/everyCoreInputBlockNum;
    }
    // printf("coreDataNum %d TileNum:%d tileDataNum:%d tailDataNum:%d\n", CoreDataNum,TileNum,tileDataNum,tailDataNum);
    // printf("tileDataNum %d\n",tileDataNum);
    tailDataNum = tailDataNum == 0 ? tileDataNum : tailDataNum; // 最后一次处理的数据量
    tiling.set_coreDataNum(CoreDataNum);  //对齐空间后的输入数量
    tiling.set_finalTileNum(finalTileNum);//需要循环处理几次
    tiling.set_tileDataNum(tileDataNum); //每次处理的数据量
    tiling.set_tailDataNum(tailDataNum); //最后一次需要处理的数据量
    tiling.set_iterStep(iterStep);          //组间地址偏移
    tiling.set_padtimes(padtimes);          //padtimes

    tiling.set_stride(stride);              //比较步长(也是单组内需要比较的次数)
    // tiling.set_dims(dims);  
    tiling.set_axesDim(shapeDim[axes]);     //每次需要比较的数值个数

    int coreNum = 40;
    int smallDataNum;
    int tail;
    if (coreNum > nIter) coreNum = nIter;
    smallDataNum = nIter/coreNum;
    tail = nIter % coreNum;
    tiling.set_smallBatch(smallDataNum);
    tiling.set_tail(tail);
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
}


namespace ops {
class GluJvp : public OpDef {
public:
    explicit GluJvp(const char* name) : OpDef(name)
    {
        this->Input("glu_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("jvp_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").Int();

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(GluJvp);
}
