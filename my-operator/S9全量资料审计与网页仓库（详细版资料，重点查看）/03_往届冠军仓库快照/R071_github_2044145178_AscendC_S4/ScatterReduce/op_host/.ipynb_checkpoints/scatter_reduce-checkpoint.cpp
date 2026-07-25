#include "scatter_reduce_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


#include<iostream>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  ScatterReduceTilingData tiling;

    const auto runtime_attrs=context->GetAttrs();
    const int dim=*(runtime_attrs->GetInt(0));
    const char *reduce=runtime_attrs->GetStr(1);
    const int8_t includeSelf=*(runtime_attrs->GetBool(2));

    
//     std::cerr << "type: " << reduce[0] << std::endl;
    
//     std::cerr << "dim: " << dim << std::endl;
    
    int type=0;
    if(reduce[0]=='s'){
        type=0;
    }else if(reduce[0]=='p'){
      type=1;
    }else if(reduce[0]=='m'){
        type=2;
    }else if(reduce[2]=='i'){
        type=3;
    }else if(reduce[2]=='a'){
            type=4;
    }else{
        std::cerr << "type: " << reduce[0] << std::endl;

    }

    tiling.set_includeSelf(includeSelf);
    tiling.set_dim(dim);
    tiling.set_type(type);


    uint32_t inputSize = context->GetInputTensor(0)->GetShapeSize();

    tiling.set_inputSize(inputSize);

    
    // std::cerr << "inputSize: " << inputSize << std::endl;
    
    
    const auto shape0=context->GetInputTensor(0)->GetOriginShape();


    uint16_t mmInputDims[8];
    int nInputDims=shape0.GetDimNum();
    for(int i=0;i<shape0.GetDimNum();i++){
        mmInputDims[i]=shape0.GetDim(i);
    }
    tiling.set_nInputDims(nInputDims);
    tiling.set_mmInputDims(mmInputDims);


    auto dt = context->GetInputTensor(0)->GetDataType(); //  ge::DT_FLOAT
    int DataTypeSize;
    if(dt==ge::DT_FLOAT ){
        DataTypeSize=4;
    }else{
        DataTypeSize=2;
    }
    
    int KS=1;
    for(int i=nInputDims-1;i>dim;i--){
        KS*=mmInputDims[i];
    }
    int HI=inputSize/KS/mmInputDims[dim];
    uint32_t J=mmInputDims[dim];

    //一行32B块数
    uint32_t blockNum=KS*DataTypeSize/32;


    //32B对齐
    uint32_t lineSize_max=20000/J/32*32/DataTypeSize;
    //空间：8*J*lineSize*(T)<180K
    //一次最多出来B？
    uint32_t lineSize=std::min(blockNum*32/DataTypeSize,lineSize_max);


    int totalTilingNum=HI*(KS/lineSize);
    //32B尾块有多少元素，负责倒数第一个32B整块的核心需要处理该尾块
    // int tailBlockSize=KS-KS/(32/DataTypeSize)*(32/DataTypeSize);

    tiling.set_totalTilingNum(totalTilingNum);
    tiling.set_KS(KS);
    tiling.set_J(J);
    tiling.set_lineSize(lineSize);


    
    auto ascendcPlatform = platform_ascendc:: PlatformAscendC(context->GetPlatformInfo());

    uint32_t  aivNum = std::min((uint32_t)ascendcPlatform.GetCoreNumAiv(),(uint32_t)totalTilingNum);

    if(KS*DataTypeSize>=32){
        context->SetBlockDim(aivNum);
        if(reduce[0]=='s'){
            context->SetTilingKey(2);
        }else if(reduce[0]=='p'){
            context->SetTilingKey(3);
        }else if(reduce[0]=='m'){
            context->SetTilingKey(4);
        }else if(reduce[2]=='i'){
            context->SetTilingKey(5);
        }else if(reduce[2]=='a'){
            context->SetTilingKey(6);
        }
    }else{
        size_t usrSize = sizeof(int)*inputSize; // 设置用户需要使用的workspace大小。
        // 如需要使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace的大小。
        uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
        currentWorkspace[0] = usrSize + sysWorkspaceSize; // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
        context->SetBlockDim(1);
        context->SetTilingKey(1);

    }
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
