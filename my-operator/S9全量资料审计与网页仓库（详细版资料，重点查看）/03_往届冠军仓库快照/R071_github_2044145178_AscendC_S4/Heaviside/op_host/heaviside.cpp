
#include "heaviside_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
  static ge::graphStatus TilingFuncBroadcast(gert::TilingContext* context)
  {
    context->SetTilingKey(5);

    HeavisideTilingData_BroadCast tiling;

    const auto shape0=context->GetInputTensor(0)->GetOriginShape();
    const auto shape1=context->GetInputTensor(1)->GetOriginShape();

    uint8_t nOutputDims=std::max(shape0.GetDimNum(),shape1.GetDimNum());


      

    uint16_t arr0[8];
    for(int i=0;i<nOutputDims;i++){  
        if(shape0.GetDimNum()+i>=nOutputDims){
          arr0[i]=shape0.GetDim(shape0.GetDimNum()-nOutputDims+i);
        }else{
          arr0[i]=1;
        }
                // std::cerr << "###inputDims: " << arr0[i] << std::endl;

    }
    tiling.set_mmInputDims(arr0);

    uint16_t arr1[8];
    for(int i=0;i<nOutputDims;i++){  
                                          // std::cerr << "###$$$$$i: " << i << std::endl;

      if(shape1.GetDimNum()+i>=nOutputDims){

        arr1[i]=shape1.GetDim(shape1.GetDimNum()-nOutputDims+i);
                                            // std::cerr << "##$$$#valuesDims: " << arr1[i] << std::endl;

      }else{

        arr1[i]=1;
                                            // std::cerr << "###$$$valuesDims: " << arr1[i] << std::endl;

      }

    }
    tiling.set_mmValuesDims(arr1);
            
    uint16_t arr[8];
    uint32_t outputSize=1;
    for(int i=0;i<nOutputDims;i++){     
        // std::cerr << "###i: " << i << std::endl;
        arr[i]=std::max(arr0[i],arr1[i]);
        outputSize*=arr[i];
        // std::cerr << "###OutputDims: " << arr[i] << std::endl;
    }
   
      
      
    tiling.set_size(outputSize);
    tiling.set_nOutputDims(nOutputDims);
    tiling.set_mmOutputDims(arr);
      
      
//       std::cerr << "output_sz: " << outputSize << std::endl;
//       std::cerr << "nInputDims: " << shape0.GetDimNum() << std::endl;
//       std::cerr << "nValuesDims: " << shape1.GetDimNum() << std::endl;
//       std::cerr << "nOutputDims: " <<  nOutputDims << std::endl;
  
    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  
  return ge::GRAPH_SUCCESS;
  }
  static ge::graphStatus TilingFunc(gert::TilingContext* context)
  {

    // auto ret = context->SetNeedAtomic(true);


    uint32_t size = context->GetInputTensor(0)->GetShapeSize();
    uint32_t valuesSize = context->GetInputTensor(1)->GetShapeSize();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = 20;//最多启动20个核心

    // std::cerr << "aivNum: " <<  aivNum << std::endl;
    // std::cerr << "coreNum: " <<  ascendcPlatform.GetCoreNum()<< std::endl;

    if(valuesSize!=1){
      const auto shape0=context->GetInputTensor(0)->GetOriginShape();
      const auto shape1=context->GetInputTensor(1)->GetOriginShape();
      uint8_t nOutputDims=std::max(shape0.GetDimNum(),shape1.GetDimNum());
      if(shape0.GetDimNum()!=shape1.GetDimNum() ){
        return TilingFuncBroadcast(context);
      }else{
        for(int i=0;i<nOutputDims;i++){
          if(shape0.GetDim(i)!=shape1.GetDim(i)){
            return TilingFuncBroadcast(context);
          }
        }
      }
    }






    auto dt = context->GetInputTensor(0)->GetDataType(); //  ge::DT_FLOAT
    int DataTypeSize=0;

    if(dt==ge::DT_FLOAT){
      DataTypeSize=4;
    }else{
      DataTypeSize=2;
    }
    constexpr uint32_t blockSize=512;
    uint32_t blockNum=(size*DataTypeSize+blockSize-1)/blockSize;
    aivNum=std::min(aivNum,blockNum);

    uint32_t smallSize=blockNum/aivNum*blockSize/DataTypeSize;
    uint16_t incSize=blockSize/DataTypeSize;
    uint16_t formerNum=blockNum%aivNum;



    if((smallSize+incSize)*DataTypeSize*3<=160000){
      if(valuesSize==1){
        context->SetTilingKey(3);
      }else{
        context->SetTilingKey(1);
      }
    }else{
      if(valuesSize==1){
        context->SetTilingKey(4);
      }else{
        context->SetTilingKey(2);
      }
    }
    HeavisideTilingData tiling;
    tiling.set_smallSize(smallSize);
    tiling.set_incSize(incSize);
    tiling.set_formerNum(formerNum);

    context->SetBlockDim(aivNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
  }

}


// namespace ge {
// static ge::graphStatus InferShape(gert::InferShapeContext* context)
// {
//     const gert::Shape* x1_shape = context->GetInputShape(0);
//     gert::Shape* y_shape = context->GetOutputShape(0);
//     *y_shape = *x1_shape;
//     return GRAPH_SUCCESS;
// }
// }


namespace ge {
  static ge::graphStatus InferShape(gert::InferShapeContext* context)
  {
       gert::Shape* shape0 = const_cast<gert::Shape*>(context->GetInputShape(0));
       gert::Shape* shape1 = const_cast<gert::Shape*>(context->GetInputShape(1));
      gert::Shape* y_shape = context->GetOutputShape(0);
      
  
      int nOutputDims=std::max(shape0->GetDimNum(),shape1->GetDimNum());
      int  diff =  nOutputDims-std::min(shape0->GetDimNum(),shape1->GetDimNum());
      
      gert::Shape* shortShape;
      gert::Shape* longShape;
      
      if(shape0->GetDimNum()>shape1->GetDimNum()){
          longShape=shape0;
          shortShape=shape1;
      }else{
          longShape=shape1;
          shortShape=shape0;
      }
      
      y_shape->SetDimNum(nOutputDims);
      
      for(int i=nOutputDims-1;i>=diff;i--){          
          y_shape->SetDim(i,std::max(longShape->GetDim(i),shortShape->GetDim(i-diff)));
          // std::cerr << "###InferShape OutputDims: " << std::max(longShape->GetDim(i),shortShape->GetDim(i-diff)) << std::endl;
      }
      for(int i=diff-1;i>=0;i--){          
         y_shape->SetDim(i,longShape->GetDim(i));
          // std::cerr << "###InferShape OutputDims: " << longShape->GetDim(i) << std::endl;
      }
      return GRAPH_SUCCESS;
  }
  }


namespace ops {
class Heaviside : public OpDef {
public:
    explicit Heaviside(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            // .InitValue(0);//清0


        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Heaviside);
}
