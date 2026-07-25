
#include "rms_norm_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include <algorithm>

constexpr int32_t BUFFER_NUM = 2;
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  RmsNormTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  const gert::Shape shape = x1_shape->GetStorageShape();

  auto rowNum = shape.GetDim(0); // 矩阵总行数
  auto colNum = shape.GetDim(1); // 每行数据量
  
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto socVersion = ascendcPlatform.GetSocVersion();
  uint64_t ub_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

  auto dt = context->GetInputDesc(0)->GetDataType();
  uint64_t sizeofdatatype;
  uint32_t tileLoop;
  if(dt == ge::DT_FLOAT16 || dt == ge::DT_BF16){       // 从D与ub反推极限N
      sizeofdatatype = 2; // 16ND + 16D + 12N 
      tileLoop = (ub_size - 16 * colNum) / (16 * colNum + 12);
  }else{
      sizeofdatatype = 4; // 20ND + 16D + 16N
      tileLoop = (ub_size - 16 * colNum) / (20 * colNum + 16);
  }
  const float* epsilonAttr = context->GetAttrs()->GetAttrPointer<float>(0);
  tileLoop = std::min((int)tileLoop, (int)rowNum);
  uint32_t rstdLength;
  if(tileLoop * sizeofdatatype % 32 != 0){            // rstd的datacopy需要满足32B，否则无法拷
      if(tileLoop * sizeofdatatype < 32){
          rstdLength = 32 / sizeofdatatype;
      }else{
          rstdLength = ((tileLoop * sizeofdatatype / 32 + 1) * 32) / sizeofdatatype;
      }
  }
  uint32_t rstdGmLength;
  if(rowNum * sizeofdatatype % 32 != 0){
      if(rowNum * sizeofdatatype < 32){
          rstdGmLength = 32 / sizeofdatatype;
      }else{
          rstdGmLength = ((rowNum * sizeofdatatype / 32 + 1) * 32) / sizeofdatatype;
      }
  }
  tiling.set_rstdGmLength(rstdGmLength);
  tiling.set_rstdLength(rstdLength);
  tiling.set_totalLength(rowNum * colNum);            // 总数据量
  tiling.set_tileLoop(tileLoop);                      // 核内单次计行数极限
  tiling.set_maxPerTime(tileLoop * colNum);           // 核内单次计算数据量,用于datacopy时长度
  tiling.set_loopCount(rowNum / tileLoop);            // 核内计算总需循环数
  tiling.set_leftNum(rowNum % tileLoop);              // 如果有余行，单独计算一下
  tiling.set_leftPerTime(rowNum % tileLoop * colNum); // 余块计算长度 
  tiling.set_tileLength(colNum);                      // 每行长度
  tiling.set_tileNum(rowNum);                         // 矩阵总行数
  tiling.set_factor(1.0f / colNum);                   // 行数倒数 
  tiling.set_eps(*epsilonAttr);

//   printf("rstdGmLength: %d\n",rstdGmLength);
//   printf("rstdLength: %d\n",rstdLength);
//   printf("ubsize: %d\n",ub_size);
//   printf("totalLength: %llu\n", rowNum * colNum);
//   printf("tileLoop: %u\n", tileLoop);
//   printf("maxPerTime: %llu\n", tileLoop * colNum);
//   printf("loopCount: %u\n", rowNum / tileLoop);
//   printf("leftNum: %u\n", rowNum % tileLoop);
//   printf("leftPerTime: %llu\n", (rowNum % tileLoop) * colNum);
//   printf("tileLength: %llu\n", colNum);
//   printf("tileNum: %llu\n", rowNum);
//   printf("factor: %f\n", 1.0f / colNum);
//   printf("eps: %f\n", *epsilonAttr);
  context->SetBlockDim(1);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
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
class RmsNorm : public OpDef {
public:
    explicit RmsNorm(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("rstd")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-06);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(RmsNorm);
}
