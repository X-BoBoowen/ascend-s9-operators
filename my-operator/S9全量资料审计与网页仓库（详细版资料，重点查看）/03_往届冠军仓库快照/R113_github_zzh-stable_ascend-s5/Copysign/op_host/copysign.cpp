
#include "common.h"
#include "copysign_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        // printf("into tilingfunc\n");
    
        const gert::StorageShape* x1_shape = context->GetInputShape(0);
        const gert::StorageShape* x2_shape = context->GetInputShape(1);
    
        uint32_t x1dimNum, x2dimNum;
        x1dimNum = x1_shape->GetStorageShape().GetDimNum();
        x2dimNum = x2_shape->GetStorageShape().GetDimNum();
        int maxDim = x1dimNum > x2dimNum ? x1dimNum : x2dimNum;
    
        uint32_t shapeInf1[MAXDIM] = {};
        uint32_t shapeInf2[MAXDIM] = {};
    
        uint32_t zipInf1[MAXDIM] = {};
        uint32_t zipInf2[MAXDIM] = {};
    
        // 补齐维度
        int dimOffset = maxDim - x1dimNum;
        int dim_i = 0;
        for (dim_i = 0; dim_i < dimOffset; dim_i++) {
            shapeInf1[dim_i] = 1;
        }
        for (; dim_i < maxDim; dim_i++) {
            shapeInf1[dim_i] = x1_shape->GetStorageShape().GetDim(dim_i - dimOffset);
        }
    
        dimOffset = maxDim - x2dimNum;
        for (dim_i = 0; dim_i < dimOffset; dim_i++) {
            shapeInf2[dim_i] = 1;
        }
        for (; dim_i < maxDim; dim_i++) {
            shapeInf2[dim_i] = x2_shape->GetStorageShape().GetDim(dim_i - dimOffset);
        }
    
        // 压缩相同广播状态的维度
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
    #ifdef OPEN_LOG
        // printf("shapeInf:\n");
        // for (int i = 0; i < maxDim; i++) {
        //     printf("%d ", shapeInf1[i]);
        // }
        // printf("\n");
        // for (int i = 0; i < maxDim; i++) {
        //     printf("%d ", shapeInf2[i]);
        // }
        // printf("\n");
        // printf("zipInf:\n");
        // for (int i = 0; i < zipDimNum; i++) {
        //     printf("%d ", zipInf1[i]);
        // }
        // printf("\n");
        // for (int i = 0; i < zipDimNum; i++) {
        //     printf("%d ", zipInf2[i]);
        // }
        // printf("\n");
    #endif
    
        uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(0)->GetDataType());
        int blockSize = BLOCK_BYTES_SIZE / inputBytes;
    
        bool allEqual = (zipDimNum == 1 && zipInf1[0] == zipInf2[0]) ? 1 : 0;
    
        if (allEqual) {
            CopysignTilingData_0 tiling;
            int totalNum = zipInf1[0];
            tiling.set_totalNum(totalNum);
            int totalBlockNum = (totalNum + blockSize - 1) / blockSize;
            int coreNum = totalBlockNum < CORENUM ? totalBlockNum : CORENUM;
    
            context->SetBlockDim(coreNum);
            if (1) {
                // 使用向量拷贝与向量计算
                context->SetTilingKey(0);
            } else {
                // 使用向量拷贝与标量计算
                context->SetTilingKey(1);
            }
    
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                                context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        } else if (zipDimNum == 3 && zipInf1[1] != zipInf2[1]) {
            // 中间维度需要广播
            CopysignTilingData_2 tiling;
            uint16_t shapeInf[3] = {zipInf1[0], zipInf1[1], zipInf1[2]};
            tiling.set_shapeInf(shapeInf);
            int coreNum = 1;
            if (zipInf1[0] >= 40) {
                // 第0维大于等于40，可以从第0维进行多核划分
                context->SetTilingKey(2);
                coreNum = zipInf1[0] < CORENUM ? zipInf1[0] : CORENUM;
            } else {
                // 第0维小于40，需要从第1维进行多核划分
                context->SetTilingKey(3);
                coreNum = zipInf1[1] < CORENUM ? zipInf1[1] : CORENUM;
            }
            context->SetBlockDim(coreNum);
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                                context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        } else {
            // 标量处理任意广播形状
            context->SetTilingKey(4);
    
            CopysignTilingData tiling;
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


namespace ops {
class Copysign : public OpDef {
   public:
    explicit Copysign(const char* name) : OpDef(name) {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

            this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Copysign);
}  // namespace ops
