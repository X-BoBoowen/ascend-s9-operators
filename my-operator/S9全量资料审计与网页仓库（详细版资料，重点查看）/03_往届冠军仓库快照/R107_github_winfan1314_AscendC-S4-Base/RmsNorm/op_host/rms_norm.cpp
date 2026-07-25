#include "rms_norm_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <vector>

#define SET(param) tiling.set_##param(param)
#define ALIGN32(mem) ((mem) / 32u * 32u)

#define CEIL(x, align_num) (((x) + (align_num) - 1) / (align_num) * (align_num))
#define FLOOR(x, align_num) ((x) / (align_num) * (align_num))
using std::max;
using std::min;
using namespace ge;
constexpr uint64_t BUFFER_NUM = 2;
constexpr uint64_t MINI_BATCH_SIZE = 32;

using namespace std;

namespace optiling {
    const uint32_t BLOCK_SIZE = 32;

    struct TilingArg {
        uint32_t tileNum;
        uint32_t tileLength;
        uint32_t lastTileLength;
    };

    uint32_t getTotalLengthAligned(uint32_t totalLength, uint32_t sizeOfDatatype){
        uint32_t totalLengthAligned;
        uint32_t ALIGN_NUM = BLOCK_SIZE / sizeOfDatatype;
        if (totalLength % ALIGN_NUM != 0) {  //不对齐，先32位对齐
            totalLengthAligned = ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
        } else {
            totalLengthAligned = totalLength;
        }
        return totalLengthAligned;
    }

    // 获得 tile 参数，单 buffer
    TilingArg getTilingArg(uint32_t lengthAligned, uint64_t ub_block_num, uint32_t sizeOfDatatype){
        uint32_t tileNum;
        uint32_t tileLength = 0;
        uint32_t lastTileLength = 0;
        uint32_t ALIGN_NUM = BLOCK_SIZE / sizeOfDatatype;

        tileNum = lengthAligned / ALIGN_NUM / ub_block_num;
        if (tileNum == 0) {
            tileNum = 1;
            tileLength = lengthAligned;
            lastTileLength = tileLength;
        } else if((lengthAligned / ALIGN_NUM) % ub_block_num == 0){
            tileLength = ub_block_num * ALIGN_NUM;
            lastTileLength = tileLength;
        }else{
            tileNum = tileNum + 1;
            tileLength = ub_block_num * ALIGN_NUM;
            lastTileLength = lengthAligned - (tileNum - 1) * tileLength;
        }
        TilingArg arg;
        arg.tileNum = tileNum;
        arg.tileLength = tileLength;
        arg.lastTileLength = lastTileLength;

        return arg;
    }

    void printShape(vector<int32_t> &x_shape){
        printf("shape: ");
        for(int i = 0; i < x_shape.size(); ++i){
            printf("%d ", x_shape[i]);
        }
        printf("\n");
    }

    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        // 1. 获取平台信息
        uint64_t ub_size;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto socVersion = ascendcPlatform.GetSocVersion();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        auto aivNum = 1;  // vector core num  1
        ub_size -= 512; // 空余 256 字节来存放每一次迭代的 rsvt, 256 作为 1 常向量。

        // 2. 获取数据信息
        uint32_t x1TotalLength = context->GetInputTensor(0)->GetShapeSize();
        uint32_t x2TotalLength = context->GetInputTensor(1)->GetShapeSize();
        auto dt = context->GetInputTensor(0)->GetDataType();
        RmsNormTilingData tiling;

        // 获取 shape 信息
        gert::Shape x1OriginShape = context->GetInputTensor(0)->GetOriginShape();
        gert::Shape x2OriginShape = context->GetInputTensor(1)->GetOriginShape();
        int32_t x1ShapeSize = x1OriginShape.GetDimNum();
        int32_t x2ShapeSize = x2OriginShape.GetDimNum();
        vector<int32_t> x1Shape(x1ShapeSize), x2Shape(x2ShapeSize);
        for(int i = 0; i < x1ShapeSize; ++i){
            x1Shape[i] = x1OriginShape[i];
        }
        printf("x1 shape:\n");
        printShape(x1Shape);
        for(int i = 0; i < x2ShapeSize; ++i){
            x2Shape[i] = x2OriginShape[i];
        }
        printf("x2 shape:\n");
        printShape(x2Shape);
        // 得到轴信息
        const float *pEpsilon = context->GetAttrs()->GetFloat(0);
        float epsilon = *pEpsilon;

        printf("epsilon = %f\n", epsilon);

        uint32_t batchNum = 1;
        for(uint32_t i = 0; i < x1ShapeSize - x2ShapeSize; ++i){
            batchNum *= x1Shape[i];
        }
        uint32_t batchLength = x1TotalLength / batchNum;
        uint32_t rstdTileLength;
        if(dt == ge::DT_FLOAT){
            rstdTileLength = 64;
        }else{
            rstdTileLength = 128;
        }
        uint32_t rstdTileNum = batchNum / rstdTileLength;
        uint32_t rstdLastTileLength = rstdTileLength;
        rstdLastTileLength = batchNum - rstdTileNum * rstdTileLength;
        rstdTileNum += 1;

        printf("x1TotalLength = %d\n", x1TotalLength);
        printf("x2TotalLength = %d\n", x2TotalLength);
        printf("batchNum = %d\n", batchNum);
        printf("batchLength = %d\n", batchLength);
        printf("rstdTileNum = %d\n", rstdTileNum);
        printf("rstdTileLength = %d\n", rstdTileLength);
        printf("rstdLastTileLength = %d\n", rstdLastTileLength);
        
        tiling.set_x1TotalLength(x1TotalLength);
        tiling.set_x2TotalLength(x2TotalLength);
        tiling.set_epsilon(epsilon);
        tiling.set_batchNum(batchNum);
        tiling.set_batchLength(batchLength);
        tiling.set_rstdTileNum(rstdTileNum);
        tiling.set_rstdTileLength(rstdTileLength);
        tiling.set_rstdLastTileLength(rstdLastTileLength);

        bool isBroadCast = false;
        for(int32_t i = 1; i <= x2ShapeSize; ++i){
            if(x1Shape[x1ShapeSize - i] != x2Shape[x2ShapeSize - i]){
                isBroadCast = true;
                break;
            }
        }

        // 3. 计算 Tiling 参数
        uint32_t sizeOfDataType;
        printf("Not Broadcast!\n");
        if(dt == ge::DT_FLOAT){
            context->SetTilingKey(0);
            printf("--------[FLOAT]--------\n");
            sizeOfDataType = 4;
            uint32_t data_num = 6;
            uint32_t ub_block_num = ub_size / BLOCK_SIZE / data_num;
            uint32_t batchLengthAligned = getTotalLengthAligned(batchLength, sizeOfDataType);
            TilingArg args = getTilingArg(batchLengthAligned, ub_block_num, sizeOfDataType);
            uint32_t tileNum = args.tileNum;
            uint32_t tileLength = args.tileLength;
            uint32_t lastTileLength = args.lastTileLength;
            uint32_t resLength = batchLengthAligned - batchLength;

            printf("tileNum = %d\n", tileNum);
            printf("tileLength = %d\n", tileLength);
            printf("lastTileLength = %d\n", lastTileLength);
            printf("resLength = %d\n", resLength);

            tiling.set_tileNum(tileNum);
            tiling.set_tileLength(tileLength);
            tiling.set_lastTileLength(lastTileLength);
            tiling.set_resLength(resLength);
        }else if(dt == ge::DT_FLOAT16){
            context->SetTilingKey(1);
            printf("--------[FLOAT16]--------\n");
            sizeOfDataType = 2;
            uint32_t data_num = 10;
            ub_size = ub_size - 512;
            uint32_t ub_block_num = ub_size / BLOCK_SIZE / data_num;
            uint32_t batchLengthAligned = getTotalLengthAligned(batchLength, sizeOfDataType);
            TilingArg args = getTilingArg(batchLengthAligned, ub_block_num, sizeOfDataType);
            uint32_t tileNum = args.tileNum;
            uint32_t tileLength = args.tileLength;
            uint32_t lastTileLength = args.lastTileLength;
            uint32_t resLength = batchLengthAligned - batchLength;
            uint32_t mulLastTileLength = lastTileLength;

            if(resLength > 0){
                mulLastTileLength = mulLastTileLength - 16;
            }

            printf("tileNum = %d\n", tileNum);
            printf("tileLength = %d\n", tileLength);
            printf("lastTileLength = %d\n", lastTileLength);
            printf("resLength = %d\n", resLength);
            printf("mulLastTileLength = %d\n", mulLastTileLength);

            tiling.set_tileNum(tileNum);
            tiling.set_tileLength(tileLength);
            tiling.set_lastTileLength(lastTileLength);
            tiling.set_resLength(resLength);
            tiling.set_mulLastTileLength(mulLastTileLength);
        }else if(dt == ge::DT_BF16){
            context->SetTilingKey(2);
            printf("--------[BF16]--------\n");
            sizeOfDataType = 2;
            uint32_t data_num = 10;
            ub_size = ub_size - 256;
            ub_size = ub_size - 512 * 2 - 32;
            uint32_t ub_block_num = ub_size / BLOCK_SIZE / data_num;
            uint32_t batchLengthAligned = getTotalLengthAligned(batchLength, sizeOfDataType);
            TilingArg args = getTilingArg(batchLengthAligned, ub_block_num, sizeOfDataType);
            uint32_t tileNum = args.tileNum;
            uint32_t tileLength = args.tileLength;
            uint32_t lastTileLength = args.lastTileLength;
            uint32_t resLength = batchLengthAligned - batchLength;

            printf("tileNum = %d\n", tileNum);
            printf("tileLength = %d\n", tileLength);
            printf("lastTileLength = %d\n", lastTileLength);
            printf("resLength = %d\n", resLength);

            tiling.set_tileNum(tileNum);
            tiling.set_tileLength(tileLength);
            tiling.set_lastTileLength(lastTileLength);
            tiling.set_resLength(resLength);
        }else{
            printf("Unknow!\n");
        }

        context->SetBlockDim(aivNum);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        return ge::GRAPH_SUCCESS;
    }
}


namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context) {
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
        this->Output("rstd")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6f);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(RmsNorm);
}
