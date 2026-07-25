
#include "fmin_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define BUFFER_NUM 2
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define ROUND_UP(a, b) (((a) + (b) - 1) / (b) * (b))
#define ROUND_DOWN(a, b) ((a) / (b) * (b))
#include <iostream>
#include <fstream>

#define ENABLE_LOG 0 // 1
#define log_file_name "/home/ma-user/work/log.txt"


static inline void print_shape(const gert::StorageShape *shape)
{
#if ENABLE_LOG
    int dimnum = shape->GetStorageShape().GetDimNum();
    std::cout << "dimnum: " << dimnum << std::endl;
    for (int i = 0; i < dimnum; i++) {
        std::cout << "dim[" << i << "]: " << shape->GetStorageShape().GetDim(i) << std::endl;
    }
#endif
}

static inline bool needBroadcast(const gert::StorageShape *x_shape, const gert::StorageShape *y_shape)
{
    bool need = false;

    int x_dimnum = x_shape->GetStorageShape().GetDimNum();
    int y_dimnum = y_shape->GetStorageShape().GetDimNum();
    if (x_dimnum != y_dimnum) {
#if ENABLE_LOG
        std::cout << "x_dimnum:" << x_dimnum << "y_dimnum: " << y_dimnum << std::endl;
#endif
        return true;
    }

    for (int i = 0; i < x_dimnum; i++) {
        if (x_shape->GetStorageShape().GetDim(i) != y_shape->GetStorageShape().GetDim(i)) {
            return true;
        }
    }

    return need;
}

static inline int mergeShape(const gert::StorageShape *x_shape, const gert::StorageShape *y_shape, int _xShapeNew[4],
                             int _yShapeNew[4], int zShapeNew[4])
{
    int dimnum = 0;
    int xShapeNew[4];
    int yShapeNew[4];

    int x_dimnum = x_shape->GetStorageShape().GetDimNum();
    int y_dimnum = y_shape->GetStorageShape().GetDimNum();

    dimnum = std::max(x_dimnum, y_dimnum);
    int x_offset = dimnum - x_dimnum;
    int y_offset = dimnum - y_dimnum;
    for (int i = 0; i < x_offset; i++) {
        xShapeNew[i] = 1;
    }
    for (int i = x_offset; i < dimnum; i++) {
        xShapeNew[i] = x_shape->GetStorageShape().GetDim(i - x_offset);
    }

    for (int i = 0; i < y_offset; i++) {
        yShapeNew[i] = 1;
    }
    for (int i = y_offset; i < dimnum; i++) {
        yShapeNew[i] = y_shape->GetStorageShape().GetDim(i - y_offset);
    }
    // std::cout << "xShapeNew[0..3]:"<<xShapeNew[0]<<", "<<xShapeNew[1]<<", "<<xShapeNew[2]<<", "<<xShapeNew[3]<<"\n";
    // std::cout << "yShapeNew[0..3]:"<<yShapeNew[0]<<", "<<yShapeNew[1]<<", "<<yShapeNew[2]<<", "<<yShapeNew[3]<<"\n";

    // merge neighbor shape
    int lastState = 0;
    int newDimnum = 0;
    int curState = 0;
    for (int i = 0; i < dimnum; i++) {
        if (xShapeNew[i] == yShapeNew[i]) {
            curState = 0;
            // std::cout<<"xi"<<xShapeNew[i]<<","<<yShapeNew[i]<<"\n";
        } else if (xShapeNew[i] == 1) {
            curState = 1;
        } else if (yShapeNew[i] == 1) {
            curState = 2;
        } else {
            curState = 3;
        }
        // std::cout << "curState: " << curState << std::endl;
        if (i > 0 && curState == lastState) {
            _xShapeNew[newDimnum - 1] *= xShapeNew[i];
            _yShapeNew[newDimnum - 1] *= yShapeNew[i];
            // std::cout << "newDimnum: " << newDimnum << std::endl;
        } else {
            newDimnum++;
            _xShapeNew[newDimnum - 1] = xShapeNew[i];
            _yShapeNew[newDimnum - 1] = yShapeNew[i];

            // std::cout << "newDimnum: " << newDimnum << std::endl;
        }
        lastState = curState;
    }

    // set zShape
    for (int i = 0; i < newDimnum; i++) {
        zShapeNew[i] = std::max(_xShapeNew[i], _yShapeNew[i]);
    }
    for (int i = newDimnum; i < 4; i++) {
        _xShapeNew[i] = 1;
        _yShapeNew[i] = 1;
        zShapeNew[i] = 1;
    }

    return newDimnum;
}

static inline void cut_1_dim(int totalLen, int maxTileLen, uint32_t &tileNum, uint32_t &tileLength,
                             uint32_t &lastTileLength)
{
    tileLength = MIN(maxTileLen, totalLen);
    tileNum = CEIL_DIV(totalLen, tileLength);
    lastTileLength = totalLen - (tileNum - 1) * tileLength;
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    uint32_t formerNum;
    uint32_t formerLength;
    uint32_t formerTileNum;
    uint32_t formerTileLength;
    uint32_t formerLastTileLength;

    uint32_t tailNum;
    uint32_t tailLength;
    uint32_t tailTileNum;
    uint32_t tailTileLength;
    uint32_t tailLastTileLength;
    int dimnum;
    FminTilingData tiling;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int block_num = ascendcPlatform.GetCoreNumAiv();

    int tilingKey = 0;

    const gert::StorageShape *x_shape = context->GetInputShape(0);
    const gert::StorageShape *y_shape = context->GetInputShape(1);

    auto dt = context->GetInputTensor(0)->GetDataType();
    uint32_t dataWidth = 4;
    if ((dt == ge::DT_BF16) || (dt == ge::DT_FLOAT16)) {
        dataWidth = 2;
    } else if ((dt == ge::DT_FLOAT)) {
        dataWidth = 4;
    }

#if ENABLE_LOG
    std::cout << "block_num:" << block_num << std::endl;
    std::cout << "dataWidth:" << dataWidth << std::endl;
#endif

#if ENABLE_LOG
    // auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t localMemSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);
    std::cout << "core num:" << coreNum << std::endl;         // 40
    std::cout << "UB mem size:" << localMemSize << std::endl; // 196352
    std::cout << "log from std-cout\n";
#endif
    print_shape(x_shape);
    print_shape(y_shape);

    bool need_broadcast = needBroadcast(x_shape, y_shape);
    if (need_broadcast) {
        int xShapeNew[4] = {1, 1, 1, 1};
        int yShapeNew[4] = {1, 1, 1, 1};
        int zShapeNew[4] = {1, 1, 1, 1};
        dimnum = mergeShape(x_shape, y_shape, xShapeNew, yShapeNew, zShapeNew);
        tiling.set_dimnum(dimnum);
        tiling.set_xShape(xShapeNew);
        tiling.set_yShape(yShapeNew);
        tiling.set_zShape(zShapeNew);


        int size = zShapeNew[0];
        if (size < block_num) {
            block_num = size;
        }
        int block_size = size / block_num;

        formerNum = (size % block_num) == 0 ? block_num : (size % block_num);
        tailNum = block_num - formerNum;
        tailLength = block_size;
        formerLength = block_size + (tailNum == 0 ? 0 : 1);

        int max_last_dim = 2048;
        if (dimnum == 1) // broadcast -1(last) dim, block cut at first dim, tile cut at -1 dim
        {
            int max_tile_len = 2048;
            {
                formerTileLength = MIN(max_tile_len, formerLength);
                formerTileNum = (formerLength + formerTileLength - 1) / formerTileLength;
                formerLastTileLength = formerLength - (formerTileNum - 1) * formerTileLength;
            }
            {
                tailTileLength = MIN(max_tile_len, tailLength);
                tailTileNum = (tailLength + tailTileLength - 1) / tailTileLength;
                tailLastTileLength = tailLength - (tailTileNum - 1) * tailTileLength;
            }
            tilingKey = 1;

        } else if (zShapeNew[dimnum - 1] > max_last_dim) {
            // broadcast -1, tile cut at -1(when -1 is too big), block cut at 0
            int max_tile_len = 4096;
            
//             block_num = 8;
//                 int block_size = size / block_num;

//                 formerNum = (size % block_num) == 0 ? block_num : (size % block_num);
//                 tailNum = block_num - formerNum;
//                 tailLength = block_size;
//                 formerLength = block_size + (tailNum == 0 ? 0 : 1);

            int tile_dim_len = zShapeNew[dimnum - 1];
            cut_1_dim(tile_dim_len, max_tile_len, formerTileNum, formerTileLength, formerLastTileLength);
            cut_1_dim(tile_dim_len, max_tile_len, tailTileNum, tailTileLength, tailLastTileLength);

            tilingKey = 5;

        } else if (xShapeNew[dimnum - 1] != yShapeNew[dimnum - 1]) {
            int max_tile_len = 2048 / (1 + zShapeNew[dimnum - 1]);
            if (max_tile_len < 1)
                max_tile_len = 1;
            if (dimnum == 2) // broadcast -1(last) dim, block cut and tile cut at first dim
            {
                cut_1_dim(formerLength, max_tile_len, formerTileNum, formerTileLength, formerLastTileLength);
                cut_1_dim(tailLength, max_tile_len, tailTileNum, tailTileLength, tailLastTileLength);

                tilingKey = 2;
            } else { // broadcast -1 dim, block cut at first dim, tile cut at -2 dim

                int tile_dim_len = zShapeNew[dimnum - 2];
                cut_1_dim(tile_dim_len, max_tile_len, formerTileNum, formerTileLength, formerLastTileLength);
                cut_1_dim(tile_dim_len, max_tile_len, tailTileNum, tailTileLength, tailLastTileLength);

                tilingKey = 3;
            }

        } else // broadcast -2 dim, block cut at first dim, tile cut at -2 dim
        {
            int max_tile_len = 2048 / (2 * ROUND_UP(zShapeNew[dimnum - 1] * dataWidth, 32) / dataWidth);
            if (max_tile_len < 1)
                max_tile_len = 1;

            if (dimnum == 2) {
                int last_dim_len = zShapeNew[dimnum - 1];

                cut_1_dim(formerLength, max_tile_len, formerTileNum, formerTileLength, formerLastTileLength);
                cut_1_dim(tailLength, max_tile_len, tailTileNum, tailTileLength, tailLastTileLength);
            } else {
                int tile_dim_len = zShapeNew[dimnum - 2];

                cut_1_dim(tile_dim_len, max_tile_len, formerTileNum, formerTileLength, formerLastTileLength);
                cut_1_dim(tile_dim_len, max_tile_len, tailTileNum, tailTileLength, tailLastTileLength);
            }
            tilingKey = 4;
        }
#if ENABLE_LOG
        std::cout << "dimnum:" << dimnum << std::endl;
        std::cout << "xShape:" << xShapeNew[0] << "," << xShapeNew[1] << "," << xShapeNew[2] << "," << xShapeNew[3]
                  << std::endl;
        std::cout << "yShape:" << yShapeNew[0] << "," << yShapeNew[1] << "," << yShapeNew[2] << "," << yShapeNew[3]
                  << std::endl;
        std::cout << "zShape:" << zShapeNew[0] << "," << zShapeNew[1] << "," << zShapeNew[2] << "," << zShapeNew[3]
                  << std::endl;
#endif
    } else {
        uint32_t size = x_shape->GetStorageShape().GetShapeSize();

        if (size < block_num * 256) {
            block_num = (size + 256 - 1) / 256;
        }

        int block_size = ROUND_UP(size / block_num, 512 / dataWidth);
        formerNum = size / block_size;
        tailNum = size % block_size == 0 ? 0 : 1;
        tailLength = size - formerNum * block_size;
        formerLength = MIN(size, block_size);
        if (formerNum + tailNum < block_num) {
            block_num = formerNum + tailNum;
        }

        const int max_tile_len = 2048;
        {
            formerTileLength = MIN(max_tile_len, formerLength);
            formerTileNum = (formerLength + formerTileLength - 1) / formerTileLength;
            formerLastTileLength = formerLength - (formerTileNum - 1) * formerTileLength;
        }
        {
            tailTileLength = MIN(max_tile_len, tailLength);
            tailTileNum = (tailLength + tailTileLength - 1) / tailTileLength;
            tailLastTileLength = tailLength - (tailTileNum - 1) * tailTileLength;
        }
        tilingKey = 0;
    }

    tiling.set_formerNum(formerNum);
    tiling.set_formerLength(formerLength);
    tiling.set_formerTileNum(formerTileNum);
    tiling.set_formerTileLength(formerTileLength);
    tiling.set_formerLastTileLength(formerLastTileLength);

    tiling.set_tailNum(tailNum);
    tiling.set_tailLength(tailLength);
    tiling.set_tailTileNum(tailTileNum);
    tiling.set_tailTileLength(tailTileLength);
    tiling.set_tailLastTileLength(tailLastTileLength);

    context->SetTilingKey(tilingKey);
#if ENABLE_LOG
    std::cout << "tilingKey: " << tilingKey << std::endl;
    std::cout << "formerNum:" << formerNum << ", formerLength:" << formerLength << std::endl;
    std::cout << "tailNum:" << tailNum << ", tailLength:" << tailLength << std::endl;
    std::cout << "formerTileNum:" << formerTileNum << std::endl;
    std::cout << "formerTileLength:" << formerTileLength << ", formerLastTileLength:" << formerLastTileLength
              << std::endl;
    std::cout << "tailTileNum:" << tailTileNum << std::endl;
    std::cout << "tailTileLength:" << tailTileLength << ", tailLastTileLength:" << tailLastTileLength << std::endl;
#endif

    context->SetBlockDim(block_num); //
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Fmin : public OpDef {
public:
    explicit Fmin(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8, ge::DT_INT64,
                       ge::DT_INT16, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8, ge::DT_INT64,
                       ge::DT_INT16, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8, ge::DT_INT64,
                       ge::DT_INT16, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Fmin);
} // namespace ops
