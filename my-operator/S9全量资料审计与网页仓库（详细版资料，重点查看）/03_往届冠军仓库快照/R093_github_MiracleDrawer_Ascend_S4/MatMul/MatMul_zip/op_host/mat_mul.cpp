#include "mat_mul_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include <iostream>
#include <cmath>

using namespace matmul_tiling;
using namespace std;

constexpr int32_t FLOAT32_SIZE_BYTE = 4;
constexpr int32_t BUFFER_NUM_STAGE1 = 10;
constexpr int32_t BUFFER_NUM_STAGE2 = 1;
constexpr float UB_USAGE_RATIO = 0.8f;
constexpr int32_t MATMUL_MIN_SHAPE = 16;
constexpr int64_t BEST_BASEN = 256;
constexpr int32_t MAX_BASEM = 256;
constexpr int32_t MAX_BASE_BLOCK = 16 * 1024;

namespace optiling {

int32_t Ceil(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

int32_t AlignUp32(int32_t a)
{
    // refer to the above comments.
    return (a + 31) & ~31; // & ~31: set last five bits of (a+31) to be zero
}

int32_t AlignUp16(int32_t a)
{
    // refer to the above comments.
    return (a + 15) & ~15; // & ~15: set last four bits of (a+15) to be zero
}

int32_t AlignUpDiv(int32_t a, int32_t base)
{
    return (a + base - 1) / base * base;
}


void CalcCoreSpilt(int32_t nLength, int32_t mLength, int32_t numCores, int32_t& bigCoreStrideBytes, int32_t& bigCoreNum){
    int32_t totalSize = nLength * mLength * 2;
    // 均匀分配每个核心的数据量（以字节为单位），不进行32字节对齐
    int32_t bytesPerCore = totalSize / numCores;
    int32_t extra = totalSize % numCores;

    // 如果有余下的字节，则部分核心分得的数据比其他核心多1字节
    if (extra > 0) {
        bigCoreStrideBytes = bytesPerCore + 1;  // 大核心分配的字节数
        bigCoreNum = extra;                // 大核心的数量
    } else {
        bigCoreStrideBytes = bytesPerCore;      // 所有核心均分
        bigCoreNum = numCores;
    }
}

void CalcMatmulBaseData(int32_t mLength, int32_t nLength, int32_t maxCalcNumStage2, int32_t& baseM, int32_t& baseN){
    baseN = BEST_BASEN;
    while (nLength < baseN) {
        baseN = baseN >> 1;
    }
    if (baseN < nLength && baseN < BEST_BASEN) {
        baseN = baseN << 1;
    }
    baseN = std::max(baseN, MATMUL_MIN_SHAPE);
    int32_t maxBaseM = maxCalcNumStage2 / baseN;
    baseM = MAX_BASE_BLOCK / baseN;
    while (baseM > maxBaseM || mLength < baseM) {
        baseM = baseM >> 1;
    }
    baseM = std::max(baseM, MATMUL_MIN_SHAPE);
    baseM = std::min(baseM, MAX_BASEM);
}

void CalcCoreSpiltData(int32_t nLength, int32_t mLength, int32_t baseN, int32_t baseM, int32_t coreNum, int32_t& singleN, int32_t& singleM, int32_t& blockDimM, int32_t& blockDimN){
    int32_t maxNLoops = Ceil(nLength, baseN);
    int32_t maxMLoops = Ceil(mLength, baseM);
    int32_t curNLoops = std::min(maxNLoops, coreNum);
    if (curNLoops == 0) curNLoops = 1;
    int32_t curMLoops = std::min(maxMLoops, coreNum / curNLoops);
    int32_t curSingleN = AlignUpDiv(Ceil(nLength, curNLoops), baseN);
    int32_t curSingleM = AlignUp16(Ceil(mLength, curMLoops));
    curSingleM = std::min(std::max(curSingleM, baseM), mLength);
    singleM = curSingleM;
    singleN = curSingleN;
    blockDimM = Ceil(mLength, curSingleM);
    blockDimN = Ceil(nLength, curSingleN);
    return;
    if (curNLoops * curMLoops <= (coreNum >> 1)) {
        return;
    }
    uint32_t minSingleCore = singleM * singleN; // calc loop on the single core
    while (curNLoops > 1) {
        // skip curNLoops in range (maxNLoops/2) + 1 to (maxNLoops - 1)
        curNLoops = std::min(curNLoops - 1, Ceil(nLength, curSingleN + baseN));
        curSingleN = AlignUpDiv(Ceil(nLength, curNLoops), baseN);
        curNLoops = Ceil(nLength, curSingleN);
        if (curNLoops == 0) {
            break;
        }
        curMLoops = std::min(coreNum / curNLoops, maxMLoops);
        if (curNLoops * curMLoops <= (coreNum >> 1)) {
            break;
        }
        curSingleM = AlignUp16(Ceil(mLength, curMLoops));
        curSingleM = std::min(std::max(curSingleM, baseM), mLength);
        curMLoops = Ceil(mLength, curSingleM);
        uint32_t curSingleCore = curSingleN * curSingleM;
        // select the smaller calc loop on the single core, preferred split N
        if (curSingleCore < minSingleCore ||
            (curSingleCore == minSingleCore && curNLoops * curMLoops < blockDimN * blockDimM) ||
            (curSingleCore == minSingleCore && curNLoops * curMLoops == blockDimN * blockDimM &&
             curSingleM + curSingleN < singleM + singleN)) {
            blockDimM = curMLoops;
            blockDimN = curNLoops;
            singleM = curSingleM;
            singleN = curSingleN;
            minSingleCore = curSingleCore;
        }
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    int32_t nowTilingKey = 1;
    context->SetTilingKey(nowTilingKey);
    int32_t aDim = context->GetInputShape(0)->GetOriginShape().GetDimNum();
    int32_t bDim = context->GetInputShape(1)->GetOriginShape().GetDimNum();
    int32_t outBatch = context->GetOutputShape(0)->GetOriginShape().GetDim(0);
    int32_t outMLength = context->GetOutputShape(0)->GetOriginShape().GetDim(1);
    int32_t outNLength = context->GetOutputShape(0)->GetOriginShape().GetDim(2);
    if (aDim <= 1 || aDim>3 || bDim <= 1 || bDim > 3 || (aDim != bDim)){
        return ge::GRAPH_FAILED;
    }
    MatMulTilingData tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto numCores = ascendcPlatform.GetCoreNum();
    int32_t mLength, nLength, kLength, batchSizeX, batchSizeY, batchSizeZ;
    batchSizeX = 1;
    batchSizeY = 1;
    batchSizeZ = 1;
    if (context->GetInputShape(0)->GetOriginShape().GetDimNum() == 2){
        mLength = context->GetInputShape(0)->GetOriginShape().GetDim(0);
        nLength = context->GetInputShape(1)->GetOriginShape().GetDim(1);
        kLength = context->GetInputShape(0)->GetOriginShape().GetDim(1);
    } else{
        batchSizeX = context->GetInputShape(0)->GetOriginShape().GetDim(0);
        batchSizeY = context->GetInputShape(1)->GetOriginShape().GetDim(0);
        mLength = context->GetInputShape(0)->GetOriginShape().GetDim(1);
        nLength = context->GetInputShape(1)->GetOriginShape().GetDim(2);
        kLength = context->GetInputShape(0)->GetOriginShape().GetDim(2);
    }
    if (batchSizeX != batchSizeY){
        return ge::GRAPH_FAILED;
    }
    if (batchSizeX > batchSizeY) batchSizeZ = batchSizeX; else batchSizeZ = batchSizeY;
    tiling.set_mLength(mLength);
    tiling.set_nLength(nLength);
    tiling.set_kLength(kLength);
    tiling.set_batchSize(batchSizeX);
    
    auto biasShape = context->GetOptionalInputShape(2);
    bool hasBias = true;
    int32_t biasBatch = 1;
    int32_t biasNLength = 1;
    int32_t biasMLength = 1;
    if (biasShape == nullptr) hasBias = false; else{
        if (biasShape->GetOriginShape().GetDimNum() == 2){
            biasMLength = biasShape->GetOriginShape().GetDim(0);
            biasNLength = biasShape->GetOriginShape().GetDim(1);
        } else if (biasShape->GetOriginShape().GetDimNum() == 1){
            biasNLength = biasShape->GetOriginShape().GetDim(0);
        } else{
            biasBatch = biasShape->GetOriginShape().GetDim(0);
            biasMLength = biasShape->GetOriginShape().GetDim(1);
            biasNLength = biasShape->GetOriginShape().GetDim(2);
            // return ge::GRAPH_FAILED;
            //if (biasBatch == 1 || biasMLength == 1 || biasNLength == 1)return ge::GRAPH_FAILED; 都大于1
            // if (biasBatch == batchSizeX && biasMLength == mLength && biasNLength == nLength && biasShape->GetOriginShape().GetDimNum() == 3 && aDim ==3 && biasBatch == batchSizeY && biasBatch == outBatch && mLength == outMLength && nLength == outNLength) return ge::GRAPH_FAILED;
        }
    }
    tiling.set_hasBias(hasBias);
    tiling.set_biasBatch(biasBatch);
    tiling.set_biasNLength(biasNLength);
    tiling.set_biasMLength(biasMLength);

    std::cout<< "mLength: " << mLength << endl;
    std::cout<< "nLength: " <<  nLength << endl;
    std::cout<< "kLength: " << kLength << endl;
    std::cout<< "biasBatch: " << biasBatch << endl;
    std::cout<< "biasNLength: " <<  biasNLength << endl;
    std::cout<< "biasMLength: " << biasMLength << endl;

    if (nowTilingKey == 1){
        context->SetBlockDim(1);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        size_t usrSize = 32768; // 设置用户需要使用的workspace大小。
        // 如需要使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace的大小。
        uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
        currentWorkspace[0] = usrSize + sysWorkspaceSize; // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
        return ge::GRAPH_SUCCESS;
    }
    // spilt data for stage1
    uint64_t ubSizeByte;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeByte);
    std::cout<< "ubSizeByte" << ubSizeByte << endl;
    int32_t bigCoreStrideA = 0;
    int32_t bigCoreNumA = 0;
    int32_t bigCoreStrideB = 0;
    int32_t bigCoreNumB = 0;

    CalcCoreSpilt(mLength, kLength, numCores, bigCoreStrideA, bigCoreNumA);
    CalcCoreSpilt(nLength, kLength, numCores, bigCoreStrideB, bigCoreNumB);
    int32_t maxCalcNumStage1 = (ubSizeByte * UB_USAGE_RATIO / FLOAT32_SIZE_BYTE) / 8 * 8;
    maxCalcNumStage1 /= BUFFER_NUM_STAGE1;
    tiling.set_maxCalcNumStage1(maxCalcNumStage1);
    tiling.set_bigCoreNumA(bigCoreNumA);
    tiling.set_bigCoreStrideA(bigCoreStrideA);
    tiling.set_bigCoreNumB(bigCoreNumB);
    tiling.set_bigCoreStrideB(bigCoreStrideB);
    
    std::cout<< "numCores:" << numCores << endl;
    std::cout<< "maxCalcNumStage1: " <<  maxCalcNumStage1 << endl;
    std::cout<< "bigCoreNumA:" << bigCoreNumA << endl;
    std::cout<< "bigCoreStrideA: " <<  bigCoreStrideA << endl;
    std::cout<< "bigCoreNumB:" << bigCoreNumB << endl;
    std::cout<< "bigCoreStrideB: " <<  bigCoreStrideB << endl;

    //now process tiling data of stage 2
    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(numCores);
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetCType(TPosition::VECIN, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBias(false);

    int32_t maxCalcNumStage2 = ((ubSizeByte) * UB_USAGE_RATIO / FLOAT32_SIZE_BYTE) / 8 * 8;
    maxCalcNumStage2 /= BUFFER_NUM_STAGE2;
    int32_t baseN, baseM;
    CalcMatmulBaseData(mLength, nLength, maxCalcNumStage2, baseM, baseN);
    baseN = 16; baseM = 16;
    int32_t singleN, singleM, blockDimM, blockDimN;
    CalcCoreSpiltData(nLength, mLength, baseN, baseM, numCores, singleN, singleM, blockDimM, blockDimN);
    cubeTiling.SetShape(singleM, singleN, kLength);
    cubeTiling.SetOrgShape(mLength, nLength, kLength);
    cubeTiling.SetFixSplit(baseM, baseN, -1);
    // cubeTiling.SetBias(true);
    cubeTiling.SetBufferSpace(-1, -1, -1);
    tiling.set_singleN(singleN);
    tiling.set_singleM(singleM);
    tiling.set_blockDimM(blockDimM);
    tiling.set_blockDimN(blockDimN);
    
    std::cout<< "baseM :" << baseM << endl;
    std::cout<< "baseN :" << baseN << endl;
    std::cout<< "singleN :" << singleN << endl;
    std::cout<< "singleM :" << singleM << endl;
    std::cout<< "blockDimM :" << blockDimM << endl;
    std::cout<< "blockDimN :" << blockDimN << endl;
    if (cubeTiling.GetTiling(tiling.mmTilingData) == -1){ 
        return ge::GRAPH_FAILED;  
    }

    int32_t mPerCore = mLength / numCores;
    int32_t extraM = mLength % numCores;
    if (extraM > 0){
        tiling.set_bigCoreNumM(extraM);
        tiling.set_bigCoreStrideM(mPerCore + 1);
        std::cout<< "bigCoreNumM :" << extraM << endl;
        std::cout<< "bigCoreStrideM :" << mPerCore + 1 << endl;
    } else {
        tiling.set_bigCoreNumM(numCores);
        tiling.set_bigCoreStrideM(mPerCore);
    }

    int32_t aiCubeCoreNum = ascendcPlatform.GetCoreNumAic();
    std::cout<< "aiCubeCoreNum :" << aiCubeCoreNum << endl;
    context->SetBlockDim(aiCubeCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());


    size_t usrSize = 2 * (mLength * kLength * FLOAT32_SIZE_BYTE + kLength * nLength * FLOAT32_SIZE_BYTE) + 4 * mLength * nLength * FLOAT32_SIZE_BYTE + 1024; // 设置用户需要使用的workspace大小。
    // 如需要使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace的大小。
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize; // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
   
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
class MatMul : public OpDef {
public:
    explicit MatMul(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(MatMul);
}
