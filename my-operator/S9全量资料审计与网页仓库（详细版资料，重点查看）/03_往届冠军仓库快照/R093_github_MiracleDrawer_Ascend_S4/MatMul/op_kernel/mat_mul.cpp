#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mat_mul_sr_base.h"
#include "mat_mul_bruteforce.h"
using namespace AscendC;
using namespace matmul;

constexpr int32_t FLOAT32_SIZE_BYTE = 4;

extern "C" __global__ __aicore__ void mat_mul(GM_ADDR x, GM_ADDR y, GM_ADDR bias, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    
    KERNEL_TASK_TYPE_DEFAULT(KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2);

    GET_TILING_DATA(tiling_data, tiling);

    if (TILING_KEY_IS(1)) {
        //brute force
        int32_t blockIdx = GetBlockIdx();
        if (blockIdx == 0){
            MatMulBruteforce::MatMulBruteforceKernel op;
            op.Process(x, y, bias, z, tiling_data.batchSize, tiling_data.mLength, tiling_data.kLength, tiling_data.nLength, tiling_data.hasBias, tiling_data.biasBatch, tiling_data.biasMLength, tiling_data.biasNLength);
        }
    } else if (TILING_KEY_IS(2)) {
        int32_t batchSize = tiling_data.batchSize;
        int32_t mLength = tiling_data.mLength;
        int32_t nLength = tiling_data.nLength;
        int32_t kLength = tiling_data.kLength;
        int32_t biasNLength = tiling_data.biasNLength;
        int32_t biasMLength = tiling_data.biasMLength;
        for (int32_t nowBatch = 0; nowBatch < batchSize; nowBatch++){
            printf("start batch %d\n",nowBatch);
            TBuf<QuePosition::VECCALC> tmpIntBuffer3, tmpIntBuffer4, tmpIntBuffer5;
            MatMulSrBase::ComplexMatMulTwoStageKernel1 opStage1;
            TPipe pipeStage1;
            printf("hi0\n");
            opStage1.InitStage1(x, y, workspace, tiling_data.bigCoreNumA, tiling_data.bigCoreStrideA, tiling_data.bigCoreNumB, tiling_data.bigCoreStrideB, tiling_data.maxCalcNumStage1, tiling_data.mLength, tiling_data.nLength, tiling_data.kLength, nowBatch);
            printf("hi1\n");
            opStage1.ProcessStage1(&pipeStage1);
            printf("hi2\n");
            DataSyncBarrier<MemDsbT::ALL>();
            SyncAll();
            printf("hi3\n");
            pipeStage1.InitBuffer(tmpIntBuffer5, 64);
            auto tmpInt1 = tmpIntBuffer5.Get<int16_t>();
            Duplicate(tmpInt1, (int16_t)0, 32);
            pipeStage1.Destroy();

            MatMulSrBase::ComplexMatMulTwoStageKernel2 opStage2;
            TPipe pipeStage2;
            REGIST_MATMUL_OBJ(&pipeStage2, GetSysWorkSpacePtr(), opStage2.matmulObj, &tiling_data.mmTilingData);
            opStage2.Init(workspace, tiling_data.mLength, tiling_data.nLength, tiling_data.kLength, tiling_data.singleM, tiling_data.singleN, tiling_data.blockDimM, tiling_data.blockDimN);
            //parms: leftInputOffset, rightInputOffset, outputOffset
            //a real * b real
            printf("hi before mm\n");
            opStage2.InitMatrixInfo(0, 0, 0);
            opStage2.Process(&pipeStage2);
            //a imag * b imag
            opStage2.InitMatrixInfo(1, 1, 1);
            opStage2.Process(&pipeStage2);
            //a real * b imag
            opStage2.InitMatrixInfo(0, 1, 2);
            opStage2.Process(&pipeStage2);
            //a imag * b real
            opStage2.InitMatrixInfo(1, 0, 3);
            opStage2.Process(&pipeStage2);

            pipeStage2.Destroy();

            MatMulSrBase::ComplexMatMulTwoStageKernel3 opStage3;
            TPipe pipeStage3;

            SyncAll();
            pipeStage3.InitBuffer(tmpIntBuffer3, 64);
            auto tmpInt3 = tmpIntBuffer3.Get<int16_t>();
            Duplicate(tmpInt3, (int16_t)0, 32);

            // SyncAll<false>();
            opStage3.Init(workspace, bias, z, tiling_data.mLength, tiling_data.nLength, tiling_data.kLength, tiling_data.bigCoreNumM, tiling_data.bigCoreStrideM, tiling_data.hasBias, tiling_data.biasNLength, tiling_data.biasMLength, &pipeStage3, nowBatch);
            opStage3.Process(&pipeStage3);
            SyncAll();
            pipeStage3.InitBuffer(tmpIntBuffer4, 64);
            auto tmpInt4 = tmpIntBuffer4.Get<int16_t>();
            Duplicate(tmpInt4, (int16_t)0, 32);
            pipeStage3.Destroy();
        }
    }
}