#pragma once

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

namespace MatMulSrBase {

constexpr int32_t BUFFER_NUM_STAGE1 = 2;
constexpr int32_t BUFFER_NUM_STAGE3 = 1;
constexpr int32_t FLOAT32_SIZE_BYTE = 4;
constexpr uint64_t SELECT_MASK = 12297829382473034410; //10101010
constexpr uint64_t MAX_UINT64 = 18446744073709551615;
constexpr int32_t DEFAULT_SYNCALL_NEED_SIZE = 8;

class ComplexMatMulTwoStageKernel1 {
public:
    __aicore__ inline ComplexMatMulTwoStageKernel1() {}
    __aicore__ inline void GetCoreIndex(int32_t blockIdx, int32_t bigCoreNum, int32_t bigCoreStride, int32_t& startIndex, int32_t& endIndex){
        if (blockIdx < bigCoreNum) {
            startIndex = blockIdx * bigCoreStride;
            endIndex = startIndex + bigCoreStride;
        } else {
            startIndex = bigCoreNum * bigCoreStride + (blockIdx - bigCoreNum) * (bigCoreStride - 1);
            endIndex = startIndex + bigCoreStride - 1;
        }
    }   
    
    __aicore__ inline void InitStage1(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, int32_t bigCoreNumA, int32_t bigCoreStrideA, int32_t bigCoreNumB, int32_t bigCoreStrideB, int32_t maxCalcNumStage1, int32_t mLength, int32_t nLength, int32_t kLength, int32_t nowBatch) {
        blockIdx = GetBlockIdx();
        // printf("blockIdx: %d \ n", blockIdx);
        //index 左闭右开
        this->nowBatch = nowBatch;
        GetCoreIndex(blockIdx, bigCoreNumA, bigCoreStrideA, startIndexX, endIndexX);
        GetCoreIndex(blockIdx, bigCoreNumB, bigCoreStrideB, startIndexY, endIndexY);
        this->tileLengthX = maxCalcNumStage1 > (endIndexX - startIndexX) ? (endIndexX - startIndexX) : maxCalcNumStage1;
        this->tileLengthY = maxCalcNumStage1 > (endIndexY - startIndexY) ? (endIndexY - startIndexY) : maxCalcNumStage1;
        this->tileLength = this->tileLengthX > this->tileLengthY ? this->tileLengthX : this->tileLengthY;
        // 注意：这里将 workspace 转换为 (__gm__ float*) 指针时，是按 float 类型来访问内存。
        // 因此，RealGm 对应的区域大小为 mLength * kLength 个 float，
        // 而 workspace 作为字节地址，第二个缓冲区的起始地址应偏移 mLength * kLength * FLOAT32_SIZE_BYTE 个字节，
        // 才能正确指向下一个 float 数组块。
        RealXGm.SetGlobalBuffer((__gm__ float*)workspace, mLength * kLength);
        ImagXGm.SetGlobalBuffer((__gm__ float*)(workspace + mLength * kLength * FLOAT32_SIZE_BYTE), mLength * kLength);
        RealYGm.SetGlobalBuffer((__gm__ float*)(workspace + 2 * mLength * kLength * FLOAT32_SIZE_BYTE), kLength * nLength);
        ImagYGm.SetGlobalBuffer((__gm__ float*)(workspace + 2 * mLength * kLength * FLOAT32_SIZE_BYTE + kLength * nLength * FLOAT32_SIZE_BYTE), kLength * nLength);
        xGm.SetGlobalBuffer((__gm__ float*)(x + nowBatch * mLength * kLength * 2 * FLOAT32_SIZE_BYTE), mLength * kLength * 2);
        yGm.SetGlobalBuffer((__gm__ float*)(y + nowBatch * kLength * nLength * 2 * FLOAT32_SIZE_BYTE), kLength * nLength * 2);
    }

    __aicore__ inline void CopyInStage1(const GlobalTensor<float> &x, int32_t offset, int32_t tileLength) {
        LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
        AscendC::DataCopyExtParams copyParams{1, (uint32_t)tileLength * FLOAT32_SIZE_BYTE, 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, x[offset], copyParams, padParams);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void ComputeStage1(int32_t startIndex, int32_t tileLength) {
        LocalTensor<float> xLocal = inQueue.DeQue<float>();
        LocalTensor<float> realLocal = outQueueReal.AllocTensor<float>();
        LocalTensor<float> imagLocal = outQueueImag.AllocTensor<float>();
        auto tmpInt1 = tmpIntBuffer1.Get<int32_t>();
        auto tmpInt2 = tmpIntBuffer2.Get<int32_t>();
        int32_t realOffset = (startIndex % 2) + startIndex / 2;
        int32_t imagOffset = startIndex / 2;
        CreateVecIndex(tmpInt1, (int32_t)0, tileLength);
        PipeBarrier<PIPE_ALL>();
        Muls(tmpInt2, tmpInt1, (2 * FLOAT32_SIZE_BYTE), tileLength);
        PipeBarrier<PIPE_ALL>();
        auto tmpUInt1 = tmpInt2.ReinterpretCast<uint32_t>();
        if (startIndex % 2 == 0) {
            Gather(realLocal, xLocal, tmpUInt1, (uint32_t)0, tileLength / 2 + (tileLength % 2));
        } else {
            Gather(imagLocal, xLocal, tmpUInt1, (uint32_t)0, tileLength / 2 + (tileLength % 2));
        }
        Adds(tmpInt2, tmpInt2, FLOAT32_SIZE_BYTE, tileLength);
        PipeBarrier<PIPE_ALL>();
        auto tmpUInt2 = tmpInt2.ReinterpretCast<uint32_t>();
        if (startIndex % 2 == 1) {
            Gather(realLocal, xLocal, tmpUInt2, (uint32_t)0, tileLength / 2);
        } else {
            Gather(imagLocal, xLocal, tmpUInt2, (uint32_t)0, tileLength / 2);
        }
        // if (blockIdx == 0){
        //     DumpTensor(realLocal, 78, 32);
        //     DumpTensor(imagLocal, 79, 32);
        //     DumpTensor(tmpUInt1, 80, 32);
        //     DumpTensor(tmpUInt2, 81, 32);
        //     DumpTensor(tmpInt1, 82, 32);
        //     DumpTensor(tmpInt2, 83, 32);
        // }
        inQueue.FreeTensor(xLocal);
        outQueueReal.EnQue(realLocal);
        outQueueImag.EnQue(imagLocal);
    }

    __aicore__ inline void CopyOutStage1(GlobalTensor<float> &z, int32_t offset, TQue<QuePosition::VECIN, BUFFER_NUM_STAGE1>& outQueue, int32_t tileLength) {
        LocalTensor<float> local = outQueue.DeQue<float>();
        AscendC::DataCopyExtParams copyParams{1, (uint32_t)tileLength * FLOAT32_SIZE_BYTE, 0, 0, 0};
        DataCopyPad(z[offset], local, copyParams);
        outQueue.FreeTensor(local);
    }

    __aicore__ inline void ProcessMatrixUnifiedStage1(AscendC::TPipe *pipe,
                                                       GlobalTensor<float>& matrixIn,
                                                       GlobalTensor<float>& matrixOutReal,
                                                       GlobalTensor<float>& matrixOutImag,
                                                       int32_t startIndex,
                                                       int32_t endIndex,
                                                       int32_t tileLength) {
        if (blockIdx == 0) printf("test6\n");
        if (blockIdx == 0) printf("now startIndex %d, endIndex %d\n",startIndex, endIndex);
        if (blockIdx == 0) printf("currentTileLength %d\n",tileLength > (endIndex - startIndex) ? (endIndex - startIndex) : tileLength);
        while (startIndex < endIndex) {
            if (blockIdx == 0) printf("now startIndex %d, endIndex %d\n",startIndex, endIndex);
            // 计算本次处理块的长度：若剩余长度不足tileLength，则处理剩余部分
            int32_t currentTileLength = tileLength > (endIndex - startIndex) ? (endIndex - startIndex) : tileLength;
            // 调用统一的拷贝与计算函数处理数据块
            if (blockIdx == 0) printf("test0\n");
            CopyInStage1(matrixIn, startIndex, currentTileLength);
            if (blockIdx == 0) printf("test1\n");
            ComputeStage1(startIndex, currentTileLength);
            if (blockIdx == 0) printf("test2\n");
            if (startIndex % 2 == 0) {
                CopyOutStage1(matrixOutReal, startIndex / 2, outQueueReal, currentTileLength / 2 + (currentTileLength % 2));
                CopyOutStage1(matrixOutImag, startIndex / 2, outQueueImag, currentTileLength / 2);
            } else {
                CopyOutStage1(matrixOutImag, startIndex / 2, outQueueImag, currentTileLength / 2 + (currentTileLength % 2));
                CopyOutStage1(matrixOutReal, startIndex / 2 + 1, outQueueReal, currentTileLength / 2);
            }
            if (blockIdx == 0) printf("test3\n");
            startIndex += currentTileLength;
        }
    }

    __aicore__ inline void ProcessStage1(AscendC::TPipe *pipe) {
        // 初始化相关缓冲区
        pipe->InitBuffer(tmpIntBuffer1, this->tileLength * sizeof(int32_t));
        pipe->InitBuffer(tmpIntBuffer2, this->tileLength * sizeof(int32_t));
        pipe->InitBuffer(inQueue, BUFFER_NUM_STAGE1, this->tileLength * FLOAT32_SIZE_BYTE);
        pipe->InitBuffer(outQueueReal, BUFFER_NUM_STAGE1, this->tileLength * FLOAT32_SIZE_BYTE);
        pipe->InitBuffer(outQueueImag, BUFFER_NUM_STAGE1, this->tileLength * FLOAT32_SIZE_BYTE);
        // 统一处理x矩阵
        ProcessMatrixUnifiedStage1(pipe, xGm, RealXGm, ImagXGm, this->startIndexX, this->endIndexX, this->tileLengthX);
        if (blockIdx == 0) printf("test4\n");
        PipeBarrier<PIPE_ALL>();
        if (blockIdx == 0) printf("test5\n");
        // 统一处理y矩阵
        ProcessMatrixUnifiedStage1(pipe, yGm, RealYGm, ImagYGm, this->startIndexY, this->endIndexY, this->tileLengthY);
        // if (blockIdx == 0){
        //     DumpTensor(RealXGm, 0, 64);
        //     DumpTensor(RealXGm[64], 1, 64);
        //     DumpTensor(ImagXGm, 2, 64);
        //     DumpTensor(ImagXGm[64], 3, 64);
        //     DumpTensor(RealYGm, 0, 64);
        //     DumpTensor(RealYGm[64], 1, 64);
        //     DumpTensor(ImagYGm, 2, 64);
        //     DumpTensor(ImagYGm[64], 3, 64);

        //     DumpTensor(RealXGm[960], 1, 64);
        //     DumpTensor(ImagXGm[960], 3, 64);
        //     DumpTensor(RealYGm[960], 1, 64);
        //     DumpTensor(ImagYGm[960], 3, 64);
        // }
    }

private:
    int32_t startIndexX, endIndexX, startIndexY, endIndexY, tileLength, tileLengthX, tileLengthY, blockIdx;
    int32_t nowBatch;
    GlobalTensor<float> RealXGm, RealYGm, ImagXGm, ImagYGm, xGm, yGm;
    TQue<QuePosition::VECIN, BUFFER_NUM_STAGE1> inQueue, outQueueReal, outQueueImag;
    TBuf<QuePosition::VECCALC> tmpIntBuffer1, tmpIntBuffer2;
};

class ComplexMatMulTwoStageKernel2 {
public:
    __aicore__ inline ComplexMatMulTwoStageKernel2() {}
    
    __aicore__ inline void Init(GM_ADDR workspace,
                                int32_t mLength, int32_t nLength, int32_t kLength,
                                int32_t singleM, int32_t singleN, int32_t blockDimM, int32_t blockDimN){
        this->mLength = mLength;
        this->nLength = nLength;
        this->kLength = kLength;
        this->singleM = singleM;
        this->singleN = singleN;
        this->blockDimM = blockDimM;
        this->blockDimN = blockDimN;
        zGm.SetGlobalBuffer((__gm__ float*)(workspace + mLength * kLength * 2 * FLOAT32_SIZE_BYTE + kLength * nLength * 2 * FLOAT32_SIZE_BYTE), mLength * nLength * 4);
        xGm.SetGlobalBuffer((__gm__ float*)workspace, mLength * kLength * 2);
        yGm.SetGlobalBuffer((__gm__ float*)(workspace + mLength * kLength * 2 * FLOAT32_SIZE_BYTE), kLength * nLength * 2);
        blockIndex = GetBlockIdx();
    };

    __aicore__ inline void InitMatrixInfo(int32_t leftInputOffset, int32_t rightInputOffset, int32_t outputOffset){
        // auto test0 = zGm.GetValue(0);
        // printf("the loop %d, test0 is %f\n", outputOffset, test0);
        // auto test1 = zGm.GetValue(32 * 32);
        // printf("the loop %d, test1 is %f\n", outputOffset, test1);
        this->leftInputOffset = leftInputOffset;
        this->rightInputOffset = rightInputOffset;
        this->outputOffset = outputOffset;
        aGm = xGm[leftInputOffset * mLength * kLength];
        bGm = yGm[rightInputOffset * kLength * nLength];
        cGm = zGm[outputOffset * mLength * nLength];
        // if (blockIndex == 0){
        //     DumpTensor(aGm, 0, 64);
        //     DumpTensor(bGm, 1, 64);
        //     DumpTensor(aGm[960], 0, 64);
        //     DumpTensor(bGm[960], 1, 64);
        // }
    }

    __aicore__ inline void Process(TPipe *pipe){
        // if (blockIndex >= blockDimN * blockDimM) return;
        int32_t mIndex = blockIndex / blockDimN;
        int32_t nIndex = blockIndex % blockDimN;
        if (mIndex >= blockDimM) mIndex = blockDimM - 1;
        int32_t tailN = nIndex * singleN;
        int32_t tailM = mIndex * singleM;
        int32_t currentSingleM = (mIndex == blockDimM - 1) ? (mLength - tailM) : singleM;
        int32_t currentSingleN = (nIndex == blockDimN - 1) ? (nLength - tailN) : singleN;
        // printf("currentSingleM: %d currentSingleN: %d ",currentSingleM, currentSingleN);
        matmulObj.SetSingleShape(currentSingleM, currentSingleN, kLength);

        int32_t leftMatrixOffset = mIndex * singleM * kLength;
        int32_t rightMatrixOffset = nIndex * singleN;
        // DumpTensor(aGm[leftMatrixOffset], 1, 64);
        // DumpTensor(bGm[rightMatrixOffset], 2, 64);
        int32_t outputOffset = mIndex * singleM * nLength + nIndex * singleN;

        matmulObj.SetTensorA(aGm[leftMatrixOffset]);
        matmulObj.SetTensorB(bGm[rightMatrixOffset]);
        matmulObj.IterateAll(cGm[outputOffset]);
        matmulObj.End();
        // DumpTensor(cGm[outputOffset], 0, 64);
        // DumpTensor(cGm[outputOffset + 960], 31, 64);
    }

    int32_t mLength, nLength, kLength, singleM, singleN, blockDimM, blockDimN, selectApiTempSize, blockIndex;
    int32_t leftInputOffset, rightInputOffset, outputOffset;
    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>> matmulObj;
    GlobalTensor<float> xGm, yGm, zGm, aGm, bGm, cGm;
};

class ComplexMatMulTwoStageKernel3 {
public:
    __aicore__ inline ComplexMatMulTwoStageKernel3() {}
    
    __aicore__ inline void Init(GM_ADDR workspace, GM_ADDR bias, GM_ADDR z, int32_t mLength, int32_t nLength, int32_t kLength, int32_t bigCoreNumM, int32_t bigCoreStrideM, bool hasBias, int32_t biasNLength, int32_t biasMLength, TPipe *pipe, int32_t nowBatch){
        int32_t blockIdx = GetBlockIdx();
        this->mLength = mLength;
        this->nLength = nLength;
        this->hasBias = hasBias;
        this->biasNLength = biasNLength;
        this->biasMLength = biasMLength;
        if (blockIdx < bigCoreNumM) {
            this->startIndexM = blockIdx * bigCoreStrideM;
            this->endIndexM = this->startIndexM + bigCoreStrideM;
        } else {
            this->startIndexM = blockIdx * bigCoreStrideM - (blockIdx - bigCoreNumM);
            this->endIndexM = this->startIndexM + bigCoreStrideM - 1;
        }
        int32_t initialOffset = mLength * kLength * 2 * FLOAT32_SIZE_BYTE + kLength * nLength * 2 * FLOAT32_SIZE_BYTE;
        aGm.SetGlobalBuffer((__gm__ float*)(workspace + initialOffset), nLength * mLength);
        auto test0 = aGm.GetValue(0);
        bGm.SetGlobalBuffer((__gm__ float*)(workspace + initialOffset + nLength * mLength * FLOAT32_SIZE_BYTE), nLength * mLength);
        auto test1 = bGm.GetValue(0);
        cGm.SetGlobalBuffer((__gm__ float*)(workspace + initialOffset + 2 * nLength * mLength * FLOAT32_SIZE_BYTE), nLength * mLength);
        dGm.SetGlobalBuffer((__gm__ float*)(workspace + initialOffset + 3 * nLength * mLength * FLOAT32_SIZE_BYTE), nLength * mLength);
        zGm.SetGlobalBuffer((__gm__ float*)(z + nowBatch * mLength * nLength * 2 * FLOAT32_SIZE_BYTE), mLength * nLength * 2);
        pipe->InitBuffer(inQueueA, BUFFER_NUM_STAGE3, nLength * sizeof(float));
        pipe->InitBuffer(inQueueB, BUFFER_NUM_STAGE3, nLength * sizeof(float));
        pipe->InitBuffer(inQueueC, BUFFER_NUM_STAGE3, nLength * sizeof(float));
        pipe->InitBuffer(inQueueD, BUFFER_NUM_STAGE3, nLength * sizeof(float));
        pipe->InitBuffer(outQueueZ, BUFFER_NUM_STAGE3, 2 * nLength * sizeof(float));
        pipe->InitBuffer(tmpIntBuffer1, this->nLength * sizeof(int32_t) * 2);
        pipe->InitBuffer(tmpIntBuffer2, sizeof(uint64_t));
        pipe->InitBuffer(tmpFloatBuffer1, this->nLength * sizeof(float) * 2);
        pipe->InitBuffer(tmpFloatBuffer2, this->nLength * sizeof(float) * 2);
        blockIndex = GetBlockIdx();

        if (hasBias){
            pipe->InitBuffer(inQueueBias, BUFFER_NUM_STAGE3, 2 * biasNLength * sizeof(float));
            biasGm.SetGlobalBuffer((__gm__ float*)(bias + nowBatch * biasNLength * biasMLength * 2 * FLOAT32_SIZE_BYTE), biasNLength * biasMLength * 2);
        }
    }

    __aicore__ inline void CopyIn(int32_t row){
        int32_t offset = row * nLength;
        LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
        LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
        LocalTensor<float> cLocal = inQueueC.AllocTensor<float>();
        LocalTensor<float> dLocal = inQueueD.AllocTensor<float>();
        AscendC::DataCopyExtParams copyParams{1, (uint32_t)nLength * FLOAT32_SIZE_BYTE, 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(aLocal, aGm[offset], copyParams, padParams);
        inQueueA.EnQue(aLocal);
        DataCopyPad(bLocal, bGm[offset], copyParams, padParams);
        inQueueB.EnQue(bLocal);
        DataCopyPad(cLocal, cGm[offset], copyParams, padParams);
        inQueueC.EnQue(cLocal);
        DataCopyPad(dLocal, dGm[offset], copyParams, padParams);
        inQueueD.EnQue(dLocal);
        if (hasBias){
            int32_t offsetBias = (row % biasMLength) * biasNLength * 2;
            LocalTensor<float> biasLocal = inQueueBias.AllocTensor<float>();
            AscendC::DataCopyExtParams copyBiasParams{1, (uint32_t)2 * biasNLength * FLOAT32_SIZE_BYTE, 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
            DataCopyPad(biasLocal, biasGm[offsetBias], copyBiasParams, padParams);
            inQueueBias.EnQue(biasLocal);
        }
    }
    
    __aicore__ inline void Compute(int32_t offset){
        LocalTensor<float> aLocal = inQueueA.DeQue<float>();
        LocalTensor<float> bLocal = inQueueB.DeQue<float>();
        LocalTensor<float> cLocal = inQueueC.DeQue<float>();
        LocalTensor<float> dLocal = inQueueD.DeQue<float>();
        LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        auto tmpIntLocal1 = tmpIntBuffer1.Get<int32_t>();
        auto tmpIntLocal2 = tmpIntBuffer2.Get<uint64_t>();
        tmpIntLocal2.SetValue(0, SELECT_MASK);
        auto tmpFloatLocal1 = tmpFloatBuffer1.Get<float>();
        auto tmpFloatLocal2 = tmpFloatBuffer2.Get<float>();
        Sub(aLocal, aLocal, bLocal, nLength);
        Add(cLocal, cLocal, dLocal, nLength);
        CreateVecIndex(tmpIntLocal1, (int32_t)0, nLength * 2);
        PipeBarrier<PIPE_ALL>();
        ShiftRight(tmpIntLocal1, tmpIntLocal1, (int32_t)1, nLength * 2);
        PipeBarrier<PIPE_ALL>();
        Muls(tmpIntLocal1, tmpIntLocal1, FLOAT32_SIZE_BYTE, nLength * 2);
        PipeBarrier<PIPE_ALL>();
        // if (offset == 0){
        //     DumpTensor(tmpIntLocal1, 97, 64);
        // }
        auto tmpUIntLocal1 = tmpIntLocal1.ReinterpretCast<uint32_t>();
        Gather(tmpFloatLocal1, aLocal, tmpUIntLocal1, (uint32_t)0, nLength * 2);
        Gather(tmpFloatLocal2, cLocal, tmpUIntLocal1, (uint32_t)0, nLength * 2);
        PipeBarrier<PIPE_ALL>();
        if (hasBias && biasNLength == 1){
            LocalTensor<float> biasLocal = inQueueBias.DeQue<float>();
            float zReal = biasLocal.GetValue(0);
            float zImag = biasLocal.GetValue(1);
            Adds(tmpFloatLocal1, tmpFloatLocal1, zReal, nLength * 2);
            Adds(tmpFloatLocal2, tmpFloatLocal2, zImag, nLength * 2);
            inQueueBias.FreeTensor(biasLocal);
            PipeBarrier<PIPE_ALL>();
        }
        AscendC::BinaryRepeatParams repeatParams = { 1, 1, 1, 8, 8, 8 };
        // dstBlkStride, src0BlkStride, src1BlkStride = 1, no gap between blocks in one repeat
        // dstRepStride, src0RepStride, src1RepStride = 8, no gap between repeats
        uint8_t repeatTimes = (uint8_t)((2 * nLength - 1) / 64);
        int32_t offsetLast = repeatTimes * 64;
        // DumpTensor(tmpFloatLocal1, 0, 64);
        // DumpTensor(tmpFloatLocal2, 1, 64);
        // DumpTensor(tmpIntLocal2, 2, 64);
        // printf("offsetLast: %d %d\n ",offsetLast,nLength * 2 - offsetLast);

        if (repeatTimes > 0) Select(zLocal, tmpIntLocal2, tmpFloatLocal2, tmpFloatLocal1, AscendC::SELMODE::VSEL_CMPMASK_SPR, (uint64_t)64, repeatTimes, repeatParams);
        Select(zLocal[offsetLast], tmpIntLocal2, tmpFloatLocal2[offsetLast], tmpFloatLocal1[offsetLast], AscendC::SELMODE::VSEL_CMPMASK_SPR, nLength * 2 - offsetLast);
        PipeBarrier<PIPE_ALL>();
        if (hasBias && nLength == biasNLength && biasNLength != 1){
            LocalTensor<float> biasLocal = inQueueBias.DeQue<float>();
            Add(zLocal, zLocal, biasLocal, nLength * 2);
            inQueueBias.FreeTensor(biasLocal);
        }
        // DumpTensor(zLocal, 3, 64);
        inQueueA.FreeTensor(aLocal);
        inQueueB.FreeTensor(bLocal);
        inQueueC.FreeTensor(cLocal);
        inQueueD.FreeTensor(dLocal);
        outQueueZ.EnQue(zLocal);
    }

    __aicore__ inline void CopyOut(int32_t offset){
        LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopyExtParams copyParams{1, (uint32_t)2 * nLength * FLOAT32_SIZE_BYTE, 0, 0, 0};
        DataCopyPad(zGm[offset * 2], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }

    __aicore__ inline void Process(TPipe *pipe){
        // if (blockIndex > 0) return;
        for (int32_t row = startIndexM; row < endIndexM; row++){
            int32_t offset = row * nLength;
            // DumpTensor(aGm[offset], 11, 32);
            // DumpTensor(bGm[offset], 12, 32);
            // DumpTensor(cGm[offset], 13, 32);
            // DumpTensor(dGm[offset], 14, 32);
            CopyIn(row);
            Compute(offset);
            CopyOut(offset);
        }
    }

    int32_t nLength, mLength, startIndexM, endIndexM, blockIndex, biasNLength, biasMLength;
    bool hasBias;
    GlobalTensor<float> aGm, bGm, cGm, dGm, biasGm, zGm;
    TQue<QuePosition::VECIN, BUFFER_NUM_STAGE3> inQueueA, inQueueB, inQueueC, inQueueD, inQueueBias, outQueueZ;
    TBuf<QuePosition::VECCALC> tmpIntBuffer1, tmpIntBuffer2, tmpFloatBuffer1, tmpFloatBuffer2;
};

}