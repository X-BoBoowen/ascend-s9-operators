#include "../op_host/common.h"
#include "kernel_operator.h"
using namespace AscendC;

#define ITER_ROW_NUM 8

template <typename T>
class CopysignSpec3 {
   public:
    using FP = std::conditional_t<std::is_same_v<T, float>, float, half>;
    __aicore__ inline CopysignSpec3() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint16_t shapeInf[MAXDIM_2],
                                TPipe *pipeIn) {
#ifdef OPEN_LOG
        printf("in spec3 init\n");
#endif
        this->pipe = pipeIn;

        this->coreNum = GetBlockNum();
        this->coreIndex = GetBlockIdx();
        for (int i = 0; i < MAXDIM_2; ++i) {
            this->shapeInf[i] = shapeInf[i];
        }

        this->blockSize = BLOCK_BYTES_SIZE / sizeof(DTYPE_X1);
        totalBatchNum = shapeInf[1];
        int batchDiv = totalBatchNum / coreNum;
        int batchMod = totalBatchNum % coreNum;
        batchNum = batchDiv + (coreIndex < batchMod ? 1 : 0);
        batchBegin = coreIndex * batchDiv + (coreIndex < batchMod ? coreIndex : batchMod);
        batchEnd = batchBegin + batchNum;

        x2Stride0 = shapeInf[2];
        x1Stride0 = shapeInf[2] * shapeInf[1];

        x1ElemBegin = batchBegin * shapeInf[2];

        x1ElemNum = shapeInf[0] * x1Stride0;
        x2ElemNum = shapeInf[0] * x2Stride0;

        shiftBits = sizeof(T) * 8 - 1;
        shapeInf2Pad = ((shapeInf[2] + blockSize - 1) / blockSize) * blockSize;
        if constexpr (sizeof(T) == 4) {
            calNum = shapeInf2Pad * 2;
        } else {
            calNum = shapeInf2Pad;
        }

#ifdef OPEN_LOG
        printf("coreNum,coreIndex = %d %d ,batchNum,batchBegin = %d %d , shapeInf = %d %d %d calNum = %d shapeInf2Pad = %d\n",
               coreNum, coreIndex, batchNum, batchBegin, shapeInf[0], shapeInf[1], shapeInf[2], calNum, shapeInf2Pad);
#endif
        x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x1, x1ElemNum);
        x2Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x2, x2ElemNum);
        yGm.SetGlobalBuffer((__gm__ DTYPE_X1 *)y, x1ElemNum);

        pipe->InitBuffer(X1Que, BUFFER_NUM,
                         ITER_ROW_NUM * shapeInf2Pad * sizeof(DTYPE_X1) + BLOCK_BYTES_SIZE);
        pipe->InitBuffer(X2Que, BUFFER_NUM, shapeInf2Pad * sizeof(DTYPE_X1) + BLOCK_BYTES_SIZE);
        pipe->InitBuffer(YQue, BUFFER_NUM,
                         ITER_ROW_NUM * shapeInf2Pad * sizeof(DTYPE_X1) + BLOCK_BYTES_SIZE);

        signMask = 1 << (sizeof(DTYPE_X1) * 8 - 1);
        valMask = ~signMask;
    }

    __aicore__ inline void Process() {
        for (int i = 0; i < shapeInf[0]; i++) {
            CopyInX2(i);
            auto x2Local = X2Que.DeQue<DTYPE_X1>();
            ComputeX2(x2Local);
            for (int j = batchBegin; j < batchEnd; j += ITER_ROW_NUM) {
                int calRowNum = ITER_ROW_NUM < batchEnd - j ? ITER_ROW_NUM : (batchEnd - j);
                CopyInX1(i, j, calRowNum);
                ComputeX1(x2Local, i, calRowNum);
                CopyOut(i, j, calRowNum);
            }
            X2Que.FreeTensor(x2Local);
        }
    }

    __aicore__ inline void CopyInX2(uint32_t i) {
        auto x2Local = X2Que.AllocTensor<DTYPE_X1>();
        DataCopyExtParams copyParams{
            (uint16_t)1, static_cast<uint32_t>(shapeInf[2] * sizeof(DTYPE_X1)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_X1> padParams{false, 0, 0, 0};

        DataCopyPad(x2Local, x2Gm[i * shapeInf[2]], copyParams, padParams);

        X2Que.EnQue(x2Local);
    }

    __aicore__ inline void CopyInX1(uint32_t i, uint32_t j, uint32_t rowNum) {
        auto x1Local = X1Que.AllocTensor<DTYPE_X1>();
        DataCopyExtParams copyParams{
            (uint16_t)rowNum, static_cast<uint32_t>(shapeInf[2] * sizeof(DTYPE_X1)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_X1> padParams{false, 0, 0, 0};

        DataCopyPad(x1Local, x1Gm[i * x1Stride0 + j * shapeInf[2]], copyParams, padParams);

        X1Que.EnQue(x1Local);
    }

    __aicore__ inline void ComputeX2(LocalTensor<DTYPE_X1> x2Local) {
        auto x2LocalUint = x2Local.template ReinterpretCast<U<FP>>();
        ShiftRight(x2LocalUint, x2LocalUint, shiftBits, shapeInf2Pad);
        ShiftLeft(x2LocalUint, x2LocalUint, shiftBits, shapeInf2Pad);
    }

    __aicore__ inline void ComputeX1(LocalTensor<DTYPE_X1> x2Local, uint32_t dim0,
                                     uint32_t rowNum) {
        auto yLocal = YQue.AllocTensor<FP>();
        auto x1Local = X1Que.DeQue<FP>();
#ifdef OPEN_LOG
        printf("dim0= %d rowNum = %d\n", dim0, rowNum);
#endif
        auto x1LocalUint16 = x1Local.template ReinterpretCast<uint16_t>();
        auto x2LocalUint16 = x2Local.template ReinterpretCast<uint16_t>();
        auto yLocalUint16 = yLocal.template ReinterpretCast<uint16_t>();

        Abs(x1Local, x1Local, rowNum * shapeInf2Pad);

#ifdef OPEN_LOG
        // printf("begin,num = %d %d\n", begin, calNum);
        // if (begin == 0) {
        //     int tmp = calNum < 10 ? calNum : 10;
        //     printf("x1:\n");
        //     PrintVecD(x1Local, 0, tmp);
        //     printf("x2:\n");
        //     PrintVecD(x2Local, 0, tmp);
        //     printf("x2HalfLocal:\n");
        //     PrintVec(x2HalfLocal, 0, tmp);
        // }
#endif

        for (int i = 0; i < rowNum; i++) {
#ifdef OPEN_LOG
            printf("i=%d\n", i);
#endif
            Or(yLocalUint16[i * calNum], x1LocalUint16[i * calNum], x2LocalUint16, calNum);
        }

        YQue.EnQue(yLocal);
        X1Que.FreeTensor(x1Local);
    }

    __aicore__ inline void CopyOut(uint32_t i, uint32_t j, uint32_t rowNum) {
        auto yLocal = YQue.DeQue<DTYPE_X1>();
        DataCopyExtParams copyParams{
            (uint16_t)rowNum, static_cast<uint32_t>(shapeInf[2] * sizeof(DTYPE_X1)), 0, 0, 0};
        DataCopyPad(yGm[i * x1Stride0 + j * shapeInf[2]], yLocal, copyParams);

        YQue.FreeTensor(yLocal);
    }

   private:
    GlobalTensor<DTYPE_X1> x1Gm, x2Gm, yGm;
    TPipe *pipe;

    uint32_t coreIndex, coreNum;
    uint32_t totalNum, totalBlockNum;
    uint32_t blockSize;
    uint32_t totalBatchNum, batchNum, batchBegin, batchEnd;
    uint32_t batch2Size;
    uint32_t elementBegin, elementNum;
    uint16_t shapeInf[MAXDIM_2];
    uint32_t x1ElemBegin, x2ElemNum, x1ElemNum;
    uint32_t x1Stride0, x2Stride0;
    uint32_t shapeInf2Pad;

    U<DTYPE_X1> signMask, valMask;
    U<FP> shiftBits;
    int calNum;

    // TBuf<QuePosition::VECCALC> TmpBuf, CntBuf, ABuf, BBuf;
    TQue<QuePosition::VECIN, BUFFER_NUM> X1Que;
    TQue<QuePosition::VECIN, BUFFER_NUM> X2Que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> YQue;
};