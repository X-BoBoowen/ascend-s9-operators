#include "../op_host/common.h"
#include "kernel_operator.h"
using namespace AscendC;

template <typename T>
class CopysignSpec2 {
   public:
    __aicore__ inline CopysignSpec2() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint16_t shapeInf[MAXDIM_2],
                                TPipe *pipeIn) {
#ifdef OPEN_LOG
        printf("in spec2 init\n");
#endif
        this->pipe = pipeIn;

        this->coreNum = GetBlockNum();
        this->coreIndex = GetBlockIdx();
        for (int i = 0; i < MAXDIM_2; ++i) {
            this->shapeInf[i] = shapeInf[i];
        }

        this->blockSize = BLOCK_BYTES_SIZE / sizeof(DTYPE_X1);

        totalBatchNum = shapeInf[0];
        int batchDiv = totalBatchNum / coreNum;
        int batchMod = totalBatchNum % coreNum;
        batchNum = batchDiv + (coreIndex < batchMod ? 1 : 0);
        batchBegin = coreIndex * batchDiv + (coreIndex < batchMod ? coreIndex : batchMod);

        x2BatchSize = shapeInf[2];
        x1BatchSize = shapeInf[1] * x2BatchSize;

        batch2Size = ((shapeInf[2] + blockSize - 1) / blockSize) * blockSize;

        x2ElemBegin = batchBegin * shapeInf[2];
        x1ElemBegin = x2ElemBegin * shapeInf[1];
        x2ElemNum = shapeInf[0] * shapeInf[2];
        x1ElemNum = shapeInf[1] * x2ElemNum;

#ifdef OPEN_LOG
        printf(
            "coreNum,coreIndex = %d %d ,batchNum,batchBegin = %d %d , elemBegin x2 x1 = %d %d "
            "shapeInf = %d %d %d\n",
            coreNum, coreIndex, batchNum, batchBegin, x2ElemBegin, x1ElemBegin, shapeInf[0],
            shapeInf[1], shapeInf[2]);
#endif
        x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x1 + x1ElemBegin, x1ElemNum);
        x2Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x2 + x2ElemBegin, x2ElemNum);
        yGm.SetGlobalBuffer((__gm__ DTYPE_X1 *)y + x1ElemBegin, x1ElemNum);

        pipe->InitBuffer(X1Que, BUFFER_NUM, batch2Size * sizeof(DTYPE_X1));
        pipe->InitBuffer(X2Que, BUFFER_NUM, batch2Size * sizeof(DTYPE_X1));
        pipe->InitBuffer(YQue, BUFFER_NUM, batch2Size * sizeof(DTYPE_X1));

        signMask = 1 << (sizeof(DTYPE_X1) * 8 - 1);
        valMask = ~signMask;
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < batchNum; i += 1) {
            CopyInX2(i);
            CopyInX1(i, 0);
            auto x2Local = X2Que.DeQue<DTYPE_X1>();
            Compute(x2Local);
            CopyOut(i, 0);
            for (uint32_t j = 1; j < shapeInf[1]; j++) {
#ifdef OPEN_LOG
                // printf("i = %d, calNum = %d\n",i,calNum);
#endif
                CopyInX1(i, j);
                Compute(x2Local);
                CopyOut(i, j);
            }

            X2Que.FreeTensor(x2Local);
        }
    }

    __aicore__ inline void CopyInX2(uint32_t i) {
        auto x2Local = X2Que.AllocTensor<DTYPE_X1>();
        DataCopyExtParams copyParams{
            (uint16_t)1, static_cast<uint32_t>(x2BatchSize * sizeof(DTYPE_X1)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_X1> padParams{false, 0, 0, 0};

        DataCopyPad(x2Local, x2Gm[i * x2BatchSize], copyParams, padParams);

        X2Que.EnQue(x2Local);
    }

    __aicore__ inline void CopyInX1(uint32_t i, uint32_t j) {
        auto x1Local = X1Que.AllocTensor<DTYPE_X1>();
        DataCopyExtParams copyParams{
            (uint16_t)1, static_cast<uint32_t>(x2BatchSize * sizeof(DTYPE_X1)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_X1> padParams{false, 0, 0, 0};

        DataCopyPad(x1Local, x1Gm[i * x1BatchSize + j * x2BatchSize], copyParams, padParams);

        X1Que.EnQue(x1Local);
    }

    __aicore__ inline void Compute(LocalTensor<DTYPE_X1> x2Local) {
        auto yLocal = YQue.AllocTensor<DTYPE_X1>();

        auto x1Local = X1Que.DeQue<DTYPE_X1>();

        for (int i = 0; i < x2BatchSize; i++) {
            DTYPE_X1 x1 = x1Local.GetValue(i);
            DTYPE_X1 x2 = x2Local.GetValue(i);

            U<DTYPE_X1> resInt = (valMask & (*reinterpret_cast<U<DTYPE_X1> *>(&x1))) |
                                 (signMask & (*reinterpret_cast<U<DTYPE_X1> *>(&x2)));
            T res = (*reinterpret_cast<T *>(&resInt));
            yLocal.SetValue(i, res);
        }

        YQue.EnQue(yLocal);

        X1Que.FreeTensor(x1Local);
    }

    __aicore__ inline void CopyOut(int i, int j) {
        auto YLocal = YQue.DeQue<DTYPE_X1>();
        DataCopyExtParams copyParams{
            (uint16_t)1, static_cast<uint32_t>(x2BatchSize * sizeof(DTYPE_X1)), 0, 0, 0};
        DataCopyPad(yGm[i * x1BatchSize + j * x2BatchSize], YLocal, copyParams);

        YQue.FreeTensor(YLocal);
    }

   private:
    GlobalTensor<DTYPE_X1> x1Gm, x2Gm, yGm;
    TPipe *pipe;

    uint32_t coreIndex, coreNum;
    uint32_t totalNum, totalBlockNum;
    uint32_t blockSize;
    uint32_t totalBatchNum, batchNum, batchBegin;
    uint32_t batch2Size;
    uint32_t elementBegin, elementNum;
    uint16_t shapeInf[MAXDIM_2];
    uint32_t x2ElemBegin, x1ElemBegin, x2ElemNum, x1ElemNum;
    uint32_t x2BatchSize, x1BatchSize;

    U<DTYPE_X1> signMask, valMask;

    // TBuf<QuePosition::VECCALC> TmpBuf, CntBuf, ABuf, BBuf;
    TQue<QuePosition::VECIN, BUFFER_NUM> X1Que, X2Que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> YQue;
};