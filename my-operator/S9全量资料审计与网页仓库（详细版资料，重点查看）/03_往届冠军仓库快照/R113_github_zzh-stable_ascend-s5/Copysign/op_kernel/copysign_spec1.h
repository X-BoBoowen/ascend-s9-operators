#include "../op_host/common.h"
#include "kernel_operator.h"
using namespace AscendC;

#define ITER_SIZE 1024

template <typename T>
class CopysignSpec1 {
   public:
    __aicore__ inline CopysignSpec1() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalNum,
                                TPipe *pipeIn) {
#ifdef OPEN_LOG
        printf("in spec1 init\n");
#endif
        this->pipe = pipeIn;
        this->totalNum = totalNum;

        this->coreNum = GetBlockNum();
        this->coreIndex = GetBlockIdx();

        this->blockSize = BLOCK_BYTES_SIZE / sizeof(DTYPE_X1);

        totalBlockNum = (totalNum + blockSize - 1) / blockSize;
        int blockDiv = totalBlockNum / coreNum;
        int blockMod = totalBlockNum % coreNum;
        blockNum = blockDiv + (coreIndex < blockMod ? 1 : 0);
        blockBegin = coreIndex * blockDiv + (coreIndex < blockMod ? coreIndex : blockMod);

        elementBegin = blockBegin * blockSize;
        elementNum = blockNum * blockSize;
        elementNum = elementNum < (totalNum - elementBegin) ? elementNum : totalNum - elementBegin;

#ifdef OPEN_LOG
        printf(
            "coreNum,coreIndex = %d %d totalNum totalBlockNum = %d %d blockNum blockBegin = %d "
            "%d\n",
            coreNum, coreIndex, totalNum, totalBlockNum, blockNum, blockBegin);
        printf("elementBegin elementNum = %d %d\n", elementBegin, elementNum);
#endif
        x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x1 + elementBegin, elementNum);
        x2Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x2 + elementBegin, elementNum);
        yGm.SetGlobalBuffer((__gm__ DTYPE_X1 *)y + elementBegin, elementNum);

        pipe->InitBuffer(X1Que, BUFFER_NUM, ITER_SIZE * sizeof(DTYPE_X1));
        pipe->InitBuffer(X2Que, BUFFER_NUM, ITER_SIZE * sizeof(DTYPE_X1));
        pipe->InitBuffer(YQue, BUFFER_NUM, ITER_SIZE * sizeof(DTYPE_X1));

        signMask = 1 << (sizeof(DTYPE_X1) * 8 - 1);
        valMask = ~signMask;
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < elementNum; i += ITER_SIZE) {
            int calNum = ITER_SIZE < (elementNum - i) ? ITER_SIZE : (elementNum - i);
#ifdef OPEN_LOG
// printf("i = %d, calNum = %d\n",i,calNum);
#endif
            CopyIn(i, calNum);
            Compute(i, calNum);
            CopyOut(i, calNum);
        }
    }

    __aicore__ inline void CopyIn(uint32_t begin, uint32_t calNum) {
        auto x1Local = X1Que.AllocTensor<DTYPE_X1>();
        auto x2Local = X2Que.AllocTensor<DTYPE_X1>();
        DataCopyExtParams copyParams{(uint16_t)1, static_cast<uint32_t>(calNum * sizeof(DTYPE_X1)),
                                     0, 0, 0};
        DataCopyPadExtParams<DTYPE_X1> padParams{false, 0, 0, 0};

        DataCopyPad(x1Local, x1Gm[begin], copyParams, padParams);
        DataCopyPad(x2Local, x2Gm[begin], copyParams, padParams);

        X1Que.EnQue(x1Local);
        X2Que.EnQue(x2Local);
    }

    __aicore__ inline void Compute(uint32_t begin, uint32_t calNum) {
        auto yLocal = YQue.AllocTensor<DTYPE_X1>();

        auto x1Local = X1Que.DeQue<DTYPE_X1>();
        auto x2Local = X2Que.DeQue<DTYPE_X1>();

        for (int i = 0; i < calNum; i++) {
            DTYPE_X1 x1 = x1Local.GetValue(i);
            DTYPE_X1 x2 = x2Local.GetValue(i);

            U<DTYPE_X1> resInt = (valMask & (*reinterpret_cast<U<DTYPE_X1> *>(&x1))) |
                                 (signMask & (*reinterpret_cast<U<DTYPE_X1> *>(&x2)));
            T res = (*reinterpret_cast<T *>(&resInt));

            yLocal.SetValue(i, res);
        }

        YQue.EnQue(yLocal);

        X1Que.FreeTensor(x1Local);
        X2Que.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(int begin, int calNum) {
        auto YLocal = YQue.DeQue<DTYPE_X1>();
        DataCopyExtParams copyParams{(uint16_t)1, static_cast<uint32_t>(calNum * sizeof(DTYPE_X1)),
                                     0, 0, 0};
        DataCopyPad(yGm[begin], YLocal, copyParams);

        YQue.FreeTensor(YLocal);
    }

   private:
    GlobalTensor<DTYPE_X1> x1Gm, x2Gm, yGm;
    TPipe *pipe;

    uint32_t coreIndex, coreNum;
    uint32_t totalNum, totalBlockNum;
    uint32_t blockSize, blockNum, blockBegin;
    U<DTYPE_X1> signMask, valMask;

    uint32_t elementBegin, elementNum;

    // TBuf<QuePosition::VECCALC> TmpBuf, CBuf, ABuf, BBuf;
    TQue<QuePosition::VECIN, BUFFER_NUM> X1Que, X2Que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> YQue;
};