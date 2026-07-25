#include "../op_host/common.h"
#include "kernel_common.h"
#include "kernel_operator.h"
using namespace AscendC;

template <typename T>
class CopysignSpec0 {
   public:
    using FP = std::conditional_t<std::is_same_v<T, float>, float, half>;
    __aicore__ inline CopysignSpec0() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalNum,
                                TPipe *pipeIn) {
#ifdef OPEN_LOG
        printf("in spec0 init\n");
#endif
        this->pipe = pipeIn;
        this->totalNum = totalNum;

        this->coreNum = GetBlockNum();
        this->coreIndex = GetBlockIdx();

        this->blockSize = BLOCK_BYTES_SIZE / sizeof(T);
        this->iterSize = ITER_BYTES_SIZE_0 / sizeof(T);
        // leftLoopNum = 6;
        // if (sizeof(T) == 4) {
        //     leftLoopBegin = 0;
        // } else {
        //     leftLoopBegin = 1;
        // }

        totalBlockNum = (totalNum + blockSize - 1) / blockSize;
        int blockDiv = totalBlockNum / coreNum;
        int blockMod = totalBlockNum % coreNum;
        blockNum = blockDiv + (coreIndex < blockMod ? 1 : 0);
        blockBegin = coreIndex * blockDiv + (coreIndex < blockMod ? coreIndex : blockMod);

        elemBegin = blockBegin * blockSize;
        elemNum = blockNum * blockSize;
        elemNum = elemNum < (totalNum - elemBegin) ? elemNum : totalNum - elemBegin;

        shiftBits = sizeof(T) * 8 - 1;

#ifdef OPEN_LOG
        printf(
            "coreNum,coreIndex = %d %d totalNum totalBlockNum = %d %d blockNum blockBegin = %d "
            "%d\n",
            coreNum, coreIndex, totalNum, totalBlockNum, blockNum, blockBegin);
        printf("elemBegin elemNum = %d %d\n", elemBegin, elemNum);
#endif
        x1Gm.SetGlobalBuffer((__gm__ T *)x1 + elemBegin, elemNum);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2 + elemBegin, elemNum);
        yGm.SetGlobalBuffer((__gm__ T *)y + elemBegin, elemNum);

        pipe->InitBuffer(X1Que, BUFFER_NUM, ITER_BYTES_SIZE_0 + 32);
        pipe->InitBuffer(X2Que, BUFFER_NUM, ITER_BYTES_SIZE_0 + 32);
        pipe->InitBuffer(YQue, BUFFER_NUM, ITER_BYTES_SIZE_0 + 32);
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < elemNum; i += iterSize) {
            int calNum = iterSize < (elemNum - i) ? iterSize : (elemNum - i);
#ifdef OPEN_LOG
// printf("i = %d, calNum = %d\n",i,calNum);
#endif
            CopyIn(i, calNum);
            ComputeVec(i, calNum);
            CopyOut(i, calNum);
        }
    }

    __aicore__ inline void CopyIn(uint32_t begin, uint32_t calNum) {
        auto x1Local = X1Que.AllocTensor<T>();
        auto x2Local = X2Que.AllocTensor<T>();
        DataCopyExtParams copyParams{(uint16_t)1, static_cast<uint32_t>(calNum * sizeof(T)), 0, 0,
                                     0};

        DataCopyPad(x1Local, x1Gm[begin], copyParams, padParams);
        DataCopyPad(x2Local, x2Gm[begin], copyParams, padParams);

        X1Que.EnQue(x1Local);
        X2Que.EnQue(x2Local);
    }

    __aicore__ inline void CopyOut(int begin, int calNum) {
        auto YLocal = YQue.DeQue<T>();
        DataCopyExtParams copyParams{(uint16_t)1, static_cast<uint32_t>(calNum * sizeof(T)), 0, 0,
                                     0};
        DataCopyPad(yGm[begin], YLocal, copyParams);

        YQue.FreeTensor(YLocal);
    }

    __aicore__ inline void ComputeVec(uint32_t begin, uint32_t calNum) {
        auto yLocal = YQue.AllocTensor<FP>();

        auto x1Local = X1Que.DeQue<FP>();
        auto x2Local = X2Que.DeQue<FP>();

        auto x2LocalUint = x2Local.template ReinterpretCast<U<FP>>();

        auto x1LocalUint16 = x1Local.template ReinterpretCast<uint16_t>();
        auto x2LocalUint16 = x2Local.template ReinterpretCast<uint16_t>();
        auto yLocalUint16 = yLocal.template ReinterpretCast<uint16_t>();

        ShiftRight(x2LocalUint, x2LocalUint, shiftBits, calNum);
        ShiftLeft(x2LocalUint, x2LocalUint, shiftBits, calNum);

        Abs(x1Local, x1Local, calNum);

#ifdef OPEN_LOG
        printf("begin,num = %d %d\n", begin, calNum);
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
        if constexpr (sizeof(T) == 4) {
            calNum <<= 1;
        }
        Or(yLocalUint16, x1LocalUint16, x2LocalUint16, calNum);

        // Adds(yLocal, x1Local, static_cast<T>(0), calNum);

        YQue.EnQue(yLocal);

        X1Que.FreeTensor(x1Local);
        X2Que.FreeTensor(x2Local);
    }

   private:
    GlobalTensor<T> x1Gm, x2Gm, yGm;
    TPipe *pipe;

    uint32_t coreIndex, coreNum;
    uint32_t totalNum, totalBlockNum;
    uint32_t blockSize, blockNum, blockBegin;
    uint32_t iterSize;

    U<FP> shiftBits;

    uint32_t elemBegin, elemNum;

    // TBuf<QuePosition::VECCALC> ResBuf, MaskBuf, X2ResBuf, X2HalfBuf;
    TQue<QuePosition::VECIN, BUFFER_NUM> X1Que;
    TQue<QuePosition::VECIN, BUFFER_NUM> X2Que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> YQue;

    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
};