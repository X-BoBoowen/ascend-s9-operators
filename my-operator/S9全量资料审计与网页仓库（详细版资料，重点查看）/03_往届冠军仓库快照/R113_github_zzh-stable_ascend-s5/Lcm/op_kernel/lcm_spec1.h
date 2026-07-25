#include "kernel_operator.h"
using namespace AscendC;

// #define PRINTF
#define BLOCK_BYTES_SIZE 32
#define WRITEBACK 1024
#define ITER_SIZE 8192

#define BUFFER_NUM 2

template <typename T>
class LcmSpec1 {
 public:
  __aicore__ inline LcmSpec1() {}
  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalNum, TPipe *pipeIn) {
#ifdef PRINTF
    printf("in spec1 init\n");
#endif
    this->pipe = pipeIn;
    this->totalNum = totalNum;

    this->coreNum = GetBlockNum();
    this->coreIndex = GetBlockIdx();

    this->blockSize = 32 / sizeof(DTYPE_X1);

    totalBlockNum = (totalNum + blockSize - 1) / blockSize;
    int blockDiv = totalBlockNum / coreNum;
    int blockMod = totalBlockNum % coreNum;
    blockNum = blockDiv + (coreIndex < blockMod ? 1 : 0);
    blockBegin = coreIndex * blockDiv + (coreIndex < blockMod ? coreIndex : blockMod);

    elementBegin = blockBegin * blockSize;
    elementNum = blockNum * blockSize;
    elementNum = elementNum < (totalNum - elementBegin) ? elementNum : totalNum - elementBegin;

#ifdef PRINTF
    printf("coreNum,coreIndex = %d %d totalNum totalBlockNum = %d %d blockNum blockBegin = %d %d\n",
           coreNum, coreIndex, totalNum, totalBlockNum, blockNum, blockBegin);
    printf("elementBegin elementNum = %d %d\n", elementBegin, elementNum);
#endif
    x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x1 + elementBegin, elementNum);
    x2Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x2 + elementBegin, elementNum);
    yGm.SetGlobalBuffer((__gm__ DTYPE_X1 *)y + elementBegin, elementNum);

    pipe->InitBuffer(X1Que, BUFFER_NUM, ITER_SIZE * sizeof(DTYPE_X1));
    pipe->InitBuffer(X2Que, BUFFER_NUM, ITER_SIZE * sizeof(DTYPE_X1));
    pipe->InitBuffer(YQue, BUFFER_NUM, ITER_SIZE * sizeof(DTYPE_X1));
  }

  __aicore__ inline void Process() {
    for (uint32_t i = 0; i < elementNum; i += ITER_SIZE) {
      int calNum = ITER_SIZE < (elementNum - i) ? ITER_SIZE : (elementNum - i);
#ifdef PRINTF
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
    DataCopyExtParams copyParams{(uint16_t)1, static_cast<uint32_t>(calNum * sizeof(DTYPE_X1)), 0,
                                 0, 0};
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

      DTYPE_X1 res = Cal(x1, x2);
      yLocal.SetValue(i, res);
    }

    YQue.EnQue(yLocal);

    X1Que.FreeTensor(x1Local);
    X2Que.FreeTensor(x2Local);
  }

  __aicore__ inline void CopyOut(int begin, int calNum) {
    auto YLocal = YQue.DeQue<DTYPE_X1>();
    DataCopyExtParams copyParams{(uint16_t)1, static_cast<uint32_t>(calNum * sizeof(DTYPE_X1)), 0,
                                 0, 0};
    DataCopyPad(yGm[begin], YLocal, copyParams);

    YQue.FreeTensor(YLocal);
  }

  __aicore__ inline DTYPE_X1 Cal(DTYPE_X1 x1, DTYPE_X1 x2) {
    DTYPE_X1 abs1 = x1 < 0 ? -x1 : x1;
    DTYPE_X1 abs2 = x2 < 0 ? -x2 : x2;

    DTYPE_X1 g = gcd(abs1, abs2);
    DTYPE_X1 res = (abs1 / g) * abs2;

    if constexpr (!std::is_same_v<T, std::int8_t>) {
      res = res < 0 ? -res : res;
    }

    // printf("addr x1 x2 abs1 abs2 g abs1/g res: %d, %d %d, %d %d, %d %d
    // %d\n",addr1,x1,x2,abs1,abs2,g,res1,res);

    return res;
  }
  __aicore__ inline DTYPE_X1 gcd(DTYPE_X1 a, DTYPE_X1 b) {
    while (b != 0) {
      DTYPE_X1 tmp = a;
      a = b;
      b = tmp % b;
    }
    return a;
  }

 private:
  GlobalTensor<DTYPE_X1> x1Gm, x2Gm, yGm;
  TPipe *pipe;

  uint32_t coreIndex, coreNum;
  uint32_t totalNum, totalBlockNum;
  uint32_t blockSize, blockNum, blockBegin;

  uint32_t elementBegin, elementNum;

  // TBuf<QuePosition::VECCALC> TmpBuf, CBuf, ABuf, BBuf;
  TQue<QuePosition::VECIN, BUFFER_NUM> X1Que, X2Que;
  TQue<QuePosition::VECOUT, BUFFER_NUM> YQue;
};