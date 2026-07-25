#include "kernel_operator.h"
using namespace AscendC;

// #define PRINTF
#define BLOCK_BYTES_SIZE 32
#define ITER_SIZE 2048

#define BUFFER_NUM 2

template <typename T>
class LcmSpec0 {
 public:
  __aicore__ inline LcmSpec0() {}
  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalNum, TPipe *pipeIn) {
#ifdef PRINTF
    printf("in spec0 init\n");
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

    pipe->InitBuffer(TmpBuf, ITER_SIZE * 4);
    pipe->InitBuffer(CBuf, ITER_SIZE * 4);
    pipe->InitBuffer(ABuf, ITER_SIZE * 4);
    pipe->InitBuffer(BBuf, ITER_SIZE * 4);

    pipe->InitBuffer(ResBuf, ITER_SIZE * 4);
    pipe->InitBuffer(MaskBuf, ITER_SIZE);
    pipe->InitBuffer(X1FloatBuf, ITER_SIZE * 4);
  }

  __aicore__ inline void Process() {
    for (uint32_t i = 0; i < elementNum; i += ITER_SIZE) {
      int calNum = ITER_SIZE < (elementNum - i) ? ITER_SIZE : (elementNum - i);
#ifdef PRINTF
// printf("i = %d, calNum = %d\n",i,calNum);
#endif
      CopyIn(i, calNum);
      ComputeVec(i, calNum);
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

  __aicore__ inline void CopyOut(int begin, int calNum) {
    auto YLocal = YQue.DeQue<DTYPE_X1>();
    DataCopyExtParams copyParams{(uint16_t)1, static_cast<uint32_t>(calNum * sizeof(DTYPE_X1)), 0,
                                 0, 0};
    DataCopyPad(yGm[begin], YLocal, copyParams);

    YQue.FreeTensor(YLocal);
  }

  __aicore__ inline void ComputeVec(uint32_t begin, uint32_t calNum) {
    auto yLocal = YQue.AllocTensor<int32_t>();

    auto resLocal = ResBuf.Get<float>();
    auto maskLocal = MaskBuf.Get<uint16_t>();

    auto x1FloatLocal = X1FloatBuf.Get<float>();

    auto aLocal = ABuf.Get<float>();
    auto bLocal = BBuf.Get<float>();
    auto cLocal = CBuf.Get<float>();
    auto tmpLocal = TmpBuf.Get<uint8_t>();

    uint32_t calNum256 = ((calNum + 63) / 64) * 64;

    auto x1Local = X1Que.DeQue<int32_t>();
    auto x2Local = X2Que.DeQue<int32_t>();

#ifdef PRINTF
    printf("begin,num,num256 = %d %d %d\n", begin, calNum, calNum256);
#endif

    Cast(x1FloatLocal, x1Local, RoundMode::CAST_NONE, calNum);
    Cast(cLocal, x2Local, RoundMode::CAST_NONE, calNum);

    Abs(x1FloatLocal, x1FloatLocal, calNum);
    Abs(cLocal, cLocal, calNum);

    Max(aLocal, x1FloatLocal, cLocal, calNum);
    Min(bLocal, x1FloatLocal, cLocal, calNum);

    Cast(x2Local, cLocal, RoundMode::CAST_RINT, calNum);

    LocalTensor<float> a = aLocal, b = bLocal, c = cLocal;
    LocalTensor<float> tmp;
    float zero = 0;

    for (int i = 0; i < 14; i++) {
      // b等于0，mask置为1,表示需要从计算结果中选择
      CompareScalar(maskLocal, b, zero, AscendC::CMPMODE::EQ, calNum256);
      Select(resLocal, maskLocal, a, resLocal, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
             calNum256);

#ifdef PRINTF
// if (begin==0)
// {
// 	printf("i=%d : \n",i);
// 	PrintVec(a,0,8);
// 	PrintVec(b,0,8);
// 	PrintVec(c,0,8);
// 	PrintVec(resLocal,0,8);
// }
#endif

      Fmod(c, a, b, tmpLocal, calNum);

      tmp = a;
      a = b;
      b = c;
      c = tmp;
    }

    // Cast(yLocal, resLocal, RoundMode::CAST_RINT, calNum);

    Div(x1FloatLocal, x1FloatLocal, resLocal, calNum);
    Cast(yLocal, x1FloatLocal, RoundMode::CAST_RINT, calNum);
    Mul(yLocal, yLocal, x2Local, calNum);

    YQue.EnQue(yLocal);

    X1Que.FreeTensor(x1Local);
    X2Que.FreeTensor(x2Local);
  }

 private:
  GlobalTensor<DTYPE_X1> x1Gm, x2Gm, yGm;
  TPipe *pipe;

  uint32_t coreIndex, coreNum;
  uint32_t totalNum, totalBlockNum;
  uint32_t blockSize, blockNum, blockBegin;

  uint32_t elementBegin, elementNum;

  TBuf<QuePosition::VECCALC> TmpBuf, CBuf, ABuf, BBuf;

  TBuf<QuePosition::VECCALC> ResBuf, MaskBuf, X1FloatBuf;
  TQue<QuePosition::VECIN, BUFFER_NUM> X1Que, X2Que;
  TQue<QuePosition::VECOUT, BUFFER_NUM> YQue;
};