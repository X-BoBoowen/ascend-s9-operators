#include "kernel_operator.h"
using namespace AscendC;

// #define PRINTF
#define BLOCK_BYTES_SIZE 32
#define ITER_SIZE 4
#define MAXDIM 3

#define BUFFER_NUM_SPEC3 1

template <typename T>
class LcmSpec3 {
 public:
  __aicore__ inline LcmSpec3() {}
  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint16_t shapeInf[MAXDIM],
                              TPipe *pipeIn) {
#ifdef PRINTF
    printf("in spec3 init\n");
#endif
    this->pipe = pipeIn;

    this->coreNum = GetBlockNum();
    this->coreIndex = GetBlockIdx();
    for (int i = 0; i < MAXDIM; ++i) {
      this->shapeInf[i] = shapeInf[i];
    }

    this->blockSize = 32 / sizeof(DTYPE_X1);
    vecIterSize = 256 / sizeof(DTYPE_X1);

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

#ifdef PRINTF
    printf("coreNum,coreIndex = %d %d ,batchNum,batchBegin = %d %d , shapeInf = %d %d %d\n",
           coreNum, coreIndex, batchNum, batchBegin, shapeInf[0], shapeInf[1], shapeInf[2]);
#endif
    x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x1, x1ElemNum);
    x2Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x2, x2ElemNum);
    yGm.SetGlobalBuffer((__gm__ DTYPE_X1 *)y, x1ElemNum);

    int x1BufSize = ITER_SIZE * shapeInf[2] * sizeof(DTYPE_X1);
    int x2BufSize = shapeInf[2] * sizeof(DTYPE_X1);
    int fp32BufSize = ITER_SIZE * shapeInf[2];

    pipe->InitBuffer(X1Que, BUFFER_NUM_SPEC3, x1BufSize + 32);
    pipe->InitBuffer(X2Que, BUFFER_NUM_SPEC3, shapeInf[2] * sizeof(DTYPE_X1) + 32);
    pipe->InitBuffer(YQue, BUFFER_NUM_SPEC3, x1BufSize + 32);

    pipe->InitBuffer(TmpBuf, fp32BufSize * 4 + 256);
    pipe->InitBuffer(CBuf, fp32BufSize * 4 + 256);
    pipe->InitBuffer(ABuf, fp32BufSize * 4 + 256);
    pipe->InitBuffer(BBuf, fp32BufSize * 4 + 256);

    pipe->InitBuffer(ResBuf, fp32BufSize * 4 + 256);
    pipe->InitBuffer(MaskBuf, fp32BufSize / 8 + 512);

    // maxVal = 1 << 30;
  }

  __aicore__ inline void Process() {
    for (int i = 0; i < shapeInf[0]; i++) {
      CopyInX2(i);
      auto x2Local = X2Que.DeQue<DTYPE_X1>();
      for (int j = batchBegin; j < batchEnd; j += ITER_SIZE) {
        int calRowNum = ITER_SIZE < batchEnd - j ? ITER_SIZE : (batchEnd - j);
        CopyInX1(i, j, calRowNum);
        auto x1Local = X1Que.DeQue<DTYPE_X1>();

        ComputeSca1(x2Local, x1Local, i, calRowNum);
        // PipeBarrier<PIPE_ALL>();
        ComputeVec(i, calRowNum);
        // PipeBarrier<PIPE_ALL>();
        ComputeSca2(x2Local, x1Local, i, calRowNum);

        X1Que.FreeTensor(x1Local);
        CopyOut(i, j, calRowNum);
      }
      X2Que.FreeTensor(x2Local);
    }
  }

  __aicore__ inline void CopyInX2(uint32_t i) {
    auto x2Local = X2Que.AllocTensor<DTYPE_X1>();
    DataCopyExtParams copyParams{(uint16_t)1, static_cast<uint32_t>(shapeInf[2] * sizeof(DTYPE_X1)),
                                 0, 0, 0};
    DataCopyPadExtParams<DTYPE_X1> padParams{false, 0, 0, 0};

    DataCopyPad(x2Local, x2Gm[i * shapeInf[2]], copyParams, padParams);

    X2Que.EnQue(x2Local);
  }

  __aicore__ inline void CopyInX1(uint32_t i, uint32_t j, uint32_t rowNum) {
    auto x1Local = X1Que.AllocTensor<DTYPE_X1>();
    DataCopyExtParams copyParams{
        (uint16_t)1, static_cast<uint32_t>(rowNum * shapeInf[2] * sizeof(DTYPE_X1)), 0, 0, 0};
    DataCopyPadExtParams<DTYPE_X1> padParams{false, 0, 0, 0};

    DataCopyPad(x1Local, x1Gm[i * x1Stride0 + j * shapeInf[2]], copyParams, padParams);

    X1Que.EnQue(x1Local);
  }

  // 通过标量stein将公约数范围降至float范围
  __aicore__ inline void ComputeSca1(LocalTensor<DTYPE_X1> x2Local, LocalTensor<DTYPE_X1> x1Local,
                                     uint32_t dim0, uint32_t rowNum) {
    auto aLocal = ABuf.Get<float>();
    auto bLocal = BBuf.Get<float>();

    int j = 0;
    DTYPE_X1 x1[4], x2[4], a[4], b[4];

    for (; j < shapeInf[2] - 3; j += 4) {
      x2[0] = x2Local.GetValue(j);
      x2[1] = x2Local.GetValue(j + 1);
      x2[2] = x2Local.GetValue(j + 2);
      x2[3] = x2Local.GetValue(j + 3);

      x2[0] = x2[0] < 0 ? -x2[0] : x2[0];
      x2[1] = x2[1] < 0 ? -x2[1] : x2[1];
      x2[2] = x2[2] < 0 ? -x2[2] : x2[2];
      x2[3] = x2[3] < 0 ? -x2[3] : x2[3];

      for (int i = 0; i < rowNum; i++) {
        int rowSt = i * shapeInf[2] + j;
        x1[0] = x1Local.GetValue(rowSt);
        x1[1] = x1Local.GetValue(rowSt + 1);
        x1[2] = x1Local.GetValue(rowSt + 2);
        x1[3] = x1Local.GetValue(rowSt + 3);

        x1[0] = x1[0] < 0 ? -x1[0] : x1[0];
        x1[1] = x1[1] < 0 ? -x1[1] : x1[1];
        x1[2] = x1[2] < 0 ? -x1[2] : x1[2];
        x1[3] = x1[3] < 0 ? -x1[3] : x1[3];
        // gcd stein
        a[0] = x1[0];
        a[1] = x1[1];
        a[2] = x1[2];
        a[3] = x1[3];

        b[0] = x2[0];
        b[1] = x2[1];
        b[2] = x2[2];
        b[3] = x2[3];

        calPart(a[0], b[0]);
        calPart(a[1], b[1]);
        calPart(a[2], b[2]);
        calPart(a[3], b[3]);

        aLocal.SetValue(rowSt, static_cast<float>(a[0]));
        aLocal.SetValue(rowSt + 1, static_cast<float>(a[1]));
        aLocal.SetValue(rowSt + 2, static_cast<float>(a[2]));
        aLocal.SetValue(rowSt + 3, static_cast<float>(a[3]));

        bLocal.SetValue(rowSt, static_cast<float>(b[0]));
        bLocal.SetValue(rowSt + 1, static_cast<float>(b[1]));
        bLocal.SetValue(rowSt + 2, static_cast<float>(b[2]));
        bLocal.SetValue(rowSt + 3, static_cast<float>(b[3]));
      }
    }

    for (; j < shapeInf[2]; j++) {
      x2[0] = x2Local.GetValue(j);

      x2[0] = x2[0] < 0 ? -x2[0] : x2[0];

      for (int i = 0; i < rowNum; i++) {
        int rowSt = i * shapeInf[2] + j;
        x1[0] = x1Local.GetValue(rowSt);

        x1[0] = x1[0] < 0 ? -x1[0] : x1[0];
        // gcd stein
        a[0] = x1[0];

        b[0] = x2[0];

        calPart(a[0], b[0]);

        // x1Local.SetValue(rowSt, x1[0]);
        aLocal.SetValue(rowSt, static_cast<float>(a[0]));
        bLocal.SetValue(rowSt, static_cast<float>(b[0]));
      }
    }
  }

  __aicore__ inline void calPart(DTYPE_X1 &a, DTYPE_X1 &b) {
    // DTYPE_X1 shift = ScalarGetSFFValue<1>(a | b);

    // x1 = x1 >> shift;

    b >>= ScalarGetSFFValue<1>(b);
    // unsigned int band = 1 << 20;
    while (1) {
      a >>= ScalarGetSFFValue<1>(a);
      if (b > a) {
        a ^= b ^= a ^= b;
      }
      if (b == 0) {
        break;
      }
      if (a < valBand) {
        break;
      }
      a -= b;
    };
  }

  __aicore__ inline void ComputeVec(uint32_t dim0, uint32_t rowNum) {
    auto aLocal = ABuf.Get<float>();
    auto bLocal = BBuf.Get<float>();
    auto cLocal = CBuf.Get<float>();
    auto tmpLocal = TmpBuf.Get<uint8_t>();

    auto resLocal = ResBuf.Get<float>();
    auto maskLocal = MaskBuf.Get<uint16_t>();

    const int calNum = rowNum * shapeInf[2];
    const int calNum256 = ((calNum + vecIterSize - 1) / vecIterSize) * vecIterSize;

    LocalTensor<float> a = aLocal, b = bLocal, c = cLocal;
    LocalTensor<float> tmp;

    for (int i = 0; i < 32; i++) {
      CompareScalar(maskLocal, b, static_cast<float>(0), AscendC::CMPMODE::EQ, row256);
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
  }

  __aicore__ inline void ComputeSca2(LocalTensor<DTYPE_X1> x2Local, LocalTensor<DTYPE_X1> x1Local,
                                     uint32_t dim0, uint32_t rowNum) {
    auto resLocal = ResBuf.Get<float>();
    auto yLocal = YQue.AllocTensor<DTYPE_X1>();

    int j = 0;
    DTYPE_X1 x1[4], x2[4], gcdVal[4], res[4];
    for (; j < shapeInf[2] - 3; j += 4) {
      x2[0] = x2Local.GetValue(j);
      x2[1] = x2Local.GetValue(j + 1);
      x2[2] = x2Local.GetValue(j + 2);
      x2[3] = x2Local.GetValue(j + 3);

      x2[0] = x2[0] < 0 ? -x2[0] : x2[0];
      x2[1] = x2[1] < 0 ? -x2[1] : x2[1];
      x2[2] = x2[2] < 0 ? -x2[2] : x2[2];
      x2[3] = x2[3] < 0 ? -x2[3] : x2[3];

      for (int i = 0; i < rowNum; i++) {
        int rowSt = i * shapeInf[2] + j;
        x1[0] = x1Local.GetValue(rowSt);
        x1[1] = x1Local.GetValue(rowSt + 1);
        x1[2] = x1Local.GetValue(rowSt + 2);
        x1[3] = x1Local.GetValue(rowSt + 3);

        gcdVal[0] = resLocal.GetValue(rowSt);
        gcdVal[1] = resLocal.GetValue(rowSt + 1);
        gcdVal[2] = resLocal.GetValue(rowSt + 2);
        gcdVal[3] = resLocal.GetValue(rowSt + 3);

        res[0] = calPart2(x1[0], x2[0], gcdVal[0]);
        res[1] = calPart2(x1[1], x2[1], gcdVal[1]);
        res[2] = calPart2(x1[2], x2[2], gcdVal[2]);
        res[3] = calPart2(x1[3], x2[3], gcdVal[3]);

        yLocal.SetValue(rowSt, res[0]);
        yLocal.SetValue(rowSt + 1, res[1]);
        yLocal.SetValue(rowSt + 2, res[2]);
        yLocal.SetValue(rowSt + 3, res[3]);
      }
    }

    for (; j < shapeInf[2]; j++) {
      x2[0] = x2Local.GetValue(j);
      x2[0] = x2[0] < 0 ? -x2[0] : x2[0];
      for (int i = 0; i < rowNum; i++) {
        int rowSt = i * shapeInf[2] + j;
        x1[0] = x1Local.GetValue(rowSt);
        gcdVal[0] = resLocal.GetValue(rowSt);

        res[0] = calPart2(x1[0], x2[0], gcdVal[0]);

        yLocal.SetValue(rowSt, res[0]);
      }
    }
    YQue.EnQue(yLocal);
  }

  __aicore__ inline DTYPE_X1 calPart2(DTYPE_X1 x1, DTYPE_X1 x2, float gcdVal) {
    DTYPE_X1 shift = ScalarGetSFFValue<1>(x1 | x2);
    x1 = x1 >> shift;

    DTYPE_X1 gcd = static_cast<DTYPE_X1>(gcdVal);
    DTYPE_X1 res = (x1 / gcd) * x2;
    res = res < 0 ? -res : res;
    return res;
  }

  __aicore__ inline void CopyOut(uint32_t i, uint32_t j, uint32_t rowNum) {
    auto YLocal = YQue.DeQue<DTYPE_X1>();
    DataCopyExtParams copyParams{
        (uint16_t)1, static_cast<uint32_t>(rowNum * shapeInf[2] * sizeof(DTYPE_X1)), 0, 0, 0};
    DataCopyPad(yGm[i * x1Stride0 + j * shapeInf[2]], YLocal, copyParams);

    YQue.FreeTensor(YLocal);
  }

 private:
  GlobalTensor<DTYPE_X1> x1Gm, x2Gm, yGm;
  TPipe *pipe;

  uint32_t coreIndex, coreNum;
  uint32_t totalNum, totalBlockNum;
  uint32_t blockSize;
  uint32_t vecIterSize, row256;
  uint32_t totalBatchNum, batchNum, batchBegin, batchEnd;
  uint32_t batch2Size;
  uint32_t elementBegin, elementNum;
  uint16_t shapeInf[MAXDIM];
  uint32_t x1ElemBegin, x2ElemNum, x1ElemNum;
  uint32_t x1Stride0, x2Stride0;
  int64_t valBand{1 << 24};

  TBuf<QuePosition::VECCALC> TmpBuf, CBuf, ABuf, BBuf;
  TBuf<QuePosition::VECCALC> ResBuf, MaskBuf, X1FloatBuf;

  TQue<QuePosition::VECIN, BUFFER_NUM_SPEC3> X1Que, X2Que;
  TQue<QuePosition::VECOUT, BUFFER_NUM_SPEC3> YQue;
};