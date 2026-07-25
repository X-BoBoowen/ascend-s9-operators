#include "kernel_operator.h"
using namespace AscendC;

// #define PRINTF
#define MAXDIM 8
#define BLOCK_BYTES_SIZE 32
#define BUFFER_NUM 2

#define WRITEBACK 1024

template <typename T>
class LcmMultSca {
 public:
  __aicore__ inline LcmMultSca() {}
  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t maxDim,
                              uint32_t shapeInf1[MAXDIM], uint32_t shapeInf2[MAXDIM],
                              TPipe *pipeIn) {
    this->pipe = pipeIn;
    this->maxDim = maxDim;
    // printf("in sca init\n");
    for (int i = 0; i < maxDim; i++) {
      x1shape[i] = shapeInf1[i];
      x2shape[i] = shapeInf2[i];
      maxshape[i] = shapeInf1[i] > shapeInf2[i] ? shapeInf1[i] : shapeInf2[i];
#ifdef PRINTF
      printf("i x1 x2 max= %d %d %d %d\n", i, x1shape[i], x2shape[i], maxshape[i]);
#endif
    }
    this->coreNum = GetBlockNum();
    this->coreIndex = GetBlockIdx();
    int maxCol = x1shape[0] > x2shape[0] ? x1shape[0] : x2shape[0];

    blockSize = 32 / sizeof(DTYPE_X1);

    int maxBlock = (maxCol + blockSize - 1) / blockSize;
    int colDiv = maxBlock / coreNum;
    int colMod = maxBlock % coreNum;

    colBegin = colDiv * coreIndex + (coreIndex < colMod ? coreIndex : colMod);
    colBegin = colBegin * blockSize;

    colNum = colDiv + (coreIndex < colMod ? 1 : 0);
    colNum = colNum * blockSize;
    if (coreIndex == coreNum - 1) {
      colNum = maxCol - colBegin;
    }
    uint64_t x1GmOffset = 0, x2GmOffset = 0;

    maxshape[0] = colNum;
    uint64_t yGmOffset = colBegin;
    if (x1shape[0] != 1) {
      x1shape[0] = colNum;
      x1GmOffset = colBegin;
    }
    if (x2shape[0] != 1) {
      x2shape[0] = colNum;
      x2GmOffset = colBegin;
    }
    sizey = 1;
    uint64_t mul1 = 1, mul2 = 1;
    for (int i = maxDim - 1; i >= 0; i--) {
      if (x1shape[i] == x2shape[i]) {
        x1stride[i] = mul1;
        x2stride[i] = mul2;
      } else if (x1shape[i] == 1) {
        x1stride[i] = 0;
        x2stride[i] = mul2;
      } else if (x2shape[i] == 1) {
        x1stride[i] = mul1;
        x2stride[i] = 0;
      }
      ystride[i] = sizey;
      mul1 *= x1shape[i];
      mul2 *= x2shape[i];
      sizey *= maxshape[i];
    }
    size1 = mul1;
    size2 = mul2;
#ifdef PRINTF
    // for(int i=0;i<maxDim;i++){
    //     printf("i str1 str2 stry= %d %d %d %d\n",i,x1stride[i],x2stride[i],ystride[i]);
    // }
    printf("size1,size2,sizey = %d %d %d\n", size1, size2, sizey);
    printf("colNum,colbegin = %d %d\n", colNum, colBegin);
#endif
    x1GmOffset *= x1stride[0];
    x2GmOffset *= x2stride[0];
    yGmOffset *= ystride[0];
#ifdef PRINTF
    printf("offset x1,x2,y = %d %d %d\n", x1GmOffset, x2GmOffset, yGmOffset);
#endif
    x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x1 + x1GmOffset, size1);
    x2Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x2 + x2GmOffset, size2);
    yGm.SetGlobalBuffer((__gm__ DTYPE_X1 *)y + yGmOffset, sizey);

    pipe->InitBuffer(OutY, BUFFER_NUM, 8192);
  }

  __aicore__ inline void Process() {
    LocalTensor<DTYPE_Y> outYLocal;
    int cnt = 0;
    uint64_t beginWrite = 0;
    for (uint64_t i = 0; i < sizey; i++) {
      if (i % WRITEBACK == 0) {
        outYLocal = OutY.AllocTensor<DTYPE_Y>();
        beginWrite = i;
      }

      uint64_t addr1 = 0, addr2 = 0;
      for (int j = 0; j < maxDim; j++) {
        addr1 += ((i / ystride[j]) % maxshape[j]) * x1stride[j];
        addr2 += ((i / ystride[j]) % maxshape[j]) * x2stride[j];
      }
      // printf("addr1,addr2,i = %d %d %d\n",addr1,addr2,i);
      auto y = Cal(addr1, addr2, i);
      outYLocal.SetValue(i % WRITEBACK, y);
      cnt++;

      if (i % WRITEBACK == WRITEBACK - 1) {
        OutY.EnQue(outYLocal);
        auto outYLocaltmp = OutY.DeQue<DTYPE_Y>();
        DataCopyParams copyParams{(uint16_t)1, static_cast<uint16_t>(cnt * sizeof(DTYPE_Y)), 0, 0};
        DataCopyPad(yGm[beginWrite], outYLocaltmp, copyParams);
        OutY.FreeTensor(outYLocaltmp);
        cnt = 0;
      }
    }
    if (cnt != 0) {
      OutY.EnQue(outYLocal);
      auto outYLocaltmp = OutY.DeQue<DTYPE_Y>();
      DataCopyParams copyParams{(uint16_t)1, static_cast<uint16_t>(cnt * sizeof(DTYPE_Y)), 0, 0};
      DataCopyPad(yGm[beginWrite], outYLocaltmp, copyParams);
      OutY.FreeTensor(outYLocaltmp);
      cnt = 0;
    }
  }

  __aicore__ inline T Cal(uint64_t addr1, uint64_t addr2, uint64_t cury) {
    T x1 = x1Gm.GetValue(addr1);
    T x2 = x2Gm.GetValue(addr2);
    // int64_t x1 = 5;
    // int64_t x2 = 10;
    // int64_t xll1 = x1;
    // int64_t xll2 = x2;
    T abs1 = x1 < 0 ? -x1 : x1;
    T abs2 = x2 < 0 ? -x2 : x2;
    // if(xll1 == INT64_MIN){
    //     abs1 = static_cast<uint64_t>(xll1);
    // }
    // if(abs2 == INT64_MIN){
    //     abs2 = static_cast<uint64_t>(xll2);
    // }

    T g = gcd(abs1, abs2);
    T res = (abs1 / g) * abs2;
    if constexpr (!std::is_same_v<T, std::int8_t>) {
      res = res < 0 ? -res : res;
    }

    // printf("addr x1 x2 abs1 abs2 g abs1/g res: %d, %d %d, %d %d, %d %d
    // %d\n",addr1,x1,x2,abs1,abs2,g,res1,res);

    return res;
  }

  __aicore__ inline T gcd(T a, T b) {
    while (b != 0) {
      T tmp = a;
      a = b;
      b = tmp % b;
    }
    return a;
  }

 private:
  GlobalTensor<DTYPE_X1> x1Gm, x2Gm, yGm;
  TPipe *pipe;
  uint64_t x1shape[MAXDIM], x2shape[MAXDIM], maxshape[MAXDIM];
  uint64_t x1stride[MAXDIM], x2stride[MAXDIM], ystride[MAXDIM];
  uint64_t size1, size2, sizey;
  uint32_t maxDim;
  uint64_t colBegin, colNum;
  uint32_t coreIndex, coreNum;
  uint32_t blockSize;
  TQue<QuePosition::VECOUT, BUFFER_NUM> OutY;
  // TBuf<QuePosition::VECCALC> TmpBuf, CntBuf, ABuf, BBuf;
};