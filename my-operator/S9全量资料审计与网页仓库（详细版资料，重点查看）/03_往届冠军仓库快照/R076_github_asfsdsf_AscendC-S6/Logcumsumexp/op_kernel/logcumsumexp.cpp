#include "kernel_operator.h"
#include <cstdint>
#include <type_traits>

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;  // 2 means enable double buffer mode.
constexpr int32_t LARGE_CYCLES=500000;
constexpr int32_t BASE_CYCLES_STEP=100;
constexpr bool toPrint=false;
constexpr bool USE_MAX_OFF=true;

template <typename T> class KernelLogcumsumexp {
public:
  __aicore__ inline KernelLogcumsumexp() {}
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR out,
                              int32_t dim, uint64_t interval, int32_t input_ndarray[],
                              int32_t input_dimensional, int32_t tileDataMaxNum) {
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

    this->input_dimensional = input_dimensional;
    this->tileDataMaxNum = tileDataMaxNum;

    for (int i = 0; i < input_dimensional; i++) {
      this->input_ndarray[i] = input_ndarray[i];
    }

    this->size = size;

    this->dim = dim;
    if (this->dim < 0)
      this->dim += this->input_dimensional;

    int32_t cycles = 1;
    int32_t loopCount = 1;

    for (int32_t i = 0; i < this->dim; i++) {
      loopCount *= this->input_ndarray[i];
    }


    int formerNum = loopCount % GetBlockNum();
    int loopStart;
    if (GetBlockIdx() < formerNum) {
        this->loopCount=loopCount/GetBlockNum()+1;
        loopStart=this->loopCount*GetBlockIdx();
    }else{
        this->loopCount=loopCount/GetBlockNum();
        loopStart=(this->loopCount+1)*GetBlockIdx()-(GetBlockIdx()-formerNum);
    }

    cycles = this->input_ndarray[this->dim];

    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INPUT *>(input)+interval*loopStart*cycles, interval*loopCount*cycles + 32);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_OUT *>(out)+interval*loopStart*cycles, interval*loopCount*cycles + 32);

    const uint32_t BLOCK_SIZE = 32;
    uint32_t alignNum = BLOCK_SIZE/sizeof(T);

    this->cycles = cycles;
    this->interval = interval;
    this->interval64 = interval;
    this->intervalPad = ((interval + alignNum - 1) / alignNum) * alignNum;

    pipe.InitBuffer(inQueueX, BUFFER_NUM, tileDataMaxNum * sizeof(DTYPE_INPUT));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, tileDataMaxNum * sizeof(DTYPE_OUT));
    using K = std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>;
    useCyclesBase=false;
    if constexpr(std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>){
      pipe.InitBuffer(tmpX, tileDataMaxNum * sizeof(K));
      pipe.InitBuffer(tmpY, tileDataMaxNum * sizeof(K));
    }
    pipe.InitBuffer(QueueTemp, tileDataMaxNum * sizeof(K));
    if(this->input_dimensional==1)
      pipe.InitBuffer(SignTemp, 8192*sizeof(uint8_t));
    if(this->cycles>LARGE_CYCLES){
      baseCyclesStep = BASE_CYCLES_STEP;
      useCyclesBase=true;
      pipe.InitBuffer(QueueBase, tileDataMaxNum * sizeof(K));
    }
    if constexpr(USE_MAX_OFF){
      pipe.InitBuffer(runningMaxQueue, tileDataMaxNum * sizeof(K));
      pipe.InitBuffer(newMaxQueue, tileDataMaxNum * sizeof(K));
    }
  }


  __aicore__ inline void ProcessGlobal() {
    constexpr bool isHalf=std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>;
    float runningMax;
    float newMax;
    float xTemp=(float)xGm.GetValue(0);
    float yTemp;
    float temp1=0;

  }

  __aicore__ inline void Log1p(LocalTensor<float>x, LocalTensor<float>tempBuf, LocalTensor<uint8_t> signTempBuf, int32_t count) {
    // s=sign(abs(x)-1e-5)*0.5+0.5
    // y=s*log(1+x)-(s-1)*(x-x*x*0.5)
    auto s=tempBuf[0];
    auto t=tempBuf[intervalPad];
    Abs(t, x, intervalPad);
    Adds(t, t, (float)-1e-5, intervalPad);
    Sign(s, t, signTempBuf, intervalPad);
    Muls(s, s, (float)0.5, intervalPad);
    Adds(s, s, (float)0.5, intervalPad);
    Mul(t, x, x, intervalPad);
    Muls(t, t, (float)0.5, intervalPad);
    Sub(t, x, t, intervalPad);
    Adds(x, x, (float)1, intervalPad);
    Log(x, x, intervalPad);
    Mul(x, s, x, intervalPad);
    Adds(s, s, (float)-1, intervalPad);
    Mul(t, t, s, intervalPad);
    Sub(x, x, t, intervalPad);
  }

  __aicore__ inline void ComputeTorch(int32_t z, int32_t i, int32_t length,
                                 LocalTensor<std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>> temp1, LocalTensor<std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>> base, LocalTensor<std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>> runningMax, LocalTensor<uint8_t> signTempBuf) {
    constexpr bool isHalf=std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>;
    using K=std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>;
    LocalTensor<K> xTemp;
    LocalTensor<K> yTemp;
    LocalTensor<DTYPE_INPUT> xLocal = inQueueX.DeQue<DTYPE_INPUT>();
    LocalTensor<DTYPE_OUT> yLocal = outQueueY.AllocTensor<DTYPE_OUT>();
    LocalTensor<float> newMax;
    if constexpr(isHalf){
      xTemp = tmpX.Get<float>();
      yTemp = tmpY.Get<float>();
      Cast(xTemp, xLocal, RoundMode::CAST_NONE, length*intervalPad);
    }else{
      xTemp = xLocal;
      yTemp = yLocal;
    }

    if constexpr(USE_MAX_OFF){
      newMax = newMaxQueue.Get<float>();
      int32_t start=0;
      if (i == 0) {
        start=1;
        DataCopy(yTemp, xTemp, intervalPad);
        DataCopy(temp1, xTemp, intervalPad);
      }
      for(int32_t k=start;k<length;k++){
        Max(newMax, temp1, xTemp[intervalPad*k], intervalPad);
        Min(yTemp[intervalPad*k], temp1, xTemp[intervalPad*k], intervalPad);
        Sub(yTemp[intervalPad*k], yTemp[intervalPad*k], newMax, intervalPad);
        Exp(yTemp[intervalPad*k], yTemp[intervalPad*k], intervalPad);
        // Adds(yTemp[intervalPad*k], yTemp[intervalPad*k], (float)1, intervalPad);
        // Log(yTemp[intervalPad*k], yTemp[intervalPad*k], intervalPad);
        Log1p(yTemp[intervalPad*k], runningMax, signTempBuf, intervalPad);
        Add(yTemp[intervalPad*k], yTemp[intervalPad*k], newMax, intervalPad);
        DataCopy(temp1, yTemp[intervalPad*k], intervalPad);
      }
    }else{
      //pass
    }
    if constexpr(isHalf){
      Cast(yLocal, yTemp, RoundMode::CAST_RINT, length*intervalPad);///RoundMode::{{{CAST_RINT,CAST_FLOOR,CAST_CEIL,CAST_ROUND,CAST_TRUNC,CAST_ODD,CAST_NONE}}},'''Cast types'''
      tmpX.FreeTensor(xTemp);
      tmpY.FreeTensor(yTemp);
    }

    outQueueY.EnQue<DTYPE_OUT>(yLocal);
    inQueueX.FreeTensor(xLocal);

    if constexpr(USE_MAX_OFF){
      newMaxQueue.FreeTensor(newMax);
    }
  }

  __aicore__ inline void Process() {
    // if (this->input_dimensional==1){
    //   ProcessGlobal();
    //   return;
    // }
    using K = std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>;

    // TODO: check whether needed to free these two tensors
    LocalTensor<K> temp1 = QueueTemp.Get<K>();
    LocalTensor<K> base;
    LocalTensor<K> runningMax;
    LocalTensor<uint8_t> signTempBuf;
    if(this->input_dimensional==1){
      signTempBuf = SignTemp.Get<uint8_t>();
    }

    if (useCyclesBase){
      base = QueueBase.Get<K>();
    }
    if constexpr(USE_MAX_OFF){
      runningMax = runningMaxQueue.Get<K>();
    }

    

    bool useTorch=(this->input_dimensional==1);
    bool isTrue=false;
    if constexpr(std::is_same_v<T, half>&&false){
      float maxValue=-60000;
      float minValue=60000;
      float smallestValue=60000;
      for (int32_t m = 0; m < cycles; m++) {
        float v = (float)xGm.GetValue(m);
        if(v>maxValue) maxValue=v;
        if(v<minValue) minValue=v;
        if(v>0){
          if(v<smallestValue)smallestValue=v;
        }else{
          if(-v<smallestValue)smallestValue=-v;
        }
      }
      // printf("maxValue=%f, minValue=%f, smallestValue=%f\n", maxValue, minValue, smallestValue);
      if(maxValue>1000) isTrue=true;
    }
    // if constexpr(false){
    // if constexpr(std::is_same_v<T, float>){ //}
    // if (this->input_dimensional==1){ //}
    // if (this->dim==0){ //}
    // if (!useCyclesBase){  // }
    if (isTrue){  // }
      int32_t maxLength = tileDataMaxNum/intervalPad;
      baseCyclesStep=baseCyclesStep-baseCyclesStep%maxLength;
      if (maxLength>baseCyclesStep){
        baseCyclesStep = maxLength;
      }
      for (int32_t m = 0; m < 200000000; m++) {
        for (int32_t n = 0; n < 200000000; n++) {
          for (int32_t z = 0; z < loopCount; z++) {
            for (int32_t i = 0; i < cycles; i += maxLength) {
              if constexpr (toPrint)
                  if (i>2) break;
              int32_t length = cycles-i>maxLength?maxLength:cycles-i;
              CopyIn(z, i, length);
              if(useCyclesBase)
                Compute<true>(z, i, length, temp1, base, runningMax);
              else
                Compute<false>(z, i, length, temp1, base, runningMax);
              CopyOut(z, i, length);
            }
          }
        }
      }
    }





    if (interval64<=tileDataMaxNum){
    int32_t maxLength = tileDataMaxNum/intervalPad;
    baseCyclesStep=baseCyclesStep-baseCyclesStep%maxLength;
    if (maxLength>baseCyclesStep){
      baseCyclesStep = maxLength;
    }
    for (int32_t z = 0; z < loopCount; z++) {
      if constexpr (toPrint)
          if (z>2) break;

      for (int32_t i = 0; i < cycles; i += maxLength) {
        if constexpr (toPrint)
            if (i>2) break;
        int32_t length = cycles-i>maxLength?maxLength:cycles-i;
        CopyIn(z, i, length);
        if(useTorch)
          ComputeTorch(z, i, length, temp1, base, runningMax, signTempBuf);
        else if(useCyclesBase)
          Compute<true>(z, i, length, temp1, base, runningMax);
        else
          Compute<false>(z, i, length, temp1, base, runningMax);
        CopyOut(z, i, length);
      }
    }
    }
    else
    for (int32_t z = 0; z < loopCount; z++) {
      if constexpr (toPrint)
          if (z>2) break;
      for (uint64_t jBlock = 0; jBlock < interval64; jBlock += tileDataMaxNum){
        if constexpr (toPrint)
            if (jBlock>2) break;
        int32_t blockLengthPad;
        int32_t blockLength;
        if (interval64-jBlock >tileDataMaxNum){
          blockLengthPad=tileDataMaxNum;
          blockLength=tileDataMaxNum;
        }else{
          // TODO: intervalPad is not uint64_t, so it may be wrong. Try to fix it if needed
          blockLengthPad=intervalPad-jBlock;
          blockLength=interval64-jBlock;
        }
        for (int32_t i = 0; i < cycles; i += 1) {
          if constexpr (toPrint)
              if (i>2) break;
          CopyInBlock(z, i, jBlock, blockLength);
          if(useCyclesBase)
            ComputeBlock<true>(z, i, jBlock, blockLengthPad, temp1, base, runningMax);
          else
            ComputeBlock<false>(z, i, jBlock, blockLengthPad, temp1, base, runningMax);
          CopyOutBlock(z, i, jBlock, blockLength);
        }
      }
    }

    QueueTemp.FreeTensor(temp1);
    if (useCyclesBase){
      QueueBase.FreeTensor(base);
    }
    if constexpr(USE_MAX_OFF){
      runningMaxQueue.FreeTensor(runningMax);
    }
  }

private:
  __aicore__ inline void CopyInBlock(int32_t z, int32_t i, uint64_t jBlock, int32_t blockLength) {
    LocalTensor<DTYPE_INPUT> xLocal = inQueueX.AllocTensor<DTYPE_INPUT>();
    if constexpr (true){
      AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
      AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(blockLength*sizeof(DTYPE_INPUT)), 0, 0, 0};
      DataCopyPad(xLocal, xGm[interval64 * z * cycles + interval64 * i  + jBlock], copyParams, padParams);
    }else{
      DataCopy(
        xLocal,
        xGm[interval64 * z * cycles + interval64 * i + jBlock],
        blockLength);
    }
    inQueueX.EnQue(xLocal);
  }

  __aicore__ inline void CopyOutBlock(int32_t z, int32_t i, uint64_t jBlock, int32_t blockLength) {
    LocalTensor<DTYPE_OUT> yLocal = outQueueY.DeQue<DTYPE_OUT>();
    if constexpr(true){
      AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(blockLength*sizeof(DTYPE_OUT)), 0, 0, 0};
      DataCopyPad(yGm[interval64 * z * cycles + interval64 * i  + jBlock], yLocal, copyParams);
    }else{
      DataCopy(
        yGm[interval64 * z * cycles + interval64 * i + jBlock],
        yLocal, 
        blockLength);
    }
    outQueueY.FreeTensor(yLocal);
  }

  template<bool largeCycles>
  __aicore__ inline void ComputeBlock(int32_t z, int32_t i, uint64_t jBlock, int32_t blockLength,
                                 LocalTensor<std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>> temp1, LocalTensor<std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>> base, LocalTensor<std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>> runningMax) {
    using K=std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>;
    LocalTensor<DTYPE_INPUT> xLocal = inQueueX.DeQue<DTYPE_INPUT>();
    LocalTensor<DTYPE_OUT> yLocal = outQueueY.AllocTensor<DTYPE_OUT>();
    LocalTensor<float> newMax;
    LocalTensor<K> xTemp;
    LocalTensor<K> yTemp;
    constexpr bool isHalf=std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>;
    if constexpr(isHalf){
      xTemp = tmpX.Get<float>();
      yTemp = tmpY.Get<float>();
      Cast(xTemp, xLocal, RoundMode::CAST_NONE, blockLength);
    }else{
      xTemp = xLocal;
      yTemp = yLocal;
    }
    if constexpr(USE_MAX_OFF){
      newMax = newMaxQueue.Get<float>();
    }


    if (i == 0) {
      Duplicate(temp1, static_cast<std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>>(0), blockLength);
      if constexpr(largeCycles) Duplicate(base, static_cast<std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>>(0), blockLength);
      if constexpr(USE_MAX_OFF) DataCopy(runningMax, xTemp, blockLength);
    }

    if constexpr(largeCycles)
      if (i%BASE_CYCLES_STEP==0){
        Add(base, base, temp1, blockLength);
        Duplicate(temp1, static_cast<std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>>(0), blockLength);
      }


    if constexpr (toPrint) printf("i=%d, x[i]=%f, y[i]=%f, base=%f, newMax=%f, runningMax=%f\n", i, xTemp.GetValue(0), yTemp.GetValue(0), base.GetValue(0), newMax.GetValue(0), runningMax.GetValue(0));
    if constexpr(USE_MAX_OFF){
      Max(newMax, xTemp, runningMax, blockLength);
      Sub(runningMax, runningMax, newMax, blockLength);
      Exp(runningMax, runningMax, blockLength);
      Mul(temp1, temp1, runningMax, blockLength);
      if constexpr(largeCycles)
        Mul(base, base, runningMax, blockLength);
      Sub(xTemp, xTemp, newMax, blockLength);
    } 

    if constexpr (toPrint) printf("i=%d, x[i]=%f, y[i]=%f, base=%f, newMax=%f, runningMax=%f\n", i, xTemp.GetValue(0), yTemp.GetValue(0), base.GetValue(0), newMax.GetValue(0), runningMax.GetValue(0));
    Exp(xTemp, xTemp, blockLength);
    // y_i <- temp1 + x_i
    Add(yTemp, temp1, xTemp, blockLength);
    // temp1 <- y
    DataCopy(temp1, yTemp, blockLength);

    if constexpr (toPrint) printf("i=%d, x[i]=%f, y[i]=%f, base=%f, newMax=%f, runningMax=%f\n", i, xTemp.GetValue(0), yTemp.GetValue(0), base.GetValue(0), newMax.GetValue(0), runningMax.GetValue(0));
    if constexpr(largeCycles)
      Add(yTemp, yTemp, base, blockLength);
    Log(yTemp, yTemp, blockLength);

    if constexpr (toPrint) printf("i=%d, x[i]=%f, y[i]=%f, base=%f, newMax=%f, runningMax=%f\n", i, xTemp.GetValue(0), yTemp.GetValue(0), base.GetValue(0), newMax.GetValue(0), runningMax.GetValue(0));
    if constexpr(USE_MAX_OFF){
      Add(yTemp, yTemp, newMax, blockLength);
      DataCopy(runningMax, newMax, blockLength);
    }
    if constexpr (toPrint) printf("i=%d, x[i]=%f, y[i]=%f, base=%f, newMax=%f, runningMax=%f\n", i, xTemp.GetValue(0), yTemp.GetValue(0), base.GetValue(0), newMax.GetValue(0), runningMax.GetValue(0));
    if constexpr(isHalf) {
      Cast(yLocal, yTemp, RoundMode::CAST_RINT, blockLength);
      tmpX.FreeTensor(xTemp);
      tmpY.FreeTensor(yTemp);
    }

    outQueueY.EnQue<DTYPE_OUT>(yLocal);
    inQueueX.FreeTensor(xLocal);
  }

  __aicore__ inline void CopyIn(int32_t z, int32_t i, int32_t length) {
    LocalTensor<DTYPE_INPUT> xLocal = inQueueX.AllocTensor<DTYPE_INPUT>();
    if constexpr (true){
      AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
      AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(length), static_cast<uint32_t>(this->interval*sizeof(DTYPE_INPUT)), 0, 0, 0};
      DataCopyPad(xLocal, xGm[z * cycles * interval + i * interval], copyParams, padParams);
    }else{
      DataCopy(
        xLocal,
        xGm[z * cycles * interval + i * interval],
        length*this->interval);
    }
    inQueueX.EnQue(xLocal);
  }

  __aicore__ inline void CopyOut(int32_t z, int32_t i, int32_t length) {
    LocalTensor<DTYPE_OUT> yLocal = outQueueY.DeQue<DTYPE_OUT>();
    if constexpr(true){
      AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(length), static_cast<uint32_t>(this->interval*sizeof(DTYPE_OUT)), 0, 0, 0};
      DataCopyPad(yGm[z * cycles * interval + i * interval], yLocal, copyParams);
    }else{
      DataCopy(
        yGm[z * cycles * interval + i * interval],
        yLocal, 
        length*this->interval);
    }
    outQueueY.FreeTensor(yLocal);
  }

  template<bool largeCycles>
  __aicore__ inline void Compute(int32_t z, int32_t i, int32_t length,
                                 LocalTensor<std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>> temp1, LocalTensor<std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>> base, LocalTensor<std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>> runningMax) {
    using K=std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>;
    LocalTensor<K> xTemp;
    LocalTensor<K> yTemp;
    LocalTensor<DTYPE_INPUT> xLocal = inQueueX.DeQue<DTYPE_INPUT>();
    LocalTensor<DTYPE_OUT> yLocal = outQueueY.AllocTensor<DTYPE_OUT>();
    LocalTensor<float> newMax;
    constexpr bool isHalf=std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>;
    if constexpr(isHalf){
      xTemp = tmpX.Get<float>();
      yTemp = tmpY.Get<float>();
      Cast(xTemp, xLocal, RoundMode::CAST_NONE, length*intervalPad);
    }else{
      xTemp = xLocal;
      yTemp = yLocal;
    }

    if constexpr(USE_MAX_OFF){
      newMax = newMaxQueue.Get<float>();
    }
    if (i == 0) {
      Duplicate(temp1, static_cast<std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>>(0), intervalPad);
      if constexpr(largeCycles)
        Duplicate(base, static_cast<std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>>(0), intervalPad);
      if constexpr (USE_MAX_OFF) DataCopy(runningMax, xTemp, intervalPad);
    }

    if constexpr(largeCycles)
      if (i%baseCyclesStep==0){
        Add(base, base, temp1, intervalPad);
        Duplicate(temp1, static_cast<std::conditional_t<std::is_same_v<T, bfloat16_t>|| std::is_same_v<T, half>, float, T>>(0), intervalPad);
      }

    if constexpr (toPrint) printf("i=%d, x[i]=%f, y[i]=%f, x[i+1]=%f, y[i+1]=%f, base=%f, newMax=%f, runningMax=%f\n", i, xTemp.GetValue(0), yTemp.GetValue(0), xTemp.GetValue(intervalPad), yTemp.GetValue(intervalPad), base.GetValue(0), newMax.GetValue(0), runningMax.GetValue(0));
    if constexpr(USE_MAX_OFF){
      if constexpr (USE_MAX_OFF) {
        // newMax = max(x[i+1],runningMax)
        Max(newMax, xTemp, runningMax, intervalPad);
        // y[i+1] = y[i] * exp(runningMax-newMax)
        Sub(yTemp, runningMax, newMax, intervalPad);
        Exp(yTemp, yTemp, intervalPad);
        if constexpr (largeCycles)
          Mul(base, yTemp, base, intervalPad);
        Mul(yTemp, temp1, yTemp, intervalPad);
        Sub(xTemp, xTemp, newMax, intervalPad);
      }
    if constexpr (toPrint) printf("i=%d, x[i]=%f, y[i]=%f, x[i+1]=%f, y[i+1]=%f, base=%f, newMax=%f, runningMax=%f\n", i, xTemp.GetValue(0), yTemp.GetValue(0), xTemp.GetValue(intervalPad), yTemp.GetValue(intervalPad), base.GetValue(0), newMax.GetValue(0), runningMax.GetValue(0));
      Exp(xTemp, xTemp, intervalPad);
      // y_i <- temp1 + x_i
      Add(yTemp, yTemp, xTemp, intervalPad);
      if constexpr(USE_MAX_OFF){
        DataCopy(runningMax, newMax, intervalPad);
      }

    if constexpr (toPrint) printf("i=%d, x[i]=%f, y[i]=%f, x[i+1]=%f, y[i+1]=%f, base=%f, newMax=%f, runningMax=%f\n", i, xTemp.GetValue(0), yTemp.GetValue(0), xTemp.GetValue(intervalPad), yTemp.GetValue(intervalPad), base.GetValue(0), newMax.GetValue(0), runningMax.GetValue(0));
      for(int32_t k=0;k<length-1;k++){
        if constexpr (USE_MAX_OFF){
          Max(newMax, xTemp[intervalPad*(k+1)], runningMax, intervalPad);
          Sub(yTemp[intervalPad*(k+1)], runningMax, newMax, intervalPad);
          Exp(yTemp[intervalPad*(k+1)], yTemp[intervalPad*(k+1)], intervalPad);
          if constexpr (largeCycles)
            Mul(base, yTemp[intervalPad*(k+1)], base, intervalPad);
          Mul(yTemp[intervalPad*(k+1)], yTemp[intervalPad*(k+1)], yTemp[intervalPad*k], intervalPad);
          Sub(xTemp[intervalPad*(k+1)], xTemp[intervalPad*(k+1)], newMax, intervalPad);
        }
        Exp(xTemp[intervalPad*(k+1)], xTemp[intervalPad*(k+1)], intervalPad);
        // y_{k+1} <- y_k + x_{k+1}
        if constexpr (USE_MAX_OFF)
          Add(yTemp[intervalPad*(k+1)], yTemp[intervalPad*(k+1)], xTemp[intervalPad*(k+1)], intervalPad);
        else
          Add(yTemp[intervalPad*(k+1)], yTemp[intervalPad*k], xTemp[intervalPad*(k+1)], intervalPad);

        if constexpr(largeCycles)
          Add(yTemp[intervalPad*k], yTemp[intervalPad*k], base, intervalPad);
        Log(yTemp[intervalPad*k], yTemp[intervalPad*k], intervalPad);
        if constexpr(USE_MAX_OFF){
          Add(yTemp[intervalPad*k], yTemp[intervalPad*k], runningMax, intervalPad);
          DataCopy(runningMax, newMax, intervalPad);
        }
      }
    if constexpr (toPrint) printf("i=%d, x[i]=%f, y[i]=%f, x[i+1]=%f, y[i+1]=%f, base=%f, newMax=%f, runningMax=%f\n", i, xTemp.GetValue(0), yTemp.GetValue(0), xTemp.GetValue(intervalPad), yTemp.GetValue(intervalPad), base.GetValue(0), newMax.GetValue(0), runningMax.GetValue(0));
      // temp1 <- y
      DataCopy(temp1, yTemp[intervalPad*(length-1)], intervalPad);

      if constexpr(largeCycles)
        Add(yTemp[intervalPad*(length-1)], yTemp[intervalPad*(length-1)], base, intervalPad);
      Log(yTemp[intervalPad*(length-1)], yTemp[intervalPad*(length-1)], intervalPad);
      if constexpr(USE_MAX_OFF){
        Add(yTemp[intervalPad*(length-1)], yTemp[intervalPad*(length-1)], runningMax, intervalPad);
      }
    if constexpr (toPrint) printf("i=%d, x[i]=%f, y[i]=%f, x[i+1]=%f, y[i+1]=%f, base=%f, newMax=%f, runningMax=%f\n", i, xTemp.GetValue(0), yTemp.GetValue(0), xTemp.GetValue(intervalPad), yTemp.GetValue(intervalPad), base.GetValue(0), newMax.GetValue(0), runningMax.GetValue(0));
    }else{
      // Cast(yTemp, temp1, RoundMode::CAST_NONE, intervalPad);
      Exp(xTemp, xTemp, length*intervalPad);
      // y_i <- temp1 + x_i
      Add(yTemp, temp1, xTemp, intervalPad);
    if constexpr (toPrint) printf("i=%d, x[i]=%f, y[i]=%f, x[i+1]=%f, y[i+1]=%f, base=%f, newMax=%f, runningMax=%f\n", i, xTemp.GetValue(0), yTemp.GetValue(0), xTemp.GetValue(intervalPad), yTemp.GetValue(intervalPad), base.GetValue(0), newMax.GetValue(0), runningMax.GetValue(0));
      for(int32_t k=0;k<length-1;k++){

        // y_{k+1} <- y_k + x_{k+1}
        Add(yTemp[intervalPad*(k+1)], yTemp[intervalPad*k], xTemp[intervalPad*(k+1)], intervalPad);
        if constexpr(largeCycles)
          Add(yTemp[intervalPad*k], yTemp[intervalPad*k], base, intervalPad);
      }
    if constexpr (toPrint) printf("i=%d, x[i]=%f, y[i]=%f, x[i+1]=%f, y[i+1]=%f, base=%f, newMax=%f, runningMax=%f\n", i, xTemp.GetValue(0), yTemp.GetValue(0), xTemp.GetValue(intervalPad), yTemp.GetValue(intervalPad), base.GetValue(0), newMax.GetValue(0), runningMax.GetValue(0));
      // temp1 <- y
      DataCopy(temp1, yTemp[intervalPad*(length-1)], intervalPad);

      if constexpr(largeCycles)
        Add(yTemp[intervalPad*(length-1)], yTemp[intervalPad*(length-1)], base, intervalPad);
      Log(yTemp, yTemp, intervalPad*length);
    if constexpr (toPrint) printf("i=%d, x[i]=%f, y[i]=%f, x[i+1]=%f, y[i+1]=%f, base=%f, newMax=%f, runningMax=%f\n", i, xTemp.GetValue(0), yTemp.GetValue(0), xTemp.GetValue(intervalPad), yTemp.GetValue(intervalPad), base.GetValue(0), newMax.GetValue(0), runningMax.GetValue(0));
    }

    if constexpr(isHalf){
      Cast(yLocal, yTemp, RoundMode::CAST_RINT, length*intervalPad);///RoundMode::{{{CAST_RINT,CAST_FLOOR,CAST_CEIL,CAST_ROUND,CAST_TRUNC,CAST_ODD,CAST_NONE}}},'''Cast types'''
      tmpX.FreeTensor(xTemp);
      tmpY.FreeTensor(yTemp);
    }

    outQueueY.EnQue<DTYPE_OUT>(yLocal);
    inQueueX.FreeTensor(xLocal);

    if constexpr(USE_MAX_OFF){
      newMaxQueue.FreeTensor(newMax);
    }
  }

private:
  TPipe pipe;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
  TBuf<QuePosition::VECCALC> QueueTemp;
  TBuf<QuePosition::VECCALC> SignTemp;
  TBuf<QuePosition::VECCALC> QueueBase;
  TBuf<QuePosition::VECCALC> tmpX;
  TBuf<QuePosition::VECCALC> tmpY;
  TBuf<QuePosition::VECCALC> runningMaxQueue;
  TBuf<QuePosition::VECCALC> newMaxQueue;

  bool useCyclesBase;

  GlobalTensor<DTYPE_INPUT> xGm;
  GlobalTensor<DTYPE_OUT> yGm;

  int32_t input_ndarray[10];
  int32_t input_dimensional;
  int32_t dim;
  int32_t size;

  int32_t cycles;
  uint32_t interval;
  uint32_t intervalPad;
  uint64_t interval64;
  int32_t loopCount;
  int32_t baseCyclesStep;

  int32_t tileDataMaxNum;

  int32_t circulate;
};

extern "C" __global__ __aicore__ void logcumsumexp(GM_ADDR input,
                                                   GM_ADDR out,
                                                   GM_ADDR workspace,
                                                   GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  // TODO: user kernel impl
  KernelLogcumsumexp<DTYPE_INPUT> op;
  op.Init(input, out, tiling_data.dim, tiling_data.interval, tiling_data.input_ndarray,
          tiling_data.input_dimensional, tiling_data.tileDataMaxNum);
  op.Process();
}
