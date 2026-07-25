#include "kernel_operator.h"
using namespace AscendC;
#define BUFFER_NUM 1
#define BLOCK_SIZE 256
template<typename T>class KernelLCM{
    public:
        __aicore__ inline KernelLCM(){}
        __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out, uint32_t blockPerCore, uint16_t nAcores, uint16_t nBcores,
            uint16_t maxBlockPerIter, TPipe* pipeIn) {
            this->pipe = pipeIn;
            int coreIdx = GetBlockIdx();
            uint32_t elemPerBlock = BLOCK_SIZE / sizeof(T);
            this->maxBlockPerIter = maxBlockPerIter;

            uint32_t globalIndex = coreIdx * (blockPerCore + 1) * elemPerBlock; //全局偏移 单位是元素数量
            elemPerIter = elemPerBlock * maxBlockPerIter;                      //每次迭代最多处理的元素数量
            if (coreIdx < nAcores) {
                this->nElem = (blockPerCore + 1) * elemPerBlock;
                nIter = (blockPerCore + 1 + maxBlockPerIter - 1) / maxBlockPerIter;
                tailElem = ((blockPerCore + 1) % maxBlockPerIter) * elemPerBlock;
            } else {
            // printf("globalIndex now: %d\n", globalIndex);
                globalIndex = globalIndex - (coreIdx - nAcores) * elemPerBlock;
                this->nElem = blockPerCore * elemPerBlock;
                nIter = (blockPerCore + maxBlockPerIter - 1) / maxBlockPerIter;
                tailElem = (blockPerCore % maxBlockPerIter) * elemPerBlock;
            }
            // printf("globalIndex: %d blockPerCore:%d elemPerBlock:%d nAcores:%d\n",globalIndex,blockPerCore,elemPerBlock,nAcores);
            // printf("nElem:%d blockPerCore:%d elemPerIter:%d tailElem:%d nIter:%d\n", nElem, blockPerCore,elemPerIter, tailElem,nIter);
            curElem = elemPerIter;
            inputGm.SetGlobalBuffer((__gm__ T*)input + globalIndex, nElem);
            otherGm.SetGlobalBuffer((__gm__ T*)other + globalIndex, nElem);
            outGm.SetGlobalBuffer((__gm__ T*)out + globalIndex, nElem);
            pipe->InitBuffer(outQue, BUFFER_NUM, elemPerIter * sizeof(T));
            pipe->InitBuffer(otherQue, BUFFER_NUM, elemPerIter * sizeof(T));
            pipe->InitBuffer(inputQue, BUFFER_NUM, elemPerIter * sizeof(T));
            pipe->InitBuffer(inputFloatQue, elemPerIter * sizeof(float));
            pipe->InitBuffer(otherFloatQue, elemPerIter * sizeof(float));
            pipe->InitBuffer(outFloatQue, elemPerIter * sizeof(float));

            pipe->InitBuffer(x1Que, elemPerIter * sizeof(float));
            pipe->InitBuffer(x2Que, elemPerIter * sizeof(float));
            pipe->InitBuffer(maskQue, elemPerIter);
            maskLocal = maskQue.Get<uint16_t>();
            inputFloatLocal = inputFloatQue.Get<float>();
            otherFloatLocal = otherFloatQue.Get<float>();
            outFloatLocal = outFloatQue.Get<float>();
            x1Local = x1Que.Get<float>();
            x2Local = x2Que.Get<float>();
        }
    
        __aicore__ inline void copyIn(int i) {
            auto inputLocal = inputQue.AllocTensor<T>();
            auto otherLocal = otherQue.AllocTensor<T>();
            DataCopy(inputLocal, inputGm[elemPerIter * i], curElem);
            DataCopy(otherLocal, otherGm[elemPerIter * i], curElem);
            inputQue.EnQue(inputLocal);
            otherQue.EnQue(otherLocal);
        }
        __aicore__ inline T gcd(T a, T b) {
            while (b != 0) {
              T tmp = a;
              a = b;
              b = tmp % b;
            }
            return a;
        }
        __aicore__ inline T fullgcd(T a, T b) {
            T absa = a < 0 ? -a : a;
            T absb = b < 0 ? -b : b;
        
            T g = gcd(absa, absb);
            T out = (absa / g) * absb;
        
            if constexpr (!std::is_same_v<T, std::int8_t>) {
                out = out < 0 ? -out : out;
            }
        
            return out;
          }

        __aicore__ inline void compute(int iterIdx) {
            auto outLocal = outQue.AllocTensor<T>();
            auto inputLocal = inputQue.DeQue<T>();
            auto otherLocal = otherQue.DeQue<T>();
            int i = 0;
              for (; i < curElem; i ++) {
              T x1 = inputLocal.GetValue(i);
              T x2 = otherLocal.GetValue(i);

              T res = fullgcd(x1, x2);
              outLocal.SetValue(i, res);
            }
            outQue.EnQue(outLocal);
            inputQue.FreeTensor(inputLocal);
            otherQue.FreeTensor(otherLocal);
        }

        __aicore__ inline void computeVec(int iterIdx) {
            auto outLocal = outQue.AllocTensor<int32_t>();
            auto inputLocal = inputQue.DeQue<int32_t>();
            auto otherLocal = otherQue.DeQue<int32_t>();
            Cast(inputFloatLocal, inputLocal, RoundMode::CAST_NONE, curElem);
            Cast(otherFloatLocal, otherLocal, RoundMode::CAST_NONE, curElem);

            Abs(inputFloatLocal, inputFloatLocal, curElem);
            Abs(otherFloatLocal, otherFloatLocal, curElem);

            Max(x1Local, inputFloatLocal, otherFloatLocal, curElem);
            Min(x2Local, inputFloatLocal, otherFloatLocal, curElem);

            Cast(otherLocal, otherFloatLocal, RoundMode::CAST_RINT, curElem);
            float scala = 0;
            int curElemAlign = (curElem + 63)/64*64;
            for (int i = 0; i < 18; i++) {
                CompareScalar(maskLocal, x2Local, scala, AscendC::CMPMODE::EQ, curElemAlign);
                Select(outFloatLocal, maskLocal, x1Local, outFloatLocal, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, curElemAlign);      
                Fmod(otherFloatLocal, x1Local, x2Local, curElem);
                tempLocal = x1Local;
                x1Local = x2Local;
                x2Local = otherFloatLocal;
                otherFloatLocal = tempLocal;
            }
            Div(outFloatLocal, inputFloatLocal, outFloatLocal, curElem);
            Cast(outLocal, outFloatLocal, RoundMode::CAST_RINT, curElem);
            Mul(outLocal, outLocal, otherLocal, curElem);

            outQue.EnQue(outLocal);
            inputQue.FreeTensor(inputLocal);
            otherQue.FreeTensor(otherLocal);
        }

        __aicore__ inline void copyOut(int iterIdx) {
            auto outLocal = outQue.DeQue<T>();
            DataCopy(outGm[elemPerIter * iterIdx], outLocal, curElem);
            outQue.FreeTensor(outLocal);
        }
        __aicore__ inline void Process() {
            for (int i = 0; i < nIter; i++) {
                if (i == nIter -1) {
                    curElem = tailElem;
                }
                copyIn(i);
                computeVec(i);
                copyOut(i);
            }
        }
    private:
        GlobalTensor<T> inputGm, otherGm, outGm;
        TPipe* pipe;
        uint16_t maxBlockPerIter;
        uint32_t nElem, nIter, elemPerIter, tailElem, curElem;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
        TQue<QuePosition::VECIN, BUFFER_NUM> inputQue;
        TQue<QuePosition::VECIN, BUFFER_NUM> otherQue;
        TBuf<QuePosition::VECCALC> inputFloatQue, otherFloatQue, outFloatQue, x1Que, x2Que, maskQue;
        LocalTensor<float>inputFloatLocal, otherFloatLocal, x1Local, x2Local, tempLocal, outFloatLocal;
        LocalTensor<uint16_t>maskLocal;
        // LocalTensor<T>signLocal;
    
};

template<typename T>class KernelLCMBcast{
    public:
        __aicore__ inline KernelLCMBcast(){}
        __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR output, uint64_t tail, uint64_t piter, uint64_t smallDataNum,uint64_t mid, uint64_t pre, TPipe* pipeIn){
            // this->pipe = pipe;
            this->piter = piter;
            uint64_t bigDataNum = smallDataNum + 1; 
            uint64_t coreNum = AscendC::GetBlockIdx();
            globalIndex = bigDataNum * coreNum;
            if (coreNum < tail) {
                this->coreDataNum = bigDataNum;
            } else {
                this->coreDataNum = smallDataNum;
                globalIndex -= (bigDataNum - smallDataNum)*(coreNum - tail);
            }
            // this->globalIndex = globalIndex;
            // globalIndex *= (piter);
            uint64_t totalLength = coreDataNum*piter*mid*pre;
            this->mid = mid;
            this->pre = pre;
            // printf("piter %d pre:%d mid:%d\n",piter,pre, mid);
            // printf("globalIndex:%d totalLength: %d coreDataNum:%d piter:%d\n",globalIndex, totalLength, coreDataNum,piter);
            inputGm.SetGlobalBuffer((__gm__ T*)input + globalIndex*piter, totalLength);
            otherGm.SetGlobalBuffer((__gm__ T*)other, piter*pre);
            outputGm.SetGlobalBuffer((__gm__ T*)output + globalIndex*piter, totalLength);
            pipeIn->InitBuffer(inQueueInput, BUFFER_NUM, piter*sizeof(T));
            pipeIn->InitBuffer(inQueueOther, BUFFER_NUM, piter*sizeof(T));
            pipeIn->InitBuffer(outQueue, BUFFER_NUM, piter*sizeof(T));
            // pipeIn->InitBuffer(sign, piter*sizeof(T));


            // AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*pre*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
            // AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            // otherLocal = inQueueOther.AllocTensor<T>();
            // signLocal = sign.Get<T>();
            // AscendC::DataCopyPad(otherLocal, otherGm, copyParams, padParams); 
            // AscendC::TQueSync<PIPE_MTE2, PIPE_V> sync;
            // sync.SetFlag(0);
            // sync.WaitFlag(0);

        }
    
        // __aicore__ inline void copyIn(int i) {
        //     auto inputLocal = inQueueInput.AllocTensor<T>();
        //     auto otherLocal = otherQue.AllocTensor<T>();
        //     DataCopy(inputLocal, inputGm[elemPerIter * i], curElem);
        //     DataCopy(otherLocal, otherGm[elemPerIter * i], curElem);
        //     inQueueInput.EnQue(inputLocal);
        //     otherQue.EnQue(otherLocal);
        // }

        __aicore__ inline T gcd(T a, T b) {
            while (b != 0) {
              T tmp = a;
              a = b;
              b = tmp % b;
            }
            return a;
        }
        __aicore__ inline T fullgcd(T a, T b) {
            T absa = a < 0 ? -a : a;
            T absb = b < 0 ? -b : b;
        
            T g = gcd(absa, absb);
            T out = (absa / g) * absb;
        
            if constexpr (!std::is_same_v<T, std::int8_t>) {
                out = out < 0 ? -out : out;
            }
        
            return out;
          }

        __aicore__ inline void Process() {
            // int32_t loopCount = this->tileNum;
            midIndx = globalIndex/mid;
            // printf("midIndx: %d\n", midIndx);
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位 
            otherLocal = inQueueOther.AllocTensor<T>();
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(otherLocal, otherGm[midIndx*piter], copyParams, padParams); // 从GM-
            inQueueOther.EnQue(otherLocal);
            otherLocal = inQueueOther.DeQue<T>();
            
            for (int i = globalIndex; i <globalIndex + coreDataNum; i++) {
                if (i/mid != midIndx) {
                    midIndx = i/mid;
                    // printf("midIndx: %d\n", midIndx);
                    AscendC::DataCopyPad(otherLocal, otherGm[midIndx*piter], copyParams, padParams); // 从GM-
                    inQueueOther.EnQue(otherLocal);
                    otherLocal = inQueueOther.DeQue<T>();
                } 
                inLocal = inQueueInput.AllocTensor<T>();
                outLocal =  outQueue.AllocTensor<T>();
                AscendC::DataCopyPad(inLocal, inputGm[(i-globalIndex)*piter], copyParams, padParams); // 从GM-
                inQueueInput.EnQue(inLocal);
                inLocal = inQueueInput.DeQue<T>();
                for (int j = 0; j < piter; j++) {
                    T x1 = inLocal.GetValue(j);
                    T x2 = otherLocal.GetValue(j);
                    T res = fullgcd(x1, x2);
                    // printf("%d %d %d\n", x1, x2, res);

                    outLocal.SetValue(j, res);
                }
                
                outQueue.EnQue(outLocal);
                
                outLocal = outQueue.DeQue<T>();
                AscendC::DataCopyPad(outputGm[(i-globalIndex)*piter],outLocal, copyParams);
                outQueue.FreeTensor(outLocal);
                inQueueInput.FreeTensor(inLocal);
            }
        }
    private:
        uint32_t totalLength,resLength,stride,iterStep, axesDim,coreBatch,mid, pre;
        uint32_t tileDataNum, processDataNum,tailDataNum ,coreDataNum,tileNum, midIndx;
        uint32_t padTimes, globalIndex, piter;
        AscendC::TPipe *pipe;
        AscendC::GlobalTensor<T> inputGm;
        AscendC::GlobalTensor<T> otherGm;
        AscendC::GlobalTensor<T> outputGm;
        TQue<QuePosition::VECIN, BUFFER_NUM>inQueueInput, inQueueOther;
        TQue<QuePosition::VECOUT, BUFFER_NUM>outQueue; 
        LocalTensor<T> inLocal, otherLocal, outLocal;

        TBuf<AscendC::TPosition::VECCALC>modA, modB, modC, temp, inputFloatBuf;
};

template <typename T> class KernelLCMint64 {
    public:
      __aicore__ inline KernelLCMint64() {}
      __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out, uint32_t blockPerCore, uint16_t nAcores, uint16_t nBcores,
        uint16_t maxBlockPerIter, TPipe* pipeIn) {
        this->pipe = pipeIn;
        int coreIdx = GetBlockIdx();
        uint32_t elemPerBlock = BLOCK_SIZE / sizeof(T);
        this->maxBlockPerIter = maxBlockPerIter;

        uint32_t globalIndex = coreIdx * (blockPerCore + 1) * elemPerBlock; //全局偏移 单位是元素数量
        elemPerIter = elemPerBlock * maxBlockPerIter;                      //每次迭代最多处理的元素数量
        if (coreIdx < nAcores) {
        this->nElem = (blockPerCore + 1) * elemPerBlock;
        nIter = (blockPerCore + 1 + maxBlockPerIter - 1) / maxBlockPerIter;
        tailElem = ((blockPerCore + 1) % maxBlockPerIter) * elemPerBlock;
        } else {
        // printf("globalIndex now: %d\n", globalIndex);
        globalIndex = globalIndex - (coreIdx - nAcores) * elemPerBlock;
        this->nElem = blockPerCore * elemPerBlock;
        nIter = (blockPerCore + maxBlockPerIter - 1) / maxBlockPerIter;
        tailElem = (blockPerCore % maxBlockPerIter) * elemPerBlock;
        }
        // printf("globalIndex: %d blockPerCore:%d elemPerBlock:%d nAcores:%d\n",globalIndex,blockPerCore,elemPerBlock,nAcores);
        // printf("nElem:%d blockPerCore:%d elemPerIter:%d tailElem:%d nIter:%d\n", nElem, blockPerCore,elemPerIter, tailElem,nIter);
        curElem = elemPerIter;
        inputGm.SetGlobalBuffer((__gm__ T*)input + globalIndex, nElem);
        otherGm.SetGlobalBuffer((__gm__ T*)other + globalIndex, nElem);
        outGm.SetGlobalBuffer((__gm__ T*)out + globalIndex, nElem);
        pipe->InitBuffer(outQue, BUFFER_NUM, elemPerIter * sizeof(T));
        pipe->InitBuffer(otherQue, BUFFER_NUM, elemPerIter * sizeof(T));
        pipe->InitBuffer(inputQue, BUFFER_NUM, elemPerIter * sizeof(T));
    }
      __aicore__ inline void CopyIn(int i) {
          // printf("curElem:%d\n", curElem);
          auto inputLocal = inputQue.AllocTensor<T>();
          auto otherLocal = otherQue.AllocTensor<T>();
          DataCopy(inputLocal, inputGm[elemPerIter * i], curElem);
          DataCopy(otherLocal, otherGm[elemPerIter * i], curElem);
          inputQue.EnQue(inputLocal);
          otherQue.EnQue(otherLocal);
      }
      __aicore__ inline T gcd(T a, T b) {
        while (b != 0) {
          T tmp = a;
          a = b;
          b = tmp % b;
        }
        return a;
    }
    __aicore__ inline T fullgcd(T a, T b) {
        T absa = a < 0 ? -a : a;
        T absb = b < 0 ? -b : b;
    
        T g = gcd(absa, absb);
        T out = (absa / g) * absb;
    
        if constexpr (!std::is_same_v<T, std::int8_t>) {
            out = out < 0 ? -out : out;
        }
    
        return out;
    }
      __aicore__ inline void Compute(int iterIdx) {
          auto outLocal = outQue.AllocTensor<T>();
          auto inputLocal = inputQue.DeQue<T>();
          auto otherLocal = otherQue.DeQue<T>();
          int i = 0;
          for (i = 0; i < curElem; i += 4) {
              T x1 = inputLocal.GetValue(i);
              T x2 = otherLocal.GetValue(i);
              T x11 = inputLocal.GetValue(i+1);
              T x21 = otherLocal.GetValue(i+1);
                // printf("%d %d\n", x1, x2);
              T x12 = inputLocal.GetValue(i+2);
              T x22 = otherLocal.GetValue(i+2);
              T x13 = inputLocal.GetValue(i+3);
              T x23 = otherLocal.GetValue(i+3);
                
              T res = fullgcd(x1, x2);
              T res1 = fullgcd(x11, x21);
              T res2 = fullgcd(x12, x22);
              T res3 = fullgcd(x13, x23);
              outLocal.SetValue(i, res);
              outLocal.SetValue(i+1, res1);
              outLocal.SetValue(i+2, res2);
              outLocal.SetValue(i+3, res3);

            }
          for (; i < curElem; i++) {
            T x1 = inputLocal.GetValue(i);
            T x2 = otherLocal.GetValue(i);
            T res = fullgcd(x1, x2);
            outLocal.SetValue(i, res);
          }

          outQue.EnQue(outLocal);
          inputQue.FreeTensor(inputLocal);
          otherQue.FreeTensor(otherLocal);
      }
      __aicore__ inline void CopyOut(int iterIdx) {
          auto outLocal = outQue.DeQue<T>();
          DataCopy(outGm[elemPerIter * iterIdx], outLocal, curElem);
          outQue.FreeTensor(outLocal);
      }
      __aicore__ inline void Process() {
          for (int i = 0; i < nIter; i++) {
              if (i == nIter -1) {
                  curElem = tailElem;
              }
              CopyIn(i);
              Compute(i);
              CopyOut(i);
          }
      }
  private:
      GlobalTensor<T> inputGm, otherGm, outGm;
      TPipe* pipe;
      uint16_t maxBlockPerIter;
      uint32_t nElem, nIter, elemPerIter, tailElem, curElem;
      TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
      TQue<QuePosition::VECIN, BUFFER_NUM> inputQue;
      TQue<QuePosition::VECIN, BUFFER_NUM> otherQue;
      // TBuf<QuePosition::VECCALC> signQue;
      // LocalTensor<T>signLocal;
  
  };
#define LIMIT 16777216

template<typename T>class KernelLCMBcast64{
    public:
        __aicore__ inline KernelLCMBcast64(){}
        __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR output, uint64_t tail, uint64_t piter, uint64_t smallDataNum,uint64_t mid, uint64_t pre, TPipe* pipeIn){
            this->pipe = pipeIn;
            this->piter = piter;
            uint64_t bigDataNum = smallDataNum + 1; 
            uint64_t coreNum = AscendC::GetBlockIdx();
            globalIndex = bigDataNum * coreNum;
            if (coreNum < tail) {
                this->coreDataNum = bigDataNum;
            } else {
                this->coreDataNum = smallDataNum;
                globalIndex -= (bigDataNum - smallDataNum)*(coreNum - tail);
            }
            // this->globalIndex = globalIndex;
            // globalIndex *= (piter);
            uint64_t totalLength = coreDataNum*piter*mid*pre;
            this->mid = mid;
            this->pre = pre;
            piter256 = (piter + 63)/64*64;
            // printf("piter %d pre:%d mid:%d\n",piter,pre, mid);
            // printf("globalIndex:%d totalLength: %d coreDataNum:%d piter:%d\n",globalIndex, totalLength, coreDataNum,piter);
            inputGm.SetGlobalBuffer((__gm__ T*)input, pre*mid*piter);
            otherGm.SetGlobalBuffer((__gm__ T*)other, pre*piter);
            outputGm.SetGlobalBuffer((__gm__ T*)output, pre*mid*piter);
            pipe->InitBuffer(inQueueInput, BUFFER_NUM, 4*piter256*sizeof(T));
            pipe->InitBuffer(inQueueOther, BUFFER_NUM, 4*piter256*sizeof(T));
            pipe->InitBuffer(outQueue, BUFFER_NUM, 4*piter256*sizeof(T));

            pipe->InitBuffer(inputFloatQue, 4*piter256 * sizeof(float));
            pipe->InitBuffer(otherFloatQue, 4*piter256 * sizeof(float));
            pipe->InitBuffer(outFloatQue, 4*piter256 * sizeof(float));

            // pipe->InitBuffer(x1Que, piter256 * sizeof(float));
            // pipe->InitBuffer(x2Que, piter256 * sizeof(float));
            pipe->InitBuffer(modQue, 4*piter256 * sizeof(float));
            pipe->InitBuffer(maskQue, 4*piter256);
            maskLocal = maskQue.Get<uint16_t>();
            modLocal = modQue.Get<float>();
            inputFloatLocal = inputFloatQue.Get<float>();
            otherFloatLocal = otherFloatQue.Get<float>();
            outFloatLocal = outFloatQue.Get<float>();
            // x1Local = x1Que.Get<float>();
            // x2Local = x2Que.Get<float>();

        }
    
        __aicore__ inline T gcd(T a, T b) {
            while (b != 0) {
              T tmp = a;
              a = b;
              b = tmp % b;
            }
            return a;
        }
          __aicore__ inline void gcdto32(T &in, T &a, T &b) {
            in = in < 0? -in: in;
            a = a < 0 ? -a : a;
            b = b < 0 ? -b : b;
            T shift = ScalarGetSFFValue<1>(a | b);

            in = in >> shift;

            b >>= ScalarGetSFFValue<1>(b);
            while (1) {
              a >>= ScalarGetSFFValue<1>(a);
              if (b > a) {
                a ^= b ^= a ^= b;
              }
              if (b == 0) {
                in = in / a;
                a = 1;
                return;
              }
              if (a < LIMIT) {
                return;
              }
              a -= b;
            };
          }
    
          // __aicore__ inline T gcdto32(T& a, T& b) {
          //   a = a < 0 ? -a : a;
          //   b = b < 0 ? -b : b;
          //   T tmp = a;
          //   while (b != 0) {
          //       tmp = a;
          //       a = b;
          //       b = tmp % b;
          //       // printf("%d %d\n", a, b);
          //       if (a < LIMIT) break;
          //   }
          //   return a;
          //   // return out;
          // }
        __aicore__ inline void process() {
            for (int i = 0; i < pre; i++) {
                otherLocal = inQueueOther.AllocTensor<T>();
                AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位 
                AscendC::DataCopyPad(otherLocal, otherGm[i*piter], copyParams, padParams); // 从GM-
                inQueueOther.EnQue(otherLocal);
                otherLocal = inQueueOther.DeQue<T>();
                int adder = 4;
                for (int z = globalIndex; z <globalIndex + coreDataNum; z+=adder) {
                    adder = min(4, int(globalIndex + coreDataNum - z));
                    inLocal = inQueueInput.AllocTensor<T>();
                    outLocal =  outQueue.AllocTensor<T>();
                    AscendC::DataCopyExtParams copyBatchParams{1, static_cast<uint32_t>(adder*this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位 
                    AscendC::DataCopyPad(inLocal, inputGm[(i*mid)*piter + z*piter], copyBatchParams, padParams); // 从GM-
                    inQueueInput.EnQue(inLocal);
                    inLocal = inQueueInput.DeQue<T>();
                    int j = 0;
                    for (; j < piter; j+=4) {
                        T x20ref = otherLocal.GetValue(j);
                        T x21ref = otherLocal.GetValue(j+1);
                        T x22ref = otherLocal.GetValue(j+2);
                        T x23ref = otherLocal.GetValue(j+3);
                        for (int batch = 0; batch < adder; batch++) {
                            int base = batch*piter;
                            T x00 = inLocal.GetValue(base+j);
                            T x01 = inLocal.GetValue(base+j+1);
                            T x02 = inLocal.GetValue(base+j+2);
                            T x03 = inLocal.GetValue(base+j+3);

                            T x10 = x00;
                            T x11 = x01;
                            T x12 = x02;
                            T x13 = x03;

                            T x20 = x20ref;
                            T x21 = x21ref;
                            T x22 = x22ref;
                            T x23 = x23ref;
                            // T res = 
                            gcdto32(x00, x10, x20);
                            gcdto32(x01, x11, x21);
                            gcdto32(x02, x12, x22);
                            gcdto32(x03, x13, x23);

                            inLocal.SetValue(base+j, x00);
                            inLocal.SetValue(base+j+1, x01);
                            inLocal.SetValue(base+j+2, x02);
                            inLocal.SetValue(base+j+3, x03);

                            inputFloatLocal.SetValue(base+j, static_cast<float>(x10));
                            inputFloatLocal.SetValue(base+j+1, static_cast<float>(x11));
                            inputFloatLocal.SetValue(base+j+2, static_cast<float>(x12));
                            inputFloatLocal.SetValue(base+j+3, static_cast<float>(x13));

                            otherFloatLocal.SetValue(base+j, static_cast<float>(x20));
                            otherFloatLocal.SetValue(base+j+1, static_cast<float>(x21));
                            otherFloatLocal.SetValue(base+j+2, static_cast<float>(x22));
                            otherFloatLocal.SetValue(base+j+3, static_cast<float>(x23));
                        }
                    }
                    for (; j < piter; j++) {
                        T x20ref = otherLocal.GetValue(j);
                        for (int batch = 0; batch < adder; batch++) {
                            int base = batch*piter;
                            T x00 = inLocal.GetValue(base+j);
                            T x10 = x00;
                            T x20 = x20ref;
                            // T res = 
                            gcdto32(x00, x10, x20);
                            inLocal.SetValue(base+j, x00);
                            inputFloatLocal.SetValue(base+j, static_cast<float>(x10));
                            otherFloatLocal.SetValue(base+j, static_cast<float>(x20));
                        // outLocal.SetValue(j, res);
                        }
                    }
                    for (int j = 0; j < 32; j++) {
                        CompareScalar(maskLocal, otherFloatLocal, static_cast<float>(0.0), AscendC::CMPMODE::EQ, adder*piter256);
                        Select(outFloatLocal, maskLocal, inputFloatLocal, outFloatLocal, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, adder*piter256);
                        Fmod(modLocal, inputFloatLocal, otherFloatLocal,adder*piter256);
                        tempLocal = inputFloatLocal;
                        inputFloatLocal = otherFloatLocal;
                        otherFloatLocal = modLocal;
                        modLocal = tempLocal;
                    }
                    for (j = 0; j < piter; j += 4) {
                        for (int batch = 0; batch < adder; batch++) {
                            int base = batch*piter;
                            T gcd = static_cast<T>(outFloatLocal.GetValue(base+j));
                            T gcd1 = static_cast<T>(outFloatLocal.GetValue(base+j+1));
                            T gcd2 = static_cast<T>(outFloatLocal.GetValue(base+j+2));
                            T gcd3 = static_cast<T>(outFloatLocal.GetValue(base+j+3));

                            T out = (inLocal.GetValue(base+j) / gcd) * otherLocal.GetValue(j);
                            T out1 = (inLocal.GetValue(base+j+1) / gcd1) * otherLocal.GetValue(j+1);
                            T out2 = (inLocal.GetValue(base+j+2) / gcd2) * otherLocal.GetValue(j+2);
                            T out3 = (inLocal.GetValue(base+j+3) / gcd3) * otherLocal.GetValue(j+3);

                            out = out < 0 ? -out : out;
                            out1 = out1 < 0 ? -out1 : out1;
                            out2 = out2 < 0 ? -out2 : out2;
                            out3 = out3 < 0 ? -out3 : out3;

                            outLocal.SetValue(base+j, out);
                            outLocal.SetValue(base+j+1, out1);
                            outLocal.SetValue(base+j+2, out2);
                            outLocal.SetValue(base+j+3, out3);
                        }
                    }
                    for (; j < piter; j++) {
                        for (int batch = 0; batch < adder; batch++) {
                            int base = batch*piter;
                            T gcd = static_cast<T>(outFloatLocal.GetValue(base+j));
                            T out = (inLocal.GetValue(base+j) / gcd) * otherLocal.GetValue(j);
                            out = out < 0 ? -out : out;
                            outLocal.SetValue(base+j, out);
                        }
                    }
                    outQueue.EnQue(outLocal);

                    outLocal = outQueue.DeQue<T>();
                    AscendC::DataCopyPad(outputGm[(i*mid)*piter + z*piter],outLocal, copyBatchParams);
                    outQueue.FreeTensor(outLocal);
                    inQueueInput.FreeTensor(inLocal);
                }
                inQueueOther.FreeTensor(otherLocal);
            }
        }
        __aicore__ inline void Process() {
            // int32_t loopCount = this->tileNum;
            midIndx = globalIndex/mid;
            // printf("midIndx: %d\n", midIndx);
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位 
            otherLocal = inQueueOther.AllocTensor<T>();
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(otherLocal, otherGm[midIndx*piter], copyParams, padParams); // 从GM-
            inQueueOther.EnQue(otherLocal);
            otherLocal = inQueueOther.DeQue<T>();
            // Abs(otherLocal, otherLocal, piter);

            for (int i = globalIndex; i <globalIndex + coreDataNum; i++) {
                if (i/mid != midIndx) {
                    midIndx = i/mid;
                    // printf("midIndx: %d\n", midIndx);
                    AscendC::DataCopyPad(otherLocal, otherGm[midIndx*piter], copyParams, padParams); // 从GM-
                    inQueueOther.EnQue(otherLocal);
                    otherLocal = inQueueOther.DeQue<T>();
                    // Abs(otherLocal, otherLocal, piter);
                } 
                inLocal = inQueueInput.AllocTensor<T>();
                outLocal =  outQueue.AllocTensor<T>();
                AscendC::DataCopyPad(inLocal, inputGm[(i-globalIndex)*piter], copyParams, padParams); // 从GM-
                inQueueInput.EnQue(inLocal);
                inLocal = inQueueInput.DeQue<T>();
                // Abs(inLocal, inLocal, piter);
                int j = 0;
                for (; j < piter; j+=4) {
                    T x00 = inLocal.GetValue(j);
                    T x01 = inLocal.GetValue(j+1);
                    T x02 = inLocal.GetValue(j+2);
                    T x03 = inLocal.GetValue(j+3);

                    T x10 = x00;
                    T x11 = x01;
                    T x12 = x02;
                    T x13 = x03;

                    T x20 = otherLocal.GetValue(j);
                    T x21 = otherLocal.GetValue(j+1);
                    T x22 = otherLocal.GetValue(j+2);
                    T x23 = otherLocal.GetValue(j+3);
                    // T res = 
                    gcdto32(x00, x10, x20);
                    gcdto32(x01, x11, x21);
                    gcdto32(x02, x12, x22);
                    gcdto32(x03, x13, x23);

                    inLocal.SetValue(j, x00);
                    inLocal.SetValue(j+1, x01);
                    inLocal.SetValue(j+2, x02);
                    inLocal.SetValue(j+3, x03);

                    inputFloatLocal.SetValue(j, static_cast<float>(x10));
                    inputFloatLocal.SetValue(j+1, static_cast<float>(x11));
                    inputFloatLocal.SetValue(j+2, static_cast<float>(x12));
                    inputFloatLocal.SetValue(j+3, static_cast<float>(x13));

                    otherFloatLocal.SetValue(j, static_cast<float>(x20));
                    otherFloatLocal.SetValue(j+1, static_cast<float>(x21));
                    otherFloatLocal.SetValue(j+2, static_cast<float>(x22));
                    otherFloatLocal.SetValue(j+3, static_cast<float>(x23));

                }
                for (; j < piter; j++) {
                    T x0 = inLocal.GetValue(j);
                    T x1 = inLocal.GetValue(j);
                    T x2 = otherLocal.GetValue(j);
                    // T res = 
                    gcdto32(x0, x1, x2);
                    inLocal.SetValue(j, x0);
                    inputFloatLocal.SetValue(j, static_cast<float>(x1));
                    otherFloatLocal.SetValue(j, static_cast<float>(x2));
                    // outLocal.SetValue(j, res);
                }
                // printf("inputFloatLocal\n");
                // for (int j = 0; j < piter; j++) {
                //     printf("%f ", inputFloatLocal.GetValue(j));
                // }
                // printf("\n");
                // printf("otherFloatLocal\n");
                // for (int j = 0; j < piter; j++) {
                //     printf("%f ", otherFloatLocal.GetValue(j));
                // }
                // printf("\n");
                for (int j = 0; j < 32; j++) {
                    // printf("otherFloatLocal\n");
                    // for (int z = 0; z < piter; z++) {
                    //     printf("%f ", otherFloatLocal.GetValue(z));
                    // }
                    // printf("\n");
                    // b等于0，mask置为1,表示需要从计算结果中选择
                    CompareScalar(maskLocal, otherFloatLocal, static_cast<float>(0.0), AscendC::CMPMODE::EQ, piter256);
                    // printf("mask\n");
                    // for (int z = 0; z < 2; z++) {
                    //     printf("%X ", maskLocal.GetValue(z));
                    // }
                    // printf("\n");
                    Select(outFloatLocal, maskLocal, inputFloatLocal, outFloatLocal, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, piter256);
                    // printf("outFloatLocal\n");
                    // for (int z = 0; z < piter; z++) {
                    //     printf("%f ", outFloatLocal.GetValue(z));
                    // }
                    // printf("\n");
                    Fmod(modLocal, inputFloatLocal, otherFloatLocal, piter256);
//                     printf("modLocal\n");
//                     for (int z = 0; z < piter; z++) {
//                         printf("%f ", modLocal.GetValue(z));
//                     }
//                     printf("\n");

//                     printf("inputFloatLocal\n");
                    // for (int z = 0; z < piter; z++) {
                    //     printf("%f ", inputFloatLocal.GetValue(z));
                    // }
                    // printf("\n");

                    tempLocal = inputFloatLocal;
                    inputFloatLocal = otherFloatLocal;
                    otherFloatLocal = modLocal;
                    // printf("otherFloatLocal\n");
                    // for (int z = 0; z < piter; z++) {
                    //     printf("%f ", otherFloatLocal.GetValue(z));
                    // }
                    // printf("\n");
                    // printf("\n");

                    modLocal = tempLocal;
                }
                // printf("outFloatLocal\n");
                // for (int j = 0; j < piter; j++) {
                //     printf("%f ", outFloatLocal.GetValue(j));
                // }
                // printf("\n");
                // int j;
                for (j = 0; j < piter; j += 4) {
                    T gcd = static_cast<T>(outFloatLocal.GetValue(j));
                    T gcd1 = static_cast<T>(outFloatLocal.GetValue(j+1));
                    T gcd2 = static_cast<T>(outFloatLocal.GetValue(j+2));
                    T gcd3 = static_cast<T>(outFloatLocal.GetValue(j+3));

                    T out = (inLocal.GetValue(j) / gcd) * otherLocal.GetValue(j);
                    T out1 = (inLocal.GetValue(j+1) / gcd1) * otherLocal.GetValue(j+1);
                    T out2 = (inLocal.GetValue(j+2) / gcd2) * otherLocal.GetValue(j+2);
                    T out3 = (inLocal.GetValue(j+3) / gcd3) * otherLocal.GetValue(j+3);

                    out = out < 0 ? -out : out;
                    out1 = out1 < 0 ? -out1 : out1;
                    out2 = out2 < 0 ? -out2 : out2;
                    out3 = out3 < 0 ? -out3 : out3;

                    outLocal.SetValue(j, out);
                    outLocal.SetValue(j+1, out1);
                    outLocal.SetValue(j+2, out2);
                    outLocal.SetValue(j+3, out3);

                }
                for (; j < piter; j++) {
                    T gcd = static_cast<T>(outFloatLocal.GetValue(j));
                    T out = (inLocal.GetValue(j) / gcd) * otherLocal.GetValue(j);
                    out = out < 0 ? -out : out;
                    outLocal.SetValue(j, out);
                }
                outQueue.EnQue(outLocal);
                
                outLocal = outQueue.DeQue<T>();
                AscendC::DataCopyPad(outputGm[(i-globalIndex)*piter],outLocal, copyParams);
                outQueue.FreeTensor(outLocal);
                inQueueInput.FreeTensor(inLocal);
            }
        }
    private:
        uint32_t totalLength,resLength,stride,iterStep, axesDim,coreBatch,mid, pre, piter256;
        uint32_t tileDataNum, processDataNum,tailDataNum ,coreDataNum,tileNum, midIndx;
        uint32_t padTimes, globalIndex, piter;
        AscendC::TPipe *pipe;
        AscendC::GlobalTensor<T> inputGm;
        AscendC::GlobalTensor<T> otherGm;
        AscendC::GlobalTensor<T> outputGm;
        TQue<QuePosition::VECIN, BUFFER_NUM>inQueueInput, inQueueOther;
        TQue<QuePosition::VECOUT, BUFFER_NUM>outQueue; 
        LocalTensor<T> inLocal, otherLocal, outLocal;
        LocalTensor<float>inputFloatLocal, otherFloatLocal, x1Local, x2Local, tempLocal, outFloatLocal, modLocal;
        LocalTensor<uint16_t>maskLocal;
        TBuf<QuePosition::VECCALC> inputFloatQue, otherFloatQue, outFloatQue, x1Que, x2Que, maskQue, modQue;

        TBuf<AscendC::TPosition::VECCALC>inputFloatBuf;
};

extern "C" __global__ __aicore__ void lcm(GM_ADDR input, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;

    if (TILING_KEY_IS(0)) { //不需要广播
        // printf("key1\n");
        if (std::is_same_v<DTYPE_INPUT, int32_t>) {
            KernelLCM<DTYPE_INPUT> op;
            op.Init(input, other, out, tiling_data.blockPerCore, tiling_data.nAcores, tiling_data.nBcores, tiling_data.maxBlockPerIter, &pipe);
            op.Process();
        }
    } else if (TILING_KEY_IS(1)) {
        KernelLCMint64<DTYPE_INPUT> op;
        op.Init(input, other, out, tiling_data.blockPerCore, tiling_data.nAcores, tiling_data.nBcores, tiling_data.maxBlockPerIter, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelLCMBcast64<DTYPE_INPUT> op;
        op.Init(input, other, out, tiling_data.nAcores, tiling_data.last, tiling_data.smallBatch, tiling_data.mid, tiling_data.pre, &pipe);
        op.process();
    }
}