#include "kernel_operator.h"

const int BLOCK_SIZE = 32;
const int MAX_DIM = 5;
#define BUFFER_NUM 1
using namespace AscendC;
template <typename T> class CopySign {
  public:
    __aicore__ inline CopySign() {}
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
        // pipe->InitBuffer(signQue, elemPerIter * sizeof(T));
        // signLocal = signQue.Get<T>();
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
    __aicore__ inline void Compute(int iterIdx) {
        auto outLocal = outQue.AllocTensor<T>();
        auto inputLocal = inputQue.DeQue<T>();
        auto otherLocal = otherQue.DeQue<T>();

        // printf("inputLocal\n");
        // for (int i = 0; i < 8; i++) {
        //     printf("%x ", inputLocal.GetValue(i));
        // }
        // printf("\n");
        if constexpr(std::is_same_v<DTYPE_INPUT, float>) {
            Abs(inputLocal, inputLocal, curElem);
            AscendC::LocalTensor<int32_t> input32Local = inputLocal.template  ReinterpretCast<int32_t>();
            AscendC::LocalTensor<int32_t> other32Local = otherLocal.template ReinterpretCast<int32_t>();
            AscendC::LocalTensor<int32_t> out32Local = outLocal.template ReinterpretCast<int32_t>();
            int32_t scalar = 31;
                uint32_t uscalar = 31;

            AscendC::ShiftRight(other32Local, other32Local, scalar, curElem);
            AscendC::ShiftLeft(other32Local, other32Local, scalar, curElem);
            AscendC::Or(out32Local, other32Local, input32Local, curElem*2);
        } else {
            Abs(inputLocal, inputLocal, curElem);
            AscendC::LocalTensor<int16_t> input32Local = inputLocal.template ReinterpretCast<int16_t>();
            AscendC::LocalTensor<int16_t> other32Local = otherLocal.template ReinterpretCast<int16_t>();
            AscendC::LocalTensor<int16_t> out32Local = outLocal.template ReinterpretCast<int16_t>();
            int16_t scalar = 15;
            // uint16_t uscalar = 15;
            // printf("otherLocal");
            // for (int i = 0; i < 8; i++) {
            //     printf("%f ", otherLocal.GetValue(i));
            // }
            // printf("\n");
            AscendC::ShiftRight(other32Local, other32Local, scalar, curElem);
            // printf("now");
            // for (int i = 0; i < 8; i++) {
            //     printf("%x ", other32Local.GetValue(i));
            // }
            // printf("\n");
            AscendC::ShiftLeft(other32Local, other32Local, scalar, curElem);
            // printf("now1");
            // for (int i = 0; i < 8; i++) {
            //     printf("%x ", other32Local.GetValue(i));
            // }
            // printf("\n");
            // for (int i = 0; i < 8; i++) {
            //     printf("%x ", other32iLocal.GetValue(i));
            // }
            // printf("\n");
            AscendC::Or(out32Local, other32Local, input32Local, curElem);
        }
//         int16_t scalar = 31;
//         AscendC::ShiftLeft(otherLocal, otherLocal, scalar, curElem);
//         AscendC::ShiftRight(otherLocal, otherLocal, scalar, curElem);
//         AscendC::Or(dstLocal, src0Local, src1Local, 512);

//         Abs(signLocal, otherLocal, curElem);
//         Abs(inputLocal, inputLocal, curElem);
//         Div(signLocal, signLocal, otherLocal, curElem);
//         Mul(outLocal, signLocal, inputLocal, curElem);
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

template<typename T> class KernelCopySignFloat{
    public:
        __aicore__ inline KernelCopySignFloat(){}
        __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR output, uint64_t tail, uint64_t piter, uint64_t smallDataNum,uint64_t mid, uint64_t pre, TPipe* pipeIn){
            // this->pipe = pipe;
            this->piter = piter;
            uint64_t bigDataNum = smallDataNum + 1; 
            uint64_t coreNum = AscendC::GetBlockIdx();
            uint64_t globalIndex = bigDataNum * coreNum;
            if (coreNum < tail) {
                this->coreDataNum = bigDataNum;
            } else {
                this->coreDataNum = smallDataNum;
                globalIndex -= (bigDataNum - smallDataNum)*(coreNum - tail);
            }
            globalIndex *= (piter*mid*pre);
            uint64_t totalLength = coreDataNum*piter*mid*pre;
            this->mid = mid;
            this->pre = pre;
            // printf("globalIndex:%d totalLength: %d coreDataNum:%d piter:%d\n",globalIndex, totalLength, coreDataNum,piter);
            inputGm.SetGlobalBuffer((__gm__ T*)input + globalIndex, totalLength);
            otherGm.SetGlobalBuffer((__gm__ T*)other, piter*pre);
            outputGm.SetGlobalBuffer((__gm__ T*)output + globalIndex, totalLength);
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
        __aicore__ void copyIn(uint64_t progress, int sub) {
            LocalTensor<T>inputLocal = inQueueInput.AllocTensor<T>();
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(inputLocal, inputGm[progress*pre*mid*piter + sub*piter], copyParams, padParams); // 从GM-
            inQueueInput.EnQue<T>(inputLocal);
        }
        __aicore__ void copyOther(uint64_t progress) {
            LocalTensor<T>otherLocal = inQueueOther.AllocTensor<T>();
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(otherLocal, otherGm[progress*piter], copyParams, padParams); // 从GM-
            inQueueOther.EnQue<T>(otherLocal);
        }
        __aicore__ void compute(uint64_t progress, int sub) {
            // LocalTensor<T>outputLocal = outQueue.AllocTensor<T>();
            // LocalTensor<T>inputLocal = inQueueInput.AllocTensor<T>();
            LocalTensor<T>otherLocal = inQueueOther.DeQue<T>();
            if constexpr(std::is_same_v<DTYPE_INPUT, float>) {
                AscendC::LocalTensor<int32_t> other32Local = otherLocal.template ReinterpretCast<int32_t>();
                int32_t scalar = 31;
                AscendC::ShiftRight(other32Local, other32Local, scalar, piter);
                AscendC::ShiftLeft(other32Local, other32Local, scalar, piter);
                // AscendC::Or(out32Local, other32Local, input32Local, curElem*2);
            } else {
                AscendC::LocalTensor<int16_t> other32Local = otherLocal.template ReinterpretCast<int16_t>();
                int16_t scalar = 15;
                AscendC::ShiftRight(other32Local, other32Local, scalar, piter);
                AscendC::ShiftLeft(other32Local, other32Local, scalar, piter);
            }
            // Abs(signLocal, otherLocal, piter);
            // Div(signLocal, signLocal, otherLocal, piter);
            for (int i = 0; i < mid; i++) {
                LocalTensor<T>outputLocal = outQueue.AllocTensor<T>();
                LocalTensor<T>inputLocal = inQueueInput.AllocTensor<T>();
                AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
                AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                AscendC::DataCopyPad(inputLocal, inputGm[progress*pre*mid*piter + (sub*mid+i)*piter], copyParams, padParams); // 从GM-
                inQueueInput.EnQue<T>(inputLocal);
                inputLocal = inQueueInput.DeQue<T>();
                // printf("inputLocal\n");
                // for (int i = 0; i < 8; i++) {
                //    inputLocal.GetValue(i);
                // }
                // printf("\n");
                // inputLocal.GetValue(0);
                // AscendC::TQueSync<PIPE_MTE2, PIPE_V> sync;
                // sync.SetFlag(0);
                // sync.WaitFlag(0);
                AscendC::Abs(inputLocal, inputLocal, piter);
                if constexpr(std::is_same_v<DTYPE_INPUT, float>) {
                    AscendC::LocalTensor<int32_t> other32Local = otherLocal.template ReinterpretCast<int32_t>();
                    AscendC::LocalTensor<int32_t> input32Local = inputLocal.template  ReinterpretCast<int32_t>();
                    AscendC::LocalTensor<int32_t> out32Local = outputLocal.template ReinterpretCast<int32_t>();

                    AscendC::Or(out32Local, other32Local, input32Local, piter*2);
                } else {
                    AscendC::LocalTensor<int16_t> other32Local = otherLocal.template ReinterpretCast<int16_t>();
                    AscendC::LocalTensor<int16_t> input32Local = inputLocal.template  ReinterpretCast<int16_t>();
                    AscendC::LocalTensor<int16_t> out32Local = outputLocal.template ReinterpretCast<int16_t>();

                    AscendC::Or(out32Local, other32Local, input32Local, piter);
                }
                // AscendC::TQueSync<PIPE_V,PIPE_MTE3> sync1;
                // sync1.SetFlag(0);
                // sync1.WaitFlag(0);
                outQueue.EnQue<T>(outputLocal);
                outputLocal = outQueue.DeQue<T>();
                // AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
                // AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                AscendC::DataCopyPad(outputGm[progress*pre*mid*piter + (sub*mid+i)*piter],outputLocal, copyParams);
                outQueue.FreeTensor(outputLocal);
                inQueueInput.FreeTensor(inputLocal);

            }
            inQueueOther.FreeTensor(otherLocal);


        }
        // __aicore__ void copyOut(uint64_t progress, int sub) {
        //     LocalTensor<T>outputLocal = outQueue.DeQue<T>();
        //     AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        //     AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        //     AscendC::DataCopyPad( outputGm[progress*pre*mid*piter + sub*piter],outputLocal, copyParams);
        //     outQueue.FreeTensor(outputLocal);
        // }
        __aicore__ inline void Process() {
            for (int i = 0; i < coreDataNum; i++) {
                for (int j = 0; j < pre;j++) {
                    copyOther(j);
                    compute(i, j);
                    // copyIn(i, j);
                    // compute(i, j);
                    // copyOut(i, j);
                }
            }
        }
        
        
    private:
        // AscendC::TPipe* pipe;
        uint64_t piter,coreDataNum,mid, pre;
        AscendC::GlobalTensor<T> inputGm, otherGm, outputGm;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInput,inQueueOther;
        TBuf<QuePosition::VECCALC> sign;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
        LocalTensor<T> signLocal;
    
};


template<typename T> class KernelCopySignNoPre{
    public:
        __aicore__ inline KernelCopySignNoPre(){}
        __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR output, uint64_t tail, uint64_t piter, uint64_t smallDataNum,uint64_t mid, uint64_t pre, TPipe* pipeIn){
            // this->pipe = pipe;
            this->piter = piter;
            uint64_t bigDataNum = smallDataNum + 1; 
            uint64_t coreNum = AscendC::GetBlockIdx();
            uint64_t globalIndex = bigDataNum * coreNum;
            if (coreNum < tail) {
                this->coreDataNum = bigDataNum;
            } else {
                this->coreDataNum = smallDataNum;
                globalIndex -= (bigDataNum - smallDataNum)*(coreNum - tail);
            }
            this->globalIndex = globalIndex;
            uint64_t totalLength = coreDataNum*piter*mid;
            this->mid = mid;
            this->pre = pre;
            uint64_t fullLength = piter*mid*pre;
            // printf("globalIndex:%d pre: %d mid: %d coreDataNum:%d piter:%d\n",globalIndex,pre, mid, coreDataNum,piter);
            // inputGm.SetGlobalBuffer((__gm__ T*)input + globalIndex, totalLength);
            // otherGm.SetGlobalBuffer((__gm__ T*)other, piter*pre);
            inputGm.SetGlobalBuffer((__gm__ T*)input , pre*mid*piter);
            otherGm.SetGlobalBuffer((__gm__ T*)other , pre*piter);
            outputGm.SetGlobalBuffer((__gm__ T*)output, pre*mid*piter);
            pipeIn->InitBuffer(inQueueInput,1, piter*sizeof(T));
            pipeIn->InitBuffer(inQueueOther,1, piter*sizeof(T));
            pipeIn->InitBuffer(outQueue,1, piter*sizeof(T));
            // this->inLocal = inQueueInput.Get<T>();
            // this->otherLocal = inQueueOther.Get<T>();
            // this->outLocal = outQueue.Get<T>();
        }
//         __aicore__ void CopyIn(int i) {
            
//         }
        __aicore__ void Process() {
            int midIndx = globalIndex/mid;
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->piter*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位 
            otherLocal = inQueueOther.AllocTensor<T>();
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(otherLocal, otherGm[midIndx*piter], copyParams, padParams); // 从GM-
            inQueueOther.EnQue(otherLocal);
            otherLocal = inQueueOther.DeQue<T>();
                if constexpr(std::is_same_v<DTYPE_INPUT, float>) {
                    AscendC::LocalTensor<uint32_t> other32Local = otherLocal.template ReinterpretCast<uint32_t>();
                    uint32_t scalar = 31;
                    AscendC::ShiftRight(other32Local, other32Local, scalar, piter);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::ShiftLeft(other32Local, other32Local, scalar, piter);
                } else {
                    AscendC::LocalTensor<int16_t> other32Local = otherLocal.template ReinterpretCast<int16_t>();
                    int16_t scalar = 15;
                    AscendC::ShiftRight(other32Local, other32Local, scalar, piter);
                    AscendC::ShiftLeft(other32Local, other32Local, scalar, piter);
                }
            for (int i = globalIndex; i < globalIndex + coreDataNum; i++) {
                if (i/mid != midIndx) {
                    midIndx = i/mid;
                    AscendC::DataCopyPad(otherLocal, otherGm[midIndx*piter], copyParams, padParams); // 从GM-
                    inQueueOther.EnQue(otherLocal);
                    otherLocal = inQueueOther.DeQue<T>();
                    if constexpr(std::is_same_v<DTYPE_INPUT, float>) {
                        AscendC::LocalTensor<uint32_t> other32Local = otherLocal.template ReinterpretCast<uint32_t>();
                        uint32_t scalar = 31;
                        AscendC::ShiftRight(other32Local, other32Local, scalar, piter);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::ShiftLeft(other32Local, other32Local, scalar, piter);
                    } else {
                        AscendC::LocalTensor<int16_t> other32Local = otherLocal.template ReinterpretCast<int16_t>();
                        int16_t scalar = 15;
                        AscendC::ShiftRight(other32Local, other32Local, scalar, piter);
                        AscendC::ShiftLeft(other32Local, other32Local, scalar, piter);
                    }

                }
                // printf("i: %d otherLocal:%f\n", i, otherLocal.GetValue(0));
                inLocal = inQueueInput.AllocTensor<T>();
                outLocal =  outQueue.AllocTensor<T>();
                AscendC::DataCopyPad(inLocal, inputGm[i*piter], copyParams, padParams); // 从GM-
                inQueueInput.EnQue(inLocal);
                inLocal = inQueueInput.DeQue<T>();
                // AscendC::TQueSync<PIPE_MTE2, PIPE_V> sync;
                // sync.SetFlag(0);
                // sync.WaitFlag(0);
                AscendC::Abs(inLocal, inLocal, piter);
                 // inLocal.GetValue(0);
                AscendC::PipeBarrier<PIPE_V>();
                // printf("inLocal: %f\n",
                // for (int i = 0; i < 8; i++) {
                //     printf("%f ", inLocal.GetValue(i));
                // }
                // printf("\n");
                 // inLocal.GetValue(0);

                if constexpr(std::is_same_v<DTYPE_INPUT, float>) {
                    AscendC::LocalTensor<int32_t> other32Local = otherLocal.template ReinterpretCast<int32_t>();
                    AscendC::LocalTensor<int32_t> input32Local = inLocal.template  ReinterpretCast<int32_t>();
                    AscendC::LocalTensor<int32_t> out32Local = outLocal.template ReinterpretCast<int32_t>();

                    AscendC::Or(out32Local, other32Local, input32Local, piter*2);
                } else {
                    AscendC::LocalTensor<int16_t> other32Local = otherLocal.template ReinterpretCast<int16_t>();
                    AscendC::LocalTensor<int16_t> input32Local = inLocal.template  ReinterpretCast<int16_t>();
                    AscendC::LocalTensor<int16_t> out32Local = outLocal.template ReinterpretCast<int16_t>();

                    AscendC::Or(out32Local, other32Local, input32Local, piter);
                }
                // AscendC::TQueSync<PIPE_V, PIPE_MTE3> sync1;
                // sync1.SetFlag(0);
                // sync1.WaitFlag(0);
                outQueue.EnQue(outLocal);
                
                outLocal = outQueue.DeQue<T>();
                AscendC::DataCopyPad(outputGm[i*piter],outLocal, copyParams);
                outQueue.FreeTensor(outLocal);
                inQueueInput.FreeTensor(inLocal);

            }
            
        }
        
    private:
        // AscendC::TPipe* pipe;
        uint64_t piter,coreDataNum,mid, pre,globalIndex;
        AscendC::GlobalTensor<T> inputGm, otherGm, outputGm;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInput,inQueueOther;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
        LocalTensor<T> inLocal, otherLocal, outLocal;
    
};


extern "C" __global__ __aicore__ void copysign(GM_ADDR input, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    TPipe pipe;

    if (TILING_KEY_IS(0)) { //不需要广播
        // printf("key1\n");
        if constexpr(std::is_same_v<DTYPE_INPUT, float>) {
            CopySign<DTYPE_INPUT> op;
            op.Init(input, other, out, tiling_data.blockPerCore, tiling_data.nAcores, tiling_data.nBcores, tiling_data.maxBlockPerIter, &pipe);
            op.Process();
        } else {
            CopySign<float16_t> op;
            op.Init(input, other, out, tiling_data.blockPerCore, tiling_data.nAcores, tiling_data.nBcores, tiling_data.maxBlockPerIter, &pipe);
            op.Process();
        }
    } else if (TILING_KEY_IS(1)) {
        if constexpr(std::is_same_v<DTYPE_INPUT, float>) {
            KernelCopySignFloat<DTYPE_INPUT> op;
            op.Init(input, other, out, tiling_data.nAcores, tiling_data.last, tiling_data.smallBatch, tiling_data.mid, tiling_data.pre, &pipe);
            op.Process();
        } else {
            KernelCopySignFloat<float16_t> op;
            op.Init(input, other, out, tiling_data.nAcores,tiling_data.last, tiling_data.smallBatch, tiling_data.mid, tiling_data.pre, &pipe);
            op.Process();
        }
    } else if (TILING_KEY_IS(2)) {
        if constexpr(std::is_same_v<DTYPE_INPUT, float>) {
            KernelCopySignNoPre<DTYPE_INPUT> op;
            op.Init(input, other, out, tiling_data.nAcores, tiling_data.last, tiling_data.smallBatch, tiling_data.mid, tiling_data.pre, &pipe);
            op.Process();
        } else {
            KernelCopySignNoPre<float16_t> op;
            op.Init(input, other, out, tiling_data.nAcores,tiling_data.last, tiling_data.smallBatch, tiling_data.mid, tiling_data.pre, &pipe);
            op.Process();
        }
    }
}