#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1; // tensor num for each queue
constexpr int32_t BUFFER_NUM_16 = 1; // tensor num for each queue

template<typename T>class NoBcastFastFloat{
    public:
        __aicore__ inline NoBcastFastFloat(){}
        __aicore__ inline void Init(GM_ADDR glu_out, GM_ADDR input,GM_ADDR v,GM_ADDR jvp_out,
                                    uint32_t finalTileNum, uint32_t tileDataNum, uint32_t tailDataNum,
                                    uint32_t iterStep, uint32_t smallBatch,uint32_t tail, uint32_t padTimes,
                                    TPipe* pipeIn) {
            // printf("axexDim: %d stride:%d iterStep:%d \n", axesdim, stride, iterStep);
            // int round = 32/sizeof(T);
            // this->axesDim = (axesdim + round - 1)/round*round;
            this->padTimes = padTimes;
            this->pipe = pipeIn;
            this->iterStep = iterStep;
            this->tileNum = finalTileNum;
            // tileDataNum = 32;
            this->tileDataNum = tileDataNum;
            this->tailDataNum = tailDataNum;
            // this->stride = stride;
            uint32_t coreNum = AscendC::GetBlockIdx();
            uint32_t bigBatch = smallBatch + 1;
            uint32_t globalIndex = bigBatch * coreNum;
            if (coreNum < tail) {
                this->coreBatch = bigBatch;
            } else {
                this->coreBatch = smallBatch;
                globalIndex -= (bigBatch - smallBatch)*(coreNum - tail);
            }
            // printf("coreBatch:%d iterStep:%d tileDataNum:%d tailDataNum:%d\n",coreBatch, iterStep,tileDataNum,tailDataNum);
            gluOutGm.SetGlobalBuffer((__gm__ T*)glu_out + globalIndex*iterStep/2, this->coreBatch*iterStep/2);
            inputGm.SetGlobalBuffer((__gm__ T*)input + globalIndex*iterStep, this->coreBatch*iterStep);
            vGm.SetGlobalBuffer((__gm__ T*)v + globalIndex*iterStep, this->coreBatch*iterStep);
            jvpOutGm.SetGlobalBuffer((__gm__ T*)jvp_out + globalIndex*iterStep/2, this->coreBatch*iterStep/2);

            pipe->InitBuffer(inQueueGluOut,  BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(inQueueInputLeft,  BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(inQueueVLeft,  BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(inQueueVRight, BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(outQueueJvpOut,  BUFFER_NUM, this->tileDataNum * sizeof(T));
    
            // pipe->InitBuffer(QueueOnes,     this->tileDataNum * sizeof(T));
            pipe->InitBuffer(QueueSigDeriv, this->tileDataNum * sizeof(T));
            // pipe->InitBuffer(QueueTerm1,    this->tileDataNum * sizeof(T));
            pipe->InitBuffer(QueueTerm2,    this->tileDataNum * sizeof(T));
    
            // ones = QueueOnes.Get<T>();
            // T inputVal(1.0);
            // AscendC::Duplicate<T>(ones, inputVal, this->tileDataNum);
            // term1 = QueueTerm1.Get<T>();
            term2 = QueueTerm2.Get<T>();
            sig_deriv = QueueSigDeriv.Get<T>();
        }
    
        __aicore__ inline void copyIn(int batch, int progress) {
            LocalTensor<T> inLeftLocal  = inQueueInputLeft.AllocTensor<T>();
            LocalTensor<T> gluOutLocal = inQueueGluOut.AllocTensor<T>();
            LocalTensor<T> vLeftLocal  = inQueueVLeft.AllocTensor<T>();
            LocalTensor<T> vRightLocal = inQueueVRight.AllocTensor<T>();
    
            DataCopy(inLeftLocal,  inputGm[batch * iterStep + progress * this->tileDataNum], this->processDataNum);
            DataCopy(gluOutLocal, gluOutGm[batch * iterStep/2 + progress * this->tileDataNum], this->processDataNum);
            DataCopy(vLeftLocal,  vGm[batch * iterStep + progress * this->tileDataNum], this->processDataNum);
            DataCopy(vRightLocal, vGm[batch * iterStep + iterStep/2  + progress * this->tileDataNum], this->processDataNum);

            inQueueInputLeft.EnQue(inLeftLocal);
            inQueueGluOut.EnQue(gluOutLocal);
            inQueueVLeft.EnQue(vLeftLocal);
            inQueueVRight.EnQue(vRightLocal);
        }
        __aicore__ inline void copyInPad(int batch) {
            LocalTensor<T> inLeftLocal  = inQueueInputLeft.AllocTensor<T>();
            LocalTensor<T> gluOutLocal = inQueueGluOut.AllocTensor<T>();
            LocalTensor<T> vLeftLocal  = inQueueVLeft.AllocTensor<T>();
            LocalTensor<T> vRightLocal = inQueueVRight.AllocTensor<T>();
            AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(padTimes), static_cast<uint32_t>(iterStep/2 * sizeof(T)), static_cast<uint32_t>(iterStep/2 * sizeof(T)), 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(inLeftLocal,  inputGm[batch * iterStep], copyParams, padParams); 
            DataCopyPad(gluOutLocal, gluOutGm[batch * iterStep/2],  copyParams, padParams); 
            DataCopyPad(vLeftLocal,  vGm[batch * iterStep], copyParams, padParams); 
            DataCopyPad(vRightLocal, vGm[batch * iterStep + iterStep/2],copyParams, padParams); 

            inQueueInputLeft.EnQue(inLeftLocal);
            inQueueGluOut.EnQue(gluOutLocal);
            inQueueVLeft.EnQue(vLeftLocal);
            inQueueVRight.EnQue(vRightLocal);
        }
        __aicore__ inline void computePad(int batch) {
            // LocalTensor<T>vLocal = inQueueV.DeQue<T>();
            LocalTensor<T>outLocal = outQueueJvpOut.AllocTensor<T>();
            // LocalTensor<T>sigRightLocal = inQueueInputSigRight.AllocTensor<T>();
            
            LocalTensor<T>gluOutLocal = inQueueGluOut.DeQue<T>();
            LocalTensor<T>xLeft = inQueueInputLeft.DeQue<T>();

            LocalTensor<T>vLeft = inQueueVLeft.DeQue<T>();
            LocalTensor<T>vRight = inQueueVRight.DeQue<T>();
            processDataNum = padTimes* (iterStep/2*sizeof(T) + 31)*32/32/sizeof(T);

            // T scalar = -1;
            AscendC::Div(gluOutLocal, gluOutLocal, xLeft, processDataNum);
            LocalTensor<float>sig = gluOutLocal;
            T scala = -1;
            AscendC::Adds(sig_deriv, sig, scala , processDataNum);
            AscendC::Mul(term2, xLeft, sig_deriv, processDataNum);
            AscendC::Mul(term2, term2, vRight, processDataNum);
            AscendC::Sub(outLocal, vLeft, term2, processDataNum);
            AscendC::Mul(outLocal, outLocal, sig, processDataNum);

            inQueueInputLeft.FreeTensor(xLeft);
            inQueueGluOut.FreeTensor(gluOutLocal);
            inQueueVLeft.FreeTensor(vLeft);
            inQueueVRight.FreeTensor(vRight);
            outQueueJvpOut.EnQue(outLocal);
        }

        __aicore__ inline void compute(int batch,int progress) {
            // LocalTensor<T>vLocal = inQueueV.DeQue<T>();
            LocalTensor<T>outLocal = outQueueJvpOut.AllocTensor<T>();
            // LocalTensor<T>sigRightLocal = inQueueInputSigRight.AllocTensor<T>();
            
            LocalTensor<T>gluOutLocal = inQueueGluOut.DeQue<T>();
            LocalTensor<T>xLeft = inQueueInputLeft.DeQue<T>();
            LocalTensor<T>vLeft = inQueueVLeft.DeQue<T>();
            LocalTensor<T>vRight = inQueueVRight.DeQue<T>();
            // sig = 1/(1 + torch.exp(-x_right))

            // T scalar = -1;
            AscendC::Div(gluOutLocal, gluOutLocal, xLeft, processDataNum);
            LocalTensor<float>sig = gluOutLocal;
            T scala = -1;
            AscendC::Adds(sig_deriv, sig, scala , processDataNum);
            AscendC::Mul(term2, xLeft, sig_deriv, processDataNum);
            AscendC::Mul(term2, term2, vRight, processDataNum);
            AscendC::Sub(outLocal, vLeft, term2, processDataNum);
            AscendC::Mul(outLocal, outLocal, sig, processDataNum);
            inQueueInputLeft.FreeTensor(xLeft);
            inQueueGluOut.FreeTensor(gluOutLocal);
            inQueueVLeft.FreeTensor(vLeft);
            inQueueVRight.FreeTensor(vRight);
            outQueueJvpOut.EnQue(outLocal);
        }
    
        __aicore__ inline void copyOut(int batch, int progress) {
            // printf("out index: %d\n", batch * iterStep/2 + progress * this->tileDataNum);
            auto outLocal = outQueueJvpOut.DeQue<T>();
            DataCopy(jvpOutGm[batch * iterStep/2 + progress * this->tileDataNum], outLocal, this->processDataNum);
            outQueueJvpOut.FreeTensor(outLocal);
        }
        __aicore__ inline void copyOutPad(int batch) {
            // printf("out index: %d\n", batch * iterStep/2 + progress * this->tileDataNum);
            AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(padTimes), static_cast<uint32_t>(iterStep/2 * sizeof(T)), 0, static_cast<uint32_t>(iterStep/2 * sizeof(T)), 0};
            auto outLocal = outQueueJvpOut.DeQue<T>();
            DataCopyPad(jvpOutGm[batch * iterStep], outLocal, copyParams);
            outQueueJvpOut.FreeTensor(outLocal);
        }
        __aicore__ inline void process() {
            int32_t loopCount = this->tileNum;
            // this->processDataNum = this->tileDataNum;
            // // printf("coreBatch:%d loopCount:%d tailDataNum:%d\n", coreBatch, loopCount, tailDataNum);
            // for (int batch = 0; batch < coreBatch; batch++) {
            //     this->processDataNum = this->tileDataNum;
            //     for (int32_t i = 0; i < loopCount; i++) {
            //         if (i == this->tileNum - 1) {
            //         this->processDataNum = this->tailDataNum;
            //         }
            //         copyIn(batch, i);
            //         compute(batch, i);
            //         copyOut(batch, i);
            //     }
            // }

            // int32_t loopCount = this->tileNum;
            // printf("padTimes :%d\n", padTimes);
            // printf("coreBatch:%d loopCount:%d tailDataNum:%d\n", coreBatch, loopCount, tailDataNum);
            if (padTimes > 1) {
                for (int batch = 0; batch < coreBatch; batch += padTimes) {
                    if (coreBatch - batch < padTimes) {
                        padTimes = coreBatch - batch;
                    }
                    // printf("batch :%d\n", batch);
                    copyInPad(batch);
                    computePad(batch);
                    copyOutPad(batch);
                }
            } else {
                for (int batch = 0; batch < coreBatch; batch++) {
                    this->processDataNum = this->tileDataNum;
                    int len = processDataNum;
                    for (int32_t i = 0; i < loopCount; i++) {
                        if (i == this->tileNum - 1) {
                           len = iterStep/2 -  (this->tileNum - 1)*this->tileDataNum;
                           this->processDataNum = this->tailDataNum;
                        }
                        copyIn(batch, i);
                        compute(batch, i);
                        copyOut(batch, i);
                    }
                }
            }
        }
    private:
        uint32_t totalLength,resLength,stride,iterStep, axesDim,coreBatch;
        uint32_t tileDataNum, processDataNum,tailDataNum ,coreDataNum,tileNum;
        uint32_t padTimes;
        AscendC::TPipe *pipe;
        AscendC::GlobalTensor<T> gluOutGm;
        AscendC::GlobalTensor<T> inputGm;
        AscendC::GlobalTensor<T> vGm;
        AscendC::GlobalTensor<T> jvpOutGm;
        TQue<QuePosition::VECIN, BUFFER_NUM>inQueueGluOut, inQueueInputLeft, inQueueVLeft,inQueueInputSigRight, inQueueVRight;
        TQue<QuePosition::VECOUT, BUFFER_NUM>outQueueJvpOut; 
        TBuf<QuePosition::VECCALC>QueueOnes, QueueSigDeriv,QueueTerm1,QueueTerm2; 
        AscendC::LocalTensor<T> ones,sig_deriv,term1, term2, out;
        AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQue;
};


template<typename T>class NoBcastFastHalf{
    public:
        __aicore__ inline NoBcastFastHalf(){}
        __aicore__ inline void Init(GM_ADDR glu_out, GM_ADDR input,GM_ADDR v,GM_ADDR jvp_out,
                                    uint32_t finalTileNum, uint32_t tileDataNum, uint32_t tailDataNum,
                                    uint32_t iterStep, uint32_t smallBatch,uint32_t tail, uint32_t padTimes,
                                    TPipe* pipeIn) {
            // printf("axexDim: %d stride:%d iterStep:%d \n", axesdim, stride, iterStep);
            // int round = 32/sizeof(T);
            // this->axesDim = (axesdim + round - 1)/round*round;
            this->pipe = pipeIn;
            this->iterStep = iterStep;
            this->tileNum = finalTileNum;
            this->tileDataNum = tileDataNum;
            this->tailDataNum = tailDataNum;
            this->padTimes = padTimes;
            // this->stride = stride;
            uint32_t coreNum = AscendC::GetBlockIdx();
            uint32_t bigBatch = smallBatch + 1;
            uint32_t globalIndex = bigBatch * coreNum;
            if (coreNum < tail) {
                this->coreBatch = bigBatch;
            } else {
                this->coreBatch = smallBatch;
                globalIndex -= (bigBatch - smallBatch)*(coreNum - tail);
            }
            // printf("coreBatch:%d iterStep:%d tileDataNum:%d tailDataNum:%d\n",coreBatch, iterStep,tileDataNum,tailDataNum);
    
            gluOutGm.SetGlobalBuffer((__gm__ T*)glu_out + globalIndex*iterStep/2, this->coreBatch*iterStep/2);
            inputGm.SetGlobalBuffer((__gm__ T*)input + globalIndex*iterStep, this->coreBatch*iterStep);
            vGm.SetGlobalBuffer((__gm__ T*)v + globalIndex*iterStep, this->coreBatch*iterStep);
            jvpOutGm.SetGlobalBuffer((__gm__ T*)jvp_out + globalIndex*iterStep/2, this->coreBatch*iterStep/2);
    
            pipe->InitBuffer(inQueueGluOut,  BUFFER_NUM_16, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(inQueueInputLeft,  BUFFER_NUM_16, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(inQueueVLeft,  BUFFER_NUM_16, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(inQueueVRight, BUFFER_NUM_16, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(outQueueJvpOut,  BUFFER_NUM_16, this->tileDataNum * sizeof(T));
    
            pipe->InitBuffer(inQueueSmallInputLeft,     this->tileDataNum * sizeof(float));
            pipe->InitBuffer(inQueueSmallVLeft,    this->tileDataNum * sizeof(float));
            pipe->InitBuffer(inQueueSmallVRight,    this->tileDataNum * sizeof(float));
            pipe->InitBuffer(outQueueSmallJvpOut, this->tileDataNum * sizeof(float));
            pipe->InitBuffer(inQueueSmallGlu, this->tileDataNum * sizeof(float));

            smallInLeftLocal = inQueueSmallInputLeft.Get<float>();
            smallVRightLocal = inQueueSmallVLeft.Get<float>();
            smallVLeftLocal = inQueueSmallVRight.Get<float>();
            smallJvpOutLocal = outQueueSmallJvpOut.Get<float>();
            smallGlu = inQueueSmallGlu.Get<float>();

            pipe->InitBuffer(QueueOnes,     this->tileDataNum * sizeof(float));
            pipe->InitBuffer(QueueSigDeriv, this->tileDataNum * sizeof(float));
            pipe->InitBuffer(QueueTerm1,    this->tileDataNum * sizeof(float));
            pipe->InitBuffer(QueueTerm2,    this->tileDataNum * sizeof(float));
            ones = QueueOnes.Get<float>();
            T inputVal(1.0);
            AscendC::Duplicate<float>(ones, inputVal, this->tileDataNum);
            term1 = QueueTerm1.Get<float>();
            term2 = QueueTerm2.Get<float>();
            sig_deriv = QueueSigDeriv.Get<float>();
            // sharedTmpBuffer = tmpQue.Get<uint8_t>();
        }
    
        __aicore__ inline void copyIn(int batch, int progress) {
            LocalTensor<T> inLeftLocal  = inQueueInputLeft.AllocTensor<T>();
            LocalTensor<T> gluOutLocal = inQueueGluOut.AllocTensor<T>();
            // LocalTensor<T> inRightLocal = inQueueInputSigRight.AllocTensor<T>();
            LocalTensor<T> vLeftLocal  = inQueueVLeft.AllocTensor<T>();
            LocalTensor<T> vRightLocal = inQueueVRight.AllocTensor<T>();
            
            // printf("inLeftLocal index %d\n", batch * iterStep + progress * this->tileDataNum);
            DataCopy(inLeftLocal,  inputGm[batch * iterStep + progress * this->tileDataNum], this->processDataNum);
            DataCopy(gluOutLocal, inputGm[batch * iterStep + iterStep/2  + progress * this->tileDataNum], this->processDataNum);
            DataCopy(vLeftLocal,  vGm[batch * iterStep + progress * this->tileDataNum], this->processDataNum);
            DataCopy(vRightLocal, vGm[batch * iterStep + iterStep/2  + progress * this->tileDataNum], this->processDataNum);

            inQueueInputLeft.EnQue(inLeftLocal);
            inQueueGluOut.EnQue(gluOutLocal);
            inQueueVLeft.EnQue(vLeftLocal);
            inQueueVRight.EnQue(vRightLocal);
        }
        __aicore__ inline void copyInPad(int batch) {
            LocalTensor<T> inLeftLocal = inQueueInputLeft.AllocTensor<T>();
            LocalTensor<T> gluOutLocal = inQueueGluOut.AllocTensor<T>();
            LocalTensor<T> vLeftLocal  = inQueueVLeft.AllocTensor<T>();
            LocalTensor<T> vRightLocal = inQueueVRight.AllocTensor<T>();
            AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(padTimes), static_cast<uint32_t>(iterStep/2 * sizeof(T)), static_cast<uint32_t>(iterStep/2 * sizeof(T)), 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
            AscendC::DataCopyExtParams copyGluParams{static_cast<uint16_t>(padTimes), static_cast<uint32_t>(iterStep/2 * sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位

            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(inLeftLocal, inputGm[batch * iterStep], copyParams, padParams); 
            DataCopyPad(gluOutLocal, inputGm[batch * iterStep + iterStep/2],  copyParams, padParams); 
            DataCopyPad(vLeftLocal,  vGm[batch * iterStep], copyParams, padParams); 
            DataCopyPad(vRightLocal, vGm[batch * iterStep + iterStep/2], copyParams, padParams); 

            inQueueInputLeft.EnQue(inLeftLocal);
            inQueueGluOut.EnQue(gluOutLocal);
            inQueueVLeft.EnQue(vLeftLocal);
            inQueueVRight.EnQue(vRightLocal);
        }
        __aicore__ inline void computePad(int batch) {
            LocalTensor<T>outLocal = outQueueJvpOut.AllocTensor<T>();
            // LocalTensor<T>sigRightLocal = inQueueInputSigRight.AllocTensor<T>();
            processDataNum = padTimes* ((iterStep/2*sizeof(T) + 31)/32*32/sizeof(T));

            LocalTensor<T>gluOutLocal = inQueueGluOut.DeQue<T>();
            LocalTensor<T>xLeft = inQueueInputLeft.DeQue<T>();
            // LocalTensor<T>xRight = inQueueInputSigRight.DeQue<T>();
            LocalTensor<T>vLeft = inQueueVLeft.DeQue<T>();
            LocalTensor<T>vRight = inQueueVRight.DeQue<T>();
            // printf("xLeft\n");
            // // for (int  j = 0; j < 2; j++) {
            // for (int i = 0; i < 8; i++) {
            //     printf("%f ", xLeft.GetValue(i));
            // // }
            //     // printf("\n");
            // }
            // printf("\n");
            // printf("sig\n");
            // for (int  j = 0; j < 2; j++) {
            // for (int i = 0; i < 8; i++) {
            //     printf("%f ", gluOutLocal.GetValue(i));
            // // }
            //     // printf("\n");
            // }
            // printf("\n");
            // ,smallInRightLocal,, smallVLeftLocal, smallJvpOutLocal;
            Cast(smallInLeftLocal, xLeft, AscendC::RoundMode::CAST_NONE, processDataNum);
            Cast(smallGlu ,gluOutLocal,  AscendC::RoundMode::CAST_NONE, processDataNum);
            Cast(smallVRightLocal, vRight, AscendC::RoundMode::CAST_NONE, processDataNum);
            Cast(smallVLeftLocal, vLeft, AscendC::RoundMode::CAST_NONE, processDataNum);
    
            float scalar = -1;
            // AscendC::Div(smallGlu, smallGlu, smallInLeftLocal, processDataNum);
            AscendC::Muls(smallGlu, smallGlu, scalar, processDataNum);
            AscendC::Exp(smallGlu, smallGlu, processDataNum);
            AscendC::Add(smallGlu, smallGlu, ones, processDataNum);
            AscendC::Div(smallGlu, ones, smallGlu, processDataNum);
            LocalTensor<float>sig = smallGlu;
            // printf("sig\n");
            // // for (int  j = 0; j < 2; j++) {
            // for (int i = 0; i < 8; i++) {
            //     printf("%f ", sig.GetValue(i));
            // // }
            //     // printf("\n");
            // }
            // printf("\n");
            AscendC::Sub(sig_deriv, ones, sig, processDataNum);

            AscendC::Mul(sig_deriv, sig, sig_deriv, processDataNum);
            // printf("sig_deriv\n");
            // // for (int  j = 0; j < 2; j++) {
            // for (int i = 0; i < 8; i++) {
            //     printf("%f ", sig_deriv.GetValue(i));
            // // }
            //     // printf("\n");
            // }
            // printf("\n");
            AscendC::Mul(term1, sig, smallVLeftLocal, processDataNum);

            AscendC::Mul(term2, smallInLeftLocal, sig_deriv, processDataNum);

            AscendC::Mul(term2, term2, smallVRightLocal, processDataNum);
            // printf("term1\n");
            // // for (int  j = 0; j < 2; j++) {
            // for (int i = 0; i < 8; i++) {
            //     printf("%f ", term1.GetValue(i));
            // // }
            //     // printf("\n");
            // }
            // printf("\n");
            // printf("term2\n");
            // // for (int  j = 0; j < 2; j++) {
            // for (int i = 0; i < 8; i++) {
            //     printf("%f ", term2.GetValue(i));
            // // }
            //     // printf("\n");
            // }
            // printf("\n");
            AscendC::Add(smallJvpOutLocal, term1, term2, processDataNum);
            Cast(outLocal, smallJvpOutLocal, AscendC::RoundMode::CAST_ROUND, processDataNum);

            inQueueInputLeft.FreeTensor(xLeft);
            inQueueGluOut.FreeTensor(gluOutLocal);
            inQueueVLeft.FreeTensor(vLeft);
            inQueueVRight.FreeTensor(vRight);
            outQueueJvpOut.EnQue(outLocal);
        }
        __aicore__ inline void compute(int batch,int progress) {
            LocalTensor<T>outLocal = outQueueJvpOut.AllocTensor<T>();
            // LocalTensor<T>sigRightLocal = inQueueInputSigRight.AllocTensor<T>();

            LocalTensor<T>gluOutLocal = inQueueGluOut.DeQue<T>();
            LocalTensor<T>xLeft = inQueueInputLeft.DeQue<T>();
            // LocalTensor<T>xRight = inQueueInputSigRight.DeQue<T>();
            LocalTensor<T>vLeft = inQueueVLeft.DeQue<T>();
            LocalTensor<T>vRight = inQueueVRight.DeQue<T>();
            // printf("xLeft\n");
            // for (int i = 0; i < processDataNum; i++) {
            //     printf("%f ", xLeft.GetValue(i));
            // }
            // printf("\n");
            // ,smallInRightLocal,, smallVLeftLocal, smallJvpOutLocal;
            Cast(smallInLeftLocal, xLeft, AscendC::RoundMode::CAST_NONE, processDataNum);
            Cast(smallGlu ,gluOutLocal,  AscendC::RoundMode::CAST_NONE, processDataNum);
            Cast(smallVRightLocal, vRight, AscendC::RoundMode::CAST_NONE, processDataNum);
            Cast(smallVLeftLocal, vLeft, AscendC::RoundMode::CAST_NONE, processDataNum);
    
            float scalar = -1;
            // AscendC::Div(smallGlu, smallGlu, smallInLeftLocal, processDataNum);
            AscendC::Muls(smallGlu, smallGlu, scalar, processDataNum);
            AscendC::Exp(smallGlu, smallGlu, processDataNum);
            AscendC::Add(smallGlu, smallGlu, ones, processDataNum);
            AscendC::Div(smallGlu, ones, smallGlu, processDataNum);
            LocalTensor<float>sig = smallGlu;
            AscendC::Sub(sig_deriv, ones, sig, processDataNum);

            AscendC::Mul(sig_deriv, sig, sig_deriv, processDataNum);

            AscendC::Mul(term1, sig, smallVLeftLocal, processDataNum);

            AscendC::Mul(term2, smallInLeftLocal, sig_deriv, processDataNum);

            AscendC::Mul(term2, term2, smallVRightLocal, processDataNum);

            AscendC::Add(smallJvpOutLocal, term1, term2, processDataNum);
            Cast(outLocal, smallJvpOutLocal, AscendC::RoundMode::CAST_ROUND, processDataNum);

            inQueueInputLeft.FreeTensor(xLeft);
            inQueueGluOut.FreeTensor(gluOutLocal);
            inQueueVLeft.FreeTensor(vLeft);
            inQueueVRight.FreeTensor(vRight);
            outQueueJvpOut.EnQue(outLocal);
        }
    
        __aicore__ inline void copyOut(int batch, int progress) {
            auto outLocal = outQueueJvpOut.DeQue<T>();
            // printf("out index: %d processDataNum:%d\n", batch * iterStep/2 + progress * this->tileDataNum,processDataNum);
            DataCopy(jvpOutGm[batch * iterStep/2 + progress * this->tileDataNum], outLocal, this->processDataNum);
            outQueueJvpOut.FreeTensor(outLocal);
        }
        __aicore__ inline void copyOutPad(int batch) {
            // printf("out index: %d\n", batch * iterStep/2 + progress * this->tileDataNum);
            auto outLocal = outQueueJvpOut.DeQue<T>();
            // printf("outLocal\n");
            // for (int i = 0; i < 16; i++) {
            //     printf("%f ", outLocal.GetValue(i));
            // }
            // printf("\n");
            AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(padTimes), static_cast<uint32_t>(iterStep/2 * sizeof(T)), 0, 0, 0};
            DataCopyPad(jvpOutGm[batch * iterStep/2], outLocal, copyParams);
            outQueueJvpOut.FreeTensor(outLocal);
        }
        __aicore__ inline void process() {
            // printf("padTimes:%d \n", padTimes);
            int32_t loopCount = this->tileNum;
            this->processDataNum = this->tileDataNum;
            // printf("coreBatch:%d loopCount:%d tailDataNum:%d\n", coreBatch, loopCount,tailDataNum);
            // for (int batch = 0; batch < coreBatch; batch++) {
            //      this->processDataNum = this->tileDataNum;
            //     for (int32_t i = 0; i < loopCount; i++) {
            //         if (i == this->tileNum - 1) {
            //         this->processDataNum = this->tailDataNum;
            //         }
            //         copyIn(batch, i);
            //         compute(batch, i);
            //         copyOut(batch, i);
            //     }
            // }
            if (padTimes > 1) {
                for (int batch = 0; batch < coreBatch; batch += padTimes) {
                    if (coreBatch - batch < padTimes) {
                        padTimes = coreBatch - batch;
                    }
                    // printf("batch :%d\n", batch);
                    copyInPad(batch);
                    computePad(batch);
                    copyOutPad(batch);
                }
            } else {
                for (int batch = 0; batch < coreBatch; batch++) {
                    this->processDataNum = this->tileDataNum;
                    int len = processDataNum;
                    for (int32_t i = 0; i < loopCount; i++) {
                        if (i == this->tileNum - 1) {
                           len = iterStep/2 -  (this->tileNum - 1)*this->tileDataNum;
                           this->processDataNum = this->tailDataNum;
                        }
                        copyIn(batch, i);
                        compute(batch, i);
                        copyOut(batch, i);
                    }
                }
            }
        }
    private:
        uint32_t totalLength,resLength,stride,iterStep, axesDim,coreBatch;
        uint32_t tileDataNum, processDataNum,tailDataNum ,coreDataNum,tileNum;
        uint32_t padTimes;
        AscendC::TPipe *pipe;
        AscendC::GlobalTensor<T> gluOutGm;
        AscendC::GlobalTensor<T> inputGm;
        AscendC::GlobalTensor<T> vGm;
        AscendC::GlobalTensor<T> jvpOutGm;
        TQue<QuePosition::VECIN, BUFFER_NUM_16>inQueueGluOut, inQueueInputLeft, inQueueVLeft,inQueueInputSigRight, inQueueVRight;
        TQue<QuePosition::VECOUT, BUFFER_NUM_16>outQueueJvpOut; 
        TBuf<QuePosition::VECCALC>QueueOnes, QueueSigDeriv,QueueTerm1,QueueTerm2; 
        TBuf<QuePosition::VECCALC>inQueueSmallInputLeft, inQueueSmallVLeft,inQueueSmallInputSigRight, inQueueSmallVRight, outQueueSmallJvpOut, inQueueSmallGlu;
        AscendC::LocalTensor<float> ones,sig_deriv,term1, term2, out;
        AscendC::LocalTensor<float>smallInLeftLocal,smallInSigRightLocal, smallVRightLocal, smallVLeftLocal, smallJvpOutLocal, smallGlu;
        AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQue;
        AscendC::LocalTensor<uint8_t> sharedTmpBuffer;
};
template<typename T>class NoBcastFastBf{
    public:
        __aicore__ inline NoBcastFastBf(){}
        __aicore__ inline void Init(GM_ADDR glu_out, GM_ADDR input,GM_ADDR v,GM_ADDR jvp_out,
                                    uint32_t finalTileNum, uint32_t tileDataNum, uint32_t tailDataNum,
                                    uint32_t iterStep, uint32_t smallBatch,uint32_t tail, 
                                    TPipe* pipeIn) {
            // printf("axexDim: %d stride:%d iterStep:%d \n", axesdim, stride, iterStep);
            // int round = 32/sizeof(T);
            // this->axesDim = (axesdim + round - 1)/round*round;
            this->pipe = pipeIn;
            this->iterStep = iterStep;
            this->tileNum = finalTileNum;
            this->tileDataNum = 32;
            this->tailDataNum = tailDataNum;
            // this->stride = stride;
            uint32_t coreNum = AscendC::GetBlockIdx();
            uint32_t bigBatch = smallBatch + 1;
            uint32_t globalIndex = bigBatch * coreNum;
            if (coreNum < tail) {
                this->coreBatch = bigBatch;
            } else {
                this->coreBatch = smallBatch;
                globalIndex -= (bigBatch - smallBatch)*(coreNum - tail);
            }
            // printf("coreBatch:%d iterStep:%d tileDataNum:%d tailDataNum:%d\n",coreBatch, iterStep,tileDataNum,tailDataNum);
    
            gluOutGm.SetGlobalBuffer((__gm__ T*)glu_out + globalIndex*iterStep/2, this->coreBatch*iterStep/2);
            inputGm.SetGlobalBuffer((__gm__ T*)input + globalIndex*iterStep, this->coreBatch*iterStep);
            vGm.SetGlobalBuffer((__gm__ T*)v + globalIndex*iterStep, this->coreBatch*iterStep);
            jvpOutGm.SetGlobalBuffer((__gm__ T*)jvp_out + globalIndex*iterStep/2, this->coreBatch*iterStep/2);
    
            pipe->InitBuffer(inQueueGluOut,  BUFFER_NUM_16, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(inQueueInputLeft,  BUFFER_NUM_16, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(inQueueVLeft,  BUFFER_NUM_16, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(inQueueVRight, BUFFER_NUM_16, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(outQueueJvpOut,  BUFFER_NUM_16, this->tileDataNum * sizeof(T));
    
            pipe->InitBuffer(inQueueSmallInputLeft,     this->tileDataNum * sizeof(float));
            pipe->InitBuffer(inQueueSmallVLeft,    this->tileDataNum * sizeof(float));
            pipe->InitBuffer(inQueueSmallVRight,    this->tileDataNum * sizeof(float));
            pipe->InitBuffer(outQueueSmallJvpOut, this->tileDataNum * sizeof(float));
            pipe->InitBuffer(inQueueSmallGlu, this->tileDataNum * sizeof(float));

            smallInLeftLocal = inQueueSmallInputLeft.Get<float>();
            smallVRightLocal = inQueueSmallVLeft.Get<float>();
            smallVLeftLocal = inQueueSmallVRight.Get<float>();
            smallJvpOutLocal = outQueueSmallJvpOut.Get<float>();
            smallGlu = inQueueSmallGlu.Get<float>();
    
            pipe->InitBuffer(QueueOnes,     this->tileDataNum * sizeof(float));
            pipe->InitBuffer(QueueSigDeriv, this->tileDataNum * sizeof(float));
            pipe->InitBuffer(QueueTerm1,    this->tileDataNum * sizeof(float));
            pipe->InitBuffer(QueueTerm2,    this->tileDataNum * sizeof(float));
            ones = QueueOnes.Get<float>();
            T inputVal(1.0);
            AscendC::Duplicate<float>(ones, inputVal, this->tileDataNum);
            term1 = QueueTerm1.Get<float>();
            term2 = QueueTerm2.Get<float>();
            sig_deriv = QueueSigDeriv.Get<float>();
            // sharedTmpBuffer = tmpQue.Get<uint8_t>();
        }
    
        __aicore__ inline void copyIn(int batch, int progress) {
            LocalTensor<T> inLeftLocal  = inQueueInputLeft.AllocTensor<T>();
            LocalTensor<T> gluOutLocal = inQueueGluOut.AllocTensor<T>();
            // LocalTensor<T> inRightLocal = inQueueInputSigRight.AllocTensor<T>();
            LocalTensor<T> vLeftLocal  = inQueueVLeft.AllocTensor<T>();
            LocalTensor<T> vRightLocal = inQueueVRight.AllocTensor<T>();
    
            DataCopy(inLeftLocal,  inputGm[batch * iterStep + progress * this->tileDataNum], this->processDataNum);
            DataCopy(gluOutLocal, inputGm[batch * iterStep + iterStep/2  + progress * this->tileDataNum], this->processDataNum);
            DataCopy(vLeftLocal,  vGm[batch * iterStep + progress * this->tileDataNum], this->processDataNum);
            DataCopy(vRightLocal, vGm[batch * iterStep + iterStep/2  + progress * this->tileDataNum], this->processDataNum);

            inQueueInputLeft.EnQue(inLeftLocal);
            inQueueGluOut.EnQue(gluOutLocal);
            inQueueVLeft.EnQue(vLeftLocal);
            inQueueVRight.EnQue(vRightLocal);
        }
    
        __aicore__ inline void compute(int batch,int progress) {
            LocalTensor<T>outLocal = outQueueJvpOut.AllocTensor<T>();
            // LocalTensor<T>sigRightLocal = inQueueInputSigRight.AllocTensor<T>();

            LocalTensor<T>gluOutLocal = inQueueGluOut.DeQue<T>();
            LocalTensor<T>xLeft = inQueueInputLeft.DeQue<T>();
            LocalTensor<T>vLeft = inQueueVLeft.DeQue<T>();
            LocalTensor<T>vRight = inQueueVRight.DeQue<T>();
            // ,smallInRightLocal,, smallVLeftLocal, smallJvpOutLocal;
            // Cast(smallInLeftLocal, xLeft, AscendC::RoundMode::CAST_ROUND, processDataNum);
            for (int i = 0; i < processDataNum; i++) {
                T x = xLeft.GetValue(i);
                float t = AscendC::ToFloat(x);
                // float v = x;
                smallInLeftLocal.SetValue(i,t);
            }
            // printf("gluOutLocal now\n");
            // for (int i = 0; i < processDataNum; i++) {
            //     printf("%f ", gluOutLocal.GetValue(i));
            // }
            // printf("\n");
            for (int i = 0; i < processDataNum; i++) {
                T x = gluOutLocal.GetValue(i);
                float t = AscendC::ToFloat(x);
                // float v = x;
                smallGlu.SetValue(i,t);
            }
            for (int i = 0; i < processDataNum; i++) {
                T x = vRight.GetValue(i);
                float t = AscendC::ToFloat(x);
                // float v = x;
                smallVRightLocal.SetValue(i,t);
            }
            for (int i = 0; i < processDataNum; i++) {
                T x = vLeft.GetValue(i);
                float t = AscendC::ToFloat(x);
                // float v = x;
                smallVLeftLocal.SetValue(i,t);
            }

            float scalar = -1;
            // AscendC::Div(smallGlu, smallGlu, smallInLeftLocal, processDataNum);
            // printf("xright\n");
            // for (int i = 0; i < processDataNum; i++) {
            //     printf("%f ", smallGlu.GetValue(i));
            // }
            // printf("\n");
            AscendC::Muls(smallGlu, smallGlu, scalar, processDataNum);
            AscendC::Exp(smallGlu, smallGlu, processDataNum);
            AscendC::Add(smallGlu, smallGlu, ones, processDataNum);
            AscendC::Div(smallGlu, ones, smallGlu, processDataNum);
            LocalTensor<float>sig = smallGlu;
            // printf("sig\n");
            // for (int i = 0; i < processDataNum; i++) {
            //     printf("%f ", sig.GetValue(i));
            // }
            // printf("\n");
            AscendC::Sub(sig_deriv, ones, sig, processDataNum);

            AscendC::Mul(sig_deriv, sig, sig_deriv, processDataNum);

            AscendC::Mul(term1, sig, smallVLeftLocal, processDataNum);

            AscendC::Mul(term2, smallInLeftLocal, sig_deriv, processDataNum);

            AscendC::Mul(term2, term2, smallVRightLocal, processDataNum);
            // printf("term2\n");
            // for (int i = 0; i < processDataNum; i++) {
            //     printf("%f ", term2.GetValue(i));
            // }
            // printf("\n");
            //     printf("term1\n");
            // for (int i = 0; i < processDataNum; i++) {
            //     printf("%f ", term1.GetValue(i));
            // }
            // printf("\n");
            AscendC::Add(smallJvpOutLocal, term1, term2, processDataNum);
            // printf("smallJvpOutLocal\n");
            // for (int i = 0; i < processDataNum; i++) {
            //     printf("%f ", smallJvpOutLocal.GetValue(i));
            // }
            // printf("\n");
            // Cast(outLocal, smallJvpOutLocal, AscendC::RoundMode::CAST_TRUNC, processDataNum);
            // Cast(outLocal, smallJvpOutLocal, AscendC::RoundMode::CAST_TRUNC, processDataNum);
            for (int i = 0; i < processDataNum; i++) {
                float x = smallJvpOutLocal.GetValue(i);
                bfloat16_t n = AscendC::ToBfloat16(x);
                outLocal.SetValue(i, n);
                // printf("%f ", smallJvpOutLocal.GetValue(i));
            }
            // printf("\n");
            inQueueInputLeft.FreeTensor(xLeft);
            inQueueGluOut.FreeTensor(gluOutLocal);
            inQueueVLeft.FreeTensor(vLeft);
            inQueueVRight.FreeTensor(vRight);
            outQueueJvpOut.EnQue(outLocal);
        }
    
        __aicore__ inline void copyOut(int batch, int progress) {
            auto outLocal = outQueueJvpOut.DeQue<T>();
            DataCopy(jvpOutGm[batch * iterStep/2 + progress * this->tileDataNum], outLocal, this->processDataNum);
            outQueueJvpOut.FreeTensor(outLocal);
        }
    
        __aicore__ inline void process() {
            int32_t loopCount = this->tileNum;
            this->processDataNum = this->tileDataNum;
            // printf("coreBatch:%d loopCount:%d tailDataNum:%d\n", coreBatch, loopCount,tailDataNum);
            for (int batch = 0; batch < coreBatch; batch++) {
                for (int32_t i = 0; i < loopCount; i++) {
                    if (i == this->tileNum - 1) {
                    this->processDataNum = this->tailDataNum;
                    }
                    copyIn(batch, i);
                    compute(batch, i);
                    copyOut(batch, i);
                }
            }
        }
    private:
        uint32_t totalLength,resLength,stride,iterStep, axesDim,coreBatch;
        uint32_t tileDataNum, processDataNum,tailDataNum ,coreDataNum,tileNum;
        AscendC::TPipe *pipe;
        AscendC::GlobalTensor<T> gluOutGm;
        AscendC::GlobalTensor<T> inputGm;
        AscendC::GlobalTensor<T> vGm;
        AscendC::GlobalTensor<T> jvpOutGm;
        TQue<QuePosition::VECIN, BUFFER_NUM_16>inQueueGluOut, inQueueInputLeft, inQueueVLeft,inQueueInputSigRight, inQueueVRight;
        TQue<QuePosition::VECOUT, BUFFER_NUM_16>outQueueJvpOut; 
        TBuf<QuePosition::VECCALC>QueueOnes, QueueSigDeriv,QueueTerm1,QueueTerm2; 
        TBuf<QuePosition::VECCALC>inQueueSmallInputLeft, inQueueSmallVLeft,inQueueSmallInputSigRight, inQueueSmallVRight, outQueueSmallJvpOut, inQueueSmallGlu;
        AscendC::LocalTensor<float> ones,sig_deriv,term1, term2, out;
        AscendC::LocalTensor<float>smallInLeftLocal,smallInSigRightLocal, smallVRightLocal, smallVLeftLocal, smallJvpOutLocal, smallGlu;
        AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQue;
        AscendC::LocalTensor<uint8_t> sharedTmpBuffer;
};