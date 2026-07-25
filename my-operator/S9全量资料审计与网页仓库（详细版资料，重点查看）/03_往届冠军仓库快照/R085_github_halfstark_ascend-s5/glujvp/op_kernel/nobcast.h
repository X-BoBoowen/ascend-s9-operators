#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
template<typename T>class NoBcast{
public:
    __aicore__ inline NoBcast(){}
    __aicore__ inline void Init(GM_ADDR glu_out, GM_ADDR input,GM_ADDR v,GM_ADDR jvp_out,
                                uint32_t iterStep, uint32_t axesdim,uint32_t smallDataNum,
                                uint32_t tail, TPipe* pipeIn) {
        printf("axexDim: %d stride:%d iterStep:%d \n", axesdim, stride, iterStep);
        int round = 32/sizeof(T);
        this->axesDim = (axesdim + round - 1)/round*round;
        this->pipe = pipeIn;
        this->iterStep = iterStep;
        // this->stride = stride;
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t bigDataNum = smallDataNum + 1;
        uint32_t globalIndex = bigDataNum * coreNum;
        if (coreNum < tail) {
            this->coreDataNum = bigDataNum;
        } else {
            this->coreDataNum = smallDataNum;
            globalIndex -= (bigDataNum - smallDataNum)*(coreNum - tail);
        }
        // gluOutGm.SetGlobalBuffer((__gm__ T*)glu_out + globalIndex*stride, this->coreDataNum*stride *sizeof(T));
        inputGm.SetGlobalBuffer((__gm__ T*)input + globalIndex*iterStep, this->coreDataNum*iterStep);
        vGm.SetGlobalBuffer((__gm__ T*)v + globalIndex*iterStep, this->coreDataNum*iterStep);
        jvpOutGm.SetGlobalBuffer((__gm__ T*)jvp_out + globalIndex*iterStep/2, this->coreDataNum*iterStep/2);

        pipe->InitBuffer(inQueueInput, BUFFER_NUM, 8*axesDim*sizeof(T));
        pipe->InitBuffer(inQueueV, BUFFER_NUM, 8*axesDim*sizeof(T));
        pipe->InitBuffer(QueueOnes, axesDim/2*sizeof(T));
        pipe->InitBuffer(QueueSigDeriv, axesDim/2*sizeof(T));
        pipe->InitBuffer(QueueTerm1, axesDim/2*sizeof(T));
        pipe->InitBuffer(QueueTerm2, axesDim/2*sizeof(T));
        pipe->InitBuffer(outQueueJvpOut, iterStep/2*sizeof(T));
        ones = QueueOnes.Get<T>();
        T inputVal(1.0);
        AscendC::Duplicate<float>(ones, inputVal, axesDim/2);
        sig_deriv = QueueSigDeriv.Get<T>();
        term1 = QueueTerm1.Get<T>();
        term2 = QueueTerm2.Get<T>();
        out = outQueueJvpOut.Get<T>();
    }
    __aicore__ inline void copyIn(int progress, int now) {
        LocalTensor<T>inputLocal = inQueueInput.AllocTensor<T>();
        LocalTensor<T>vLocal = inQueueV.AllocTensor<T>();
        AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(this->axesDim),1*sizeof(T), static_cast<uint32_t>((this->stride-1) * sizeof(T)), 0,0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        // AscendC::DataCopyPad(inputLocal, inputGm[progress*iterStep + now], copyParams, padParams); // 从GM->VECIN搬运40Bytes
        // AscendC::DataCopyPad(vLocal, vGm[progress*iterStep + now], copyParams, padParams); // 从GM->VECIN搬运40Bytes
        // for (int i = 0; i < axesDim; i++) {
        //     inputLocal.SetValue(i, inputLocal.GetValue(i*32/sizeof(T)));
        //     vLocal.SetValue(i, vLocal.GetValue(i*32/sizeof(T)));
        //     // printf("%f ", inputLocal.GetValue(i));
        // }
        // // printf("\n");    
        // for (int i = 0; i < axesDim; i++) {
        //     inputLocal.SetValue(i, inputLocal.GetValue(i*32/sizeof(T)));
        //     vLocal.SetValue(i, vLocal.GetValue(i*32/sizeof(T)));
        //     // printf("%f ", vLocal.GetValue(i));

        // }
        // printf("\n");
        inQueueInput.EnQue<T>(inputLocal);
        inQueueV.EnQue<T>(vLocal);
    }

    __aicore__ inline void compute(int progress, int now) {
        LocalTensor<T>inputLocal = inQueueInput.DeQue<T>();
        LocalTensor<T>vLocal = inQueueV.DeQue<T>();
        // LocalTensor<T>outLocal = inQueueGluOut.AllocTensor<T>();


        LocalTensor<T>xLeft = inputLocal[0];
        LocalTensor<T>xRight = inputLocal[axesDim/2];
        // printf("xRight\n");
        // for (int i = 0; i < axesDim/2; i++) {
        //     printf("%f ", xRight.GetValue(i));
        // }
        // printf("\n");
        LocalTensor<T>vLeft = vLocal[0];
        LocalTensor<T>vRight = vLocal[axesDim/2];

        T scalar = -1;
        AscendC::Muls(xRight, xRight, scalar, axesDim/2);
        // printf("xR\n");
        // for (int i = 0; i < axesDim/2; i++) {
        //     printf("%f ", xRight.GetValue(i));
        // }
        // printf("\n");
        AscendC::Exp(xRight, xRight, axesDim/2);
        AscendC::Add(xRight, ones, xRight, axesDim/2);
        AscendC::Div(xRight, ones, xRight, axesDim/2);
        // printf("xRight div\n");
        // for (int i = 0; i < axesDim/2; i++) {
        //     printf("%f ", xRight.GetValue(i));
        // }
        // printf("\n");
        AscendC::Sub(sig_deriv, ones, xRight, axesDim/2);
        // printf("sig_deriv first\n");
        // for (int i = 0; i < axesDim/2; i++) {
        //     printf("%f ", sig_deriv.GetValue(i));
        // }
        // printf("\n");
        AscendC::Mul(sig_deriv, xRight, sig_deriv, axesDim/2);
        // printf("sig_deriv\n");
        // for (int i = 0; i < axesDim/2; i++) {
        //     printf("%f ", sig_deriv.GetValue(i));
        // }
        // printf("\n");
        AscendC::Mul(term1, xRight, vLeft, axesDim/2);
        AscendC::Mul(term2, xLeft, sig_deriv, axesDim/2);
        AscendC::Mul(term2, term2, vRight, axesDim/2);
        AscendC::Add(term1, term1, term2, axesDim/2);
        // printf("out\n");
        // for (int i = 0; i < axesDim/2; i++) {
        //     printf("%f ", term1.GetValue(i));
        // }
        // printf("\n");
        for (int i = 0; i < axesDim/2; i++) {
            out.SetValue(i*stride + now, term1.GetValue(i));
        }
        inQueueV.FreeTensor(vLocal);
        inQueueInput.FreeTensor(inputLocal);
    }

    __aicore__ inline void copyOut(int progress) {
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->iterStep/2 * sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        // AscendC::DataCopyPad(jvpOutGm[progress*iterStep/2], out, copyParams); // 从GM->VECIN搬运40Bytes
    }

    __aicore__ inline void process() {
        // printf("coreDataNum:%d stride:%d\n", coreDataNum, stride);
        for (int i = 0; i < coreDataNum; i++) {
            for (int j = 0; j < stride; j++) {
                copyIn(i, j);
                compute(i, j);
            }
            copyOut(i);
        }
    }
private:
    uint32_t totalLength,resLength,stride,iterStep, axesDim,coreDataNum;
    AscendC::TPipe *pipe;
    AscendC::GlobalTensor<T> gluOutGm;
    AscendC::GlobalTensor<T> inputGm;
    AscendC::GlobalTensor<T> vGm;
    AscendC::GlobalTensor<T> jvpOutGm;
    TQue<QuePosition::VECIN, BUFFER_NUM>inQueueGluOut, inQueueInput, inQueueV;
    TBuf<QuePosition::VECCALC>outQueueJvpOut; 
    TBuf<QuePosition::VECCALC>QueueOnes, QueueSigDeriv,QueueTerm1,QueueTerm2; 
    AscendC::LocalTensor<T> ones,sig_deriv,term1, term2, out;

};