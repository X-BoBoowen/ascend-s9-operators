#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1; // tensor num for each queue
constexpr int32_t BUFFER_NUM_16 = 1; // tensor num for each queue
template<typename T>class NoBcastFastFloat{
    public:
        __aicore__ inline NoBcastFastFloat(){}
        __aicore__ inline void Init(GM_ADDR grad_x,GM_ADDR y_grad, GM_ADDR x, GM_ADDR v_y, GM_ADDR v_x, GM_ADDR jvp_out,
                                    uint32_t finalTileNum, uint32_t tileDataNum, uint32_t tailDataNum,
                                    uint32_t iterStep, uint32_t smallBatch,uint32_t tail, uint32_t padTimes,
                                    TPipe* pipeIn) {
            // printf("axexDim: %d stride:%d iterStep:%d \n", axesdim, stride, iterStep);
            // int round = 32/sizeof(T);
            // this->axesDim = (axesdim + round - 1)/round*round;
            this->pipe = pipeIn;
            this->padTimes = padTimes;
            this->iterStep = iterStep;
            this->tileNum = finalTileNum;
            this->tileDataNum = tileDataNum;
            this->tailDataNum = tailDataNum;
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
            gradXGm.SetGlobalBuffer((__gm__ T*)grad_x + globalIndex*iterStep, this->coreBatch*iterStep);
            yGradGm.SetGlobalBuffer((__gm__ T*)y_grad + globalIndex*iterStep/2, this->coreBatch*iterStep/2);
            xGm.SetGlobalBuffer((__gm__ T*)x + globalIndex*iterStep, this->coreBatch*iterStep);
            dgradGluGm.SetGlobalBuffer((__gm__ T*)v_y + globalIndex*iterStep/2, this->coreBatch*iterStep/2);
            dxGm.SetGlobalBuffer((__gm__ T*)v_x + globalIndex*iterStep, this->coreBatch*iterStep);
            outGm.SetGlobalBuffer((__gm__ T*)jvp_out + globalIndex*iterStep, this->coreBatch*iterStep);

            pipe->InitBuffer(inQueueB,  BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(inQueueA,  BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(inQueueDA, BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(inQueueDB, BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(inQueueGradxa, BUFFER_NUM, this->tileDataNum * sizeof(T));
            // pipe->InitBuffer(inQueueGradxb, BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(outQueueXa,  BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(outQueueXb,  BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(inQueuedgrad_glu,  BUFFER_NUM, this->tileDataNum * sizeof(T));
            // Queuedb_neg_sig_b
            pipe->InitBuffer(Queuedb_neg_sig_b,     this->tileDataNum * sizeof(T));
            pipe->InitBuffer(QueueOnes,     this->tileDataNum * sizeof(T));
            // pipe->InitBuffer(QueueTerm1mid, this->tileDataNum * sizeof(T));
            // pipe->InitBuffer(QueueTerm2a,   this->tileDataNum * sizeof(T));
            // pipe->InitBuffer(QueueTerm2b,   this->tileDataNum * sizeof(T));
            pipe->InitBuffer(Queueglu,   this->tileDataNum * sizeof(T));

            ones = QueueOnes.Get<T>();
            T inputVal(1.0);
            AscendC::Duplicate<T>(ones, inputVal, this->tileDataNum);
            db_neg_sig_b = Queuedb_neg_sig_b.Get<T>();
            // term1mid = QueueTerm1mid.Get<T>();
            // term2a = QueueTerm2a.Get<T>();
            // term2b = QueueTerm2b.Get<T>();
            glu = Queueglu.Get<T>();
        }
    
        __aicore__ inline void copyIn(int batch, int progress) {
            LocalTensor<T> a  = inQueueA.AllocTensor<T>();
            LocalTensor<T> b = inQueueB.AllocTensor<T>();
            LocalTensor<T> da  = inQueueDA.AllocTensor<T>();
            LocalTensor<T> db = inQueueDB.AllocTensor<T>();
            LocalTensor<T> gradXa = inQueueGradxa.AllocTensor<T>();
            // LocalTensor<T> gradXb = inQueueGradxb.AllocTensor<T>();
            LocalTensor<T> dgrad_glu = inQueuedgrad_glu.AllocTensor<T>();

            // LocalTensor<T> vyLocal = inQueueVY.AllocTensor<T>();
            
            // printf("input index1: %d\n", batch * iterStep + progress * this->tileDataNum);
            // printf("input index2: %d\n", batch * iterStep + iterStep/2 + progress * this->tileDataNum);

            DataCopy(a,  xGm[batch * iterStep + progress * this->tileDataNum], this->processDataNum);
            DataCopy(b,  xGm[batch * iterStep + iterStep/2 + progress * this->tileDataNum], this->processDataNum);
            DataCopy(da,  dxGm[batch * iterStep + progress * this->tileDataNum], this->processDataNum);
            DataCopy(db, dxGm[batch * iterStep + iterStep/2  + progress * this->tileDataNum], this->processDataNum);
            DataCopy(gradXa,  gradXGm[batch * iterStep + progress * this->tileDataNum], this->processDataNum);
            // DataCopy(gradXb, gradXGm[batch * iterStep + iterStep/2  + progress * this->tileDataNum], this->processDataNum);
            DataCopy(dgrad_glu, dgradGluGm[batch * iterStep/2 + progress * this->tileDataNum], this->processDataNum);


            inQueueA.EnQue(a);
            inQueueB.EnQue(b);
            inQueueDA.EnQue(da);
            inQueueDB.EnQue(db);
            inQueueGradxa.EnQue(gradXa);
            // inQueueGradxb.EnQue(gradXb);
            inQueuedgrad_glu.EnQue(dgrad_glu);

        }
        __aicore__ inline void copyInPad(int batchIdx) {
            LocalTensor<T> a  = inQueueA.AllocTensor<T>();
            LocalTensor<T> b = inQueueB.AllocTensor<T>();
            LocalTensor<T> da  = inQueueDA.AllocTensor<T>();
            LocalTensor<T> db = inQueueDB.AllocTensor<T>();
            LocalTensor<T> gradXa = inQueueGradxa.AllocTensor<T>();
            // LocalTensor<T> gradXb = inQueueGradxb.AllocTensor<T>();
            LocalTensor<T> dgrad_glu = inQueuedgrad_glu.AllocTensor<T>();
            AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(padTimes), static_cast<uint32_t>(iterStep/2 * sizeof(T)), static_cast<uint32_t>(iterStep/2 * sizeof(T)), 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(a,  xGm[batchIdx * iterStep], copyParams, padParams); 
            AscendC::DataCopyPad(b,  xGm[batchIdx * iterStep + iterStep/2], copyParams, padParams); 
            AscendC::DataCopyPad(da,  dxGm[batchIdx * iterStep], copyParams, padParams); 
            AscendC::DataCopyPad(db,  dxGm[batchIdx * iterStep + iterStep/2], copyParams, padParams); 
            AscendC::DataCopyPad(gradXa,  gradXGm[batchIdx * iterStep], copyParams, padParams); 
            // AscendC::DataCopyPad(gradXb,  gradXGm[batchIdx * iterStep + iterStep/2], copyParams, padParams); 
            // printf("a iterStep/2:%d\n",iterStep/2);
            // for (int i = 0; i < padTimes; i++) {
            //     for (int j = 0; j < iterStep/2 ; j++) {
            //         printf("%f ",a.GetValue(i*iterStep/2 + j));
            //     }
            //     printf("\n");
            // }

            AscendC::DataCopyExtParams copyGParams{static_cast<uint16_t>(padTimes), static_cast<uint32_t>(iterStep/2 * sizeof(T)),0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
            AscendC::DataCopyPad(dgrad_glu,  dgradGluGm[batchIdx * iterStep/2], copyGParams, padParams); 

            inQueueA.EnQue(a);
            inQueueB.EnQue(b);
            inQueueDA.EnQue(da);
            inQueueDB.EnQue(db);
            inQueueGradxa.EnQue(gradXa);
            // inQueueGradxb.EnQue(gradXb);
            inQueuedgrad_glu.EnQue(dgrad_glu);
        }
        __aicore__ inline void computePad(int batch) {
            // LocalTensor<T>vLocal = inQueueV.DeQue<T>();
            LocalTensor<T>dgrad_x_a = outQueueXa.AllocTensor<T>();
            LocalTensor<T>dgrad_x_b = outQueueXb.AllocTensor<T>();
            // LocalTensor<T>sigRightLocal = inQueueInputSigRight.AllocTensor<T>();
            float two = -2;
            processDataNum = padTimes* (iterStep/2*sizeof(T) + 31)*32/32/sizeof(T);
            LocalTensor<T>a = inQueueA.DeQue<T>();
            LocalTensor<T>b = inQueueB.DeQue<T>();
            LocalTensor<T>da = inQueueDA.DeQue<T>();
            LocalTensor<T>db = inQueueDB.DeQue<T>();
            LocalTensor<T>gradxa = inQueueGradxa.DeQue<T>();
            // LocalTensor<T>gradxb = inQueueGradxb.DeQue<T>();
            LocalTensor<T>dgrad_glu =  inQueuedgrad_glu.DeQue<T>();
            float scalar = -1;
            float ooo = 1;
            AscendC::Muls(b, b, scalar, processDataNum);
            AscendC::Exp(b, b, processDataNum);
            AscendC::Adds(b, b, ooo, processDataNum);
            AscendC::Div(b, ones, b, processDataNum);
            LocalTensor<float>sig_b = b; // sigb
            AscendC::Mul(glu, a, sig_b, processDataNum); // glu
            AscendC::Mul(db_neg_sig_b, db, sig_b, processDataNum);
            AscendC::Sub(db_neg_sig_b, db, db_neg_sig_b, processDataNum);
            
            // AscendC::Mul(dgrad_glu, dgrad_glu, sig_b, processDataNum);
            // AscendC::Mul(dgrad_x_a, gradxa,  db_neg_sig_b, processDataNum);
            // AscendC::Add(dgrad_x_a, dgrad_glu,  dgrad_x_a, processDataNum);
            
            AscendC::Mul(dgrad_x_a, gradxa,  db_neg_sig_b, processDataNum);
            AscendC::MulAddDst(dgrad_x_a, dgrad_glu, sig_b, processDataNum);

            outQueueXa.EnQue(dgrad_x_a);
            AscendC::Sub(dgrad_x_b, a,  glu, processDataNum);
            AscendC::Mul(dgrad_x_b, dgrad_x_a,  dgrad_x_b, processDataNum);
            AscendC::Mul(a, da, sig_b, processDataNum);
            AscendC::Mul(b, glu, db_neg_sig_b, processDataNum);
            AscendC::Sub(da, da,  a, processDataNum);
            AscendC::Sub(da, da,  b, processDataNum);
            // AscendC::Mul(a, gradxa, da, processDataNum);
            // AscendC::Add(dgrad_x_b, dgrad_x_b, a, processDataNum);
            AscendC::MulAddDst(dgrad_x_b, gradxa, da, processDataNum);

            outQueueXb.EnQue(dgrad_x_b);

            inQueueA.FreeTensor(a);
            inQueueB.FreeTensor(b);
            inQueueDA.FreeTensor(da);
            inQueueDB.FreeTensor(db);
            inQueueGradxa.FreeTensor(gradxa);
            // inQueueGradxb.FreeTensor(gradxb);
            inQueuedgrad_glu.FreeTensor(dgrad_glu);

        }
        __aicore__ inline void copyOutPad(int batch) {
            // printf("out index: %d\n", batch * iterStep/2 + progress * this->tileDataNum);
            auto term1 = outQueueXa.DeQue<T>();
            auto term2 = outQueueXb.DeQue<T>();
            // printf("term2 iterStep/2:%d\n",iterStep/2);
            // for (int i = 0; i < padTimes; i++) {
            //     for (int j = 0; j < iterStep/2 ; j++) {
            //         printf("%f ",term2.GetValue(i*iterStep/2 + j));
            //     }
            //     printf("\n");
            // }

            // printf("outindex1: %d\n",batch * iterStep + progress * this->tileDataNum);
            // printf("outindex2: %d\n",batch * iterStep + iterStep/2  + progress * this->tileDataNum);
            AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(padTimes), static_cast<uint32_t>(iterStep/2 * sizeof(T)), 0, static_cast<uint32_t>(iterStep/2 * sizeof(T)), 0};
            // printf("len :%d\n",len);
            AscendC::DataCopyPad(outGm[batch * iterStep], term1, copyParams); 
            AscendC::DataCopyPad(outGm[batch * iterStep + iterStep/2], term2, copyParams); 
            // DataCopy(outGm[batch * iterStep + progress * this->tileDataNum], term1, this->processDataNum);
            // DataCopy(outGm[batch * iterStep + iterStep/2  + progress * this->tileDataNum], term2, this->processDataNum);

            outQueueXa.FreeTensor(term1);
            outQueueXb.FreeTensor(term2);
        }
    
        __aicore__ inline void compute(int batch,int progress) {
            // LocalTensor<T>vLocal = inQueueV.DeQue<T>();
            LocalTensor<T>dgrad_x_a = outQueueXa.AllocTensor<T>();
            LocalTensor<T>dgrad_x_b = outQueueXb.AllocTensor<T>();
            // LocalTensor<T>sigRightLocal = inQueueInputSigRight.AllocTensor<T>();
            float two = -2;
            
            LocalTensor<T>a = inQueueA.DeQue<T>();
            LocalTensor<T>b = inQueueB.DeQue<T>();
            LocalTensor<T>da = inQueueDA.DeQue<T>();
            LocalTensor<T>db = inQueueDB.DeQue<T>();
            LocalTensor<T>gradxa = inQueueGradxa.DeQue<T>();
            // LocalTensor<T>gradxb = inQueueGradxb.DeQue<T>();
            LocalTensor<T>dgrad_glu =  inQueuedgrad_glu.DeQue<T>();
            float scalar = -1;
            float ooo = 1;
            AscendC::Muls(b, b, scalar, processDataNum);
            AscendC::Exp(b, b, processDataNum);
            AscendC::Adds(b, b, ooo, processDataNum);
            AscendC::Div(b, ones, b, processDataNum);
            LocalTensor<float>sig_b = b; // sigb
            AscendC::Mul(glu, a, sig_b, processDataNum); // glu
            AscendC::Mul(db_neg_sig_b, db, sig_b, processDataNum);
            AscendC::Sub(db_neg_sig_b, db, db_neg_sig_b, processDataNum);
            
            AscendC::Mul(dgrad_glu, dgrad_glu, sig_b, processDataNum);
            AscendC::Mul(dgrad_x_a, gradxa,  db_neg_sig_b, processDataNum);
            AscendC::Add(dgrad_x_a, dgrad_glu,  dgrad_x_a, processDataNum);
            outQueueXa.EnQue(dgrad_x_a);
            AscendC::Sub(dgrad_x_b, a,  glu, processDataNum);
            AscendC::Mul(dgrad_x_b, dgrad_x_a,  dgrad_x_b, processDataNum);
            AscendC::Mul(a, da, sig_b, processDataNum);
            AscendC::Mul(b, glu, db_neg_sig_b, processDataNum);
            AscendC::Sub(da, da,  a, processDataNum);
            AscendC::Sub(da, da,  b, processDataNum);
            AscendC::Mul(a, gradxa, da, processDataNum);
            AscendC::Add(dgrad_x_b, dgrad_x_b, a, processDataNum);
            outQueueXb.EnQue(dgrad_x_b);

            inQueueA.FreeTensor(a);
            inQueueB.FreeTensor(b);
            inQueueDA.FreeTensor(da);
            inQueueDB.FreeTensor(db);
            inQueueGradxa.FreeTensor(gradxa);
            // inQueueGradxb.FreeTensor(gradxb);
            inQueuedgrad_glu.FreeTensor(dgrad_glu);

        }
    
        __aicore__ inline void copyOut(int batch, int progress, int len) {
            // printf("out index: %d\n", batch * iterStep/2 + progress * this->tileDataNum);
            auto term1 = outQueueXa.DeQue<T>();
            auto term2 = outQueueXb.DeQue<T>();
            // printf("outindex1: %d\n",batch * iterStep + progress * this->tileDataNum);
            // printf("outindex2: %d\n",batch * iterStep + iterStep/2  + progress * this->tileDataNum);
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
            // printf("len :%d\n",len);
            AscendC::DataCopyPad(outGm[batch * iterStep + progress * this->tileDataNum], term1, copyParams); 
            AscendC::DataCopyPad(outGm[batch * iterStep + iterStep/2  + progress * this->tileDataNum], term2, copyParams); 
            // DataCopy(outGm[batch * iterStep + progress * this->tileDataNum], term1, this->processDataNum);
            // DataCopy(outGm[batch * iterStep + iterStep/2  + progress * this->tileDataNum], term2, this->processDataNum);

            outQueueXa.FreeTensor(term1);
            outQueueXb.FreeTensor(term2);
        }
    
        __aicore__ inline void process() {
            int32_t loopCount = this->tileNum;
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
                        copyOut(batch, i, len);
                    }
                }
            }
        }
    private:
        uint32_t totalLength,resLength,stride,iterStep, axesDim,coreBatch, padTimes;
        uint32_t tileDataNum, processDataNum,tailDataNum ,coreDataNum,tileNum;
        AscendC::TPipe *pipe;
        AscendC::GlobalTensor<T> gluOutGm;
        AscendC::GlobalTensor<T> xGm,vGm;
        AscendC::GlobalTensor<T> dxGm,dgradGluGm,yGradGm,gradXGm;
        AscendC::GlobalTensor<T> outGm;
        TQue<QuePosition::VECIN, BUFFER_NUM>inQueueB, inQueueA, inQueueDA, inQueueDB, inQueueGradxa,inQueueGradxb,inQueuedgrad_glu;
        TQue<QuePosition::VECOUT, BUFFER_NUM>outQueueXa,outQueueXb; 
        TBuf<QuePosition::VECCALC>QueueOnes, Queueds, Queued2s,QueueTerm2b,QueueTerm2a, QueueTerm1mid,Queuedb_neg_sig_b,Queueglu; 
        AscendC::LocalTensor<T> ones,ds, d2s, out, term1mid, term2a, term2b, db_neg_sig_b, glu;
};


template<typename T>class NoBcastFastHalf{
    public:
        __aicore__ inline NoBcastFastHalf(){}
        __aicore__ inline void Init(GM_ADDR grad_x,GM_ADDR y_grad, GM_ADDR x, GM_ADDR v_y, GM_ADDR v_x, GM_ADDR jvp_out,
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
            this->tileDataNum = tileDataNum;
            this->tailDataNum = tailDataNum;
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
            gradXGm.SetGlobalBuffer((__gm__ T*)grad_x + globalIndex*iterStep, this->coreBatch*iterStep);
            yGradGm.SetGlobalBuffer((__gm__ T*)y_grad + globalIndex*iterStep/2, this->coreBatch*iterStep/2);
            xGm.SetGlobalBuffer((__gm__ T*)x + globalIndex*iterStep, this->coreBatch*iterStep);
            dgradGluGm.SetGlobalBuffer((__gm__ T*)v_y + globalIndex*iterStep/2, this->coreBatch*iterStep/2);
            dxGm.SetGlobalBuffer((__gm__ T*)v_x + globalIndex*iterStep, this->coreBatch*iterStep);
            outGm.SetGlobalBuffer((__gm__ T*)jvp_out + globalIndex*iterStep, this->coreBatch*iterStep);

            pipe->InitBuffer(inQueueB,  BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(inQueueA,  BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(inQueueDA, BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(inQueueDB, BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(inQueueGradxa, BUFFER_NUM, this->tileDataNum * sizeof(T));
            // pipe->InitBuffer(inQueueGradxb, BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(outQueueXa,  BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(outQueueXb,  BUFFER_NUM, this->tileDataNum * sizeof(T));
            pipe->InitBuffer(inQueuedgrad_glu,  BUFFER_NUM, this->tileDataNum * sizeof(T));
            
            pipe->InitBuffer(QcastB, this->tileDataNum * sizeof(float));
            pipe->InitBuffer(QcastA, this->tileDataNum * sizeof(float));
            pipe->InitBuffer(QcastDA, this->tileDataNum * sizeof(float));
            pipe->InitBuffer(QcastDB, this->tileDataNum * sizeof(float));
            pipe->InitBuffer(QcastGradXa, this->tileDataNum * sizeof(float));
            // pipe->InitBuffer(QcastGradXb, this->tileDataNum * sizeof(float));
            pipe->InitBuffer(QcastXa, this->tileDataNum * sizeof(float));
            pipe->InitBuffer(QcastXb, this->tileDataNum * sizeof(float));
            pipe->InitBuffer(QcastGradGlu, this->tileDataNum * sizeof(float));
            pipe->InitBuffer(QCastdgrad_x_a, this->tileDataNum * sizeof(float));
            pipe->InitBuffer(QCastdgrad_x_b, this->tileDataNum * sizeof(float));   
            
            casta = QcastA.Get<float>();
            castb = QcastB.Get<float>();
            castda = QcastDA.Get<float>();
            castdb = QcastDB.Get<float>();
            castgradxa = QcastGradXa.Get<float>();
            // castgradxb = QcastGradXb.Get<float>();
            castxa = QcastXa.Get<float>();
            castxb = QcastXb.Get<float>();
            castdgrad_x_a = QCastdgrad_x_a.Get<float>();
            castdgrad_x_b = QCastdgrad_x_b.Get<float>();

            castdgrad_glu = QcastGradGlu.Get<float>();

            // Queuedb_neg_sig_b
            pipe->InitBuffer(Queuedb_neg_sig_b,     this->tileDataNum * sizeof(float));
            pipe->InitBuffer(QueueOnes,     this->tileDataNum * sizeof(float));
            pipe->InitBuffer(Queueglu,      this->tileDataNum * sizeof(float));

            ones = QueueOnes.Get<float>();
            float inputVal(1.0);
            AscendC::Duplicate<float>(ones, inputVal, this->tileDataNum);
            // ds = Queueds.Get<T>();
            // d2s = Queued2s.Get<T>();
            db_neg_sig_b = Queuedb_neg_sig_b.Get<float>();
            // term1mid = QueueTerm1mid.Get<float>();
            // term2a = QueueTerm2a.Get<float>();
            // term2b = QueueTerm2b.Get<float>();
            glu = Queueglu.Get<float>();
        }
    
        __aicore__ inline void copyIn(int batch, int progress) {
            LocalTensor<T> a  = inQueueA.AllocTensor<T>();
            LocalTensor<T> b = inQueueB.AllocTensor<T>();
            LocalTensor<T> da  = inQueueDA.AllocTensor<T>();
            LocalTensor<T> db = inQueueDB.AllocTensor<T>();
            LocalTensor<T> gradXa = inQueueGradxa.AllocTensor<T>();
            // LocalTensor<T> gradXb = inQueueGradxb.AllocTensor<T>();
            LocalTensor<T> dgrad_glu = inQueuedgrad_glu.AllocTensor<T>();
            
            // printf("iterStep: %d batch:%d this->tileDataNum:%d\n", iterStep, batch, this->tileDataNum);
            // LocalTensor<T> vyLocal = inQueueVY.AllocTensor<T>();
            // printf("input index1: %d\n", batch * iterStep + progress * this->tileDataNum);
            // printf("input index2: %d\n", batch * iterStep + iterStep/2 + progress * this->tileDataNum);
            DataCopy(a,  xGm[batch * iterStep + progress * this->tileDataNum], this->processDataNum);
            DataCopy(b, xGm[batch * iterStep + iterStep/2 + progress * this->tileDataNum], this->processDataNum);
            DataCopy(da,  dxGm[batch * iterStep + progress * this->tileDataNum], this->processDataNum);
            DataCopy(db, dxGm[batch * iterStep + iterStep/2  + progress * this->tileDataNum], this->processDataNum);
            DataCopy(gradXa,  gradXGm[batch * iterStep + progress * this->tileDataNum], this->processDataNum);
            // DataCopy(gradXb, gradXGm[batch * iterStep + iterStep/2  + progress * this->tileDataNum], this->processDataNum);
            DataCopy(dgrad_glu, dgradGluGm[batch * iterStep/2 + progress * this->tileDataNum], this->processDataNum);

            inQueueA.EnQue(a);
            inQueueB.EnQue(b);
            inQueueDA.EnQue(da);
            inQueueDB.EnQue(db);
            inQueueGradxa.EnQue(gradXa);
            // inQueueGradxb.EnQue(gradXb);
            inQueuedgrad_glu.EnQue(dgrad_glu);

        }
        __aicore__ inline void copyInPad(int batchIdx) {
            LocalTensor<T> a  = inQueueA.AllocTensor<T>();
            LocalTensor<T> b = inQueueB.AllocTensor<T>();
            LocalTensor<T> da  = inQueueDA.AllocTensor<T>();
            LocalTensor<T> db = inQueueDB.AllocTensor<T>();
            LocalTensor<T> gradXa = inQueueGradxa.AllocTensor<T>();
            // LocalTensor<T> gradXb = inQueueGradxb.AllocTensor<T>();
            LocalTensor<T> dgrad_glu = inQueuedgrad_glu.AllocTensor<T>();
            AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(padTimes), static_cast<uint32_t>(iterStep/2 * sizeof(T)), static_cast<uint32_t>(iterStep/2 * sizeof(T)), 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(a,  xGm[batchIdx * iterStep], copyParams, padParams); 
            AscendC::DataCopyPad(b,  xGm[batchIdx * iterStep + iterStep/2], copyParams, padParams); 
            AscendC::DataCopyPad(da,  dxGm[batchIdx * iterStep], copyParams, padParams); 
            AscendC::DataCopyPad(db,  dxGm[batchIdx * iterStep + iterStep/2], copyParams, padParams); 
            AscendC::DataCopyPad(gradXa,  gradXGm[batchIdx * iterStep], copyParams, padParams); 
            // AscendC::DataCopyPad(gradXb,  gradXGm[batchIdx * iterStep + iterStep/2], copyParams, padParams); 
            // printf("a iterStep/2:%d\n",iterStep/2);
            // for (int i = 0; i < padTimes; i++) {
            //     for (int j = 0; j < iterStep/2 ; j++) {
            //         printf("%f ",a.GetValue(i*iterStep/2 + j));
            //     }
            //     printf("\n");
            // }

            AscendC::DataCopyExtParams copyGParams{static_cast<uint16_t>(padTimes), static_cast<uint32_t>(iterStep/2 * sizeof(T)),0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
            AscendC::DataCopyPad(dgrad_glu,  dgradGluGm[batchIdx * iterStep/2], copyGParams, padParams); 

            inQueueA.EnQue(a);
            inQueueB.EnQue(b);
            inQueueDA.EnQue(da);
            inQueueDB.EnQue(db);
            inQueueGradxa.EnQue(gradXa);
            // inQueueGradxb.EnQue(gradXb);
            inQueuedgrad_glu.EnQue(dgrad_glu);
        }
        __aicore__ inline void computePad(int batch) {
            // LocalTensor<T>vLocal = inQueueV.DeQue<T>();
            LocalTensor<T>dgrad_x_a = outQueueXa.AllocTensor<T>();
            LocalTensor<T>dgrad_x_b = outQueueXb.AllocTensor<T>();
            // LocalTensor<T>sigRightLocal = inQueueInputSigRight.AllocTensor<T>();
            float two = -2;
            float ooo = 1;
            processDataNum = padTimes* (iterStep/2*sizeof(T) + 31)*32/32/sizeof(T);

            LocalTensor<T>a = inQueueA.DeQue<T>();
            LocalTensor<T>b = inQueueB.DeQue<T>();
            LocalTensor<T>da = inQueueDA.DeQue<T>();
            LocalTensor<T>db = inQueueDB.DeQue<T>();
            LocalTensor<T>gradxa = inQueueGradxa.DeQue<T>();
            // LocalTensor<T>gradxb = inQueueGradxb.DeQue<T>();
            LocalTensor<T>dgrad_glu =  inQueuedgrad_glu.DeQue<T>();
            
            Cast(casta, a, AscendC::RoundMode::CAST_NONE, processDataNum);
            Cast(castb ,b,  AscendC::RoundMode::CAST_NONE, processDataNum);
            Cast(castda, da, AscendC::RoundMode::CAST_NONE, processDataNum);
            Cast(castdb, db, AscendC::RoundMode::CAST_NONE, processDataNum);
            Cast(castgradxa , gradxa,  AscendC::RoundMode::CAST_NONE, processDataNum);
            // Cast(castgradxb   ,gradxb, AscendC::RoundMode::CAST_NONE, processDataNum);
            Cast(castdgrad_glu, dgrad_glu, AscendC::RoundMode::CAST_NONE, processDataNum);
            
            // LocalTensor<T>v_y = inQueueVY.DeQue<T>();
            // T scalar = -1;
            float scalar = -1;

            // printf("a\n");
            // for (int i = 0; i < processDataNum; i++) {
            //     printf("%f ", a.GetValue(i));
            // }
            // printf("\n");
            // AscendC::Div(smallGlu, smallGlu, smalla, processDataNum);
            AscendC::Muls(castb, castb, scalar, processDataNum);
            AscendC::Exp(castb, castb, processDataNum);
            AscendC::Adds(castb, castb, ooo, processDataNum);
            AscendC::Div(castb, ones, castb, processDataNum);
            LocalTensor<float>sig_b = castb; // sigb
            AscendC::Mul(glu, casta, sig_b, processDataNum); // glu
            // printf("sig_b\n");
            // for (int i = 0; i < processDataNum; i++) {
            //     printf("%f ", sig_b.GetValue(i));
            // }
            // printf("\n");
            AscendC::Mul(db_neg_sig_b, castdb, sig_b, processDataNum);
            AscendC::Sub(db_neg_sig_b, castdb, db_neg_sig_b, processDataNum);
            
            // AscendC::Mul(castdgrad_glu, castdgrad_glu, sig_b, processDataNum);
            // AscendC::Mul(castdgrad_x_a, castgradxa,  db_neg_sig_b, processDataNum);
            // AscendC::Add(castdgrad_x_a, castdgrad_glu,  castdgrad_x_a, processDataNum);
            AscendC::Mul(castdgrad_x_a, castgradxa,  db_neg_sig_b, processDataNum);
            AscendC::MulAddDst(castdgrad_x_a, castdgrad_glu, sig_b, processDataNum);

            
            // printf("castdgrad_x_a\n");
            // for (int i = 0; i < processDataNum; i++) {
            //     printf("%f ", castdgrad_x_a.GetValue(i));
            // }
            // printf("\n");
            AscendC::Sub(castdgrad_x_b, casta,  glu, processDataNum);
            AscendC::Mul(castdgrad_x_b, castdgrad_x_a,  castdgrad_x_b, processDataNum);
            AscendC::Mul(casta, castda, sig_b, processDataNum);
            AscendC::Mul(castb, glu, db_neg_sig_b, processDataNum);
            AscendC::Sub(castda, castda,  casta, processDataNum);
            AscendC::Sub(castda, castda,  castb, processDataNum);
            // AscendC::Mul(casta, castgradxa, castda, processDataNum);
            // AscendC::Add(castdgrad_x_b, castdgrad_x_b, casta, processDataNum);
            AscendC::MulAddDst(castdgrad_x_b, castgradxa, castda, processDataNum);

            // printf("castdgrad_x_b\n");
            // for (int i = 0; i <processDataNum; i++) {
            //     printf("%f ", castdgrad_x_b.GetValue(i));
            // }
            // printf("\n");
            Cast(dgrad_x_a, castdgrad_x_a, AscendC::RoundMode::CAST_ROUND, processDataNum);
            Cast(dgrad_x_b, castdgrad_x_b, AscendC::RoundMode::CAST_ROUND, processDataNum);
            inQueueA.FreeTensor(a);
            inQueueB.FreeTensor(b);
            inQueueDA.FreeTensor(da);
            inQueueDB.FreeTensor(db);
            inQueueGradxa.FreeTensor(gradxa);
            // inQueueGradxb.FreeTensor(gradxb);
            inQueuedgrad_glu.FreeTensor(dgrad_glu);

            outQueueXa.EnQue(dgrad_x_a);
            outQueueXb.EnQue(dgrad_x_b);
        }
        __aicore__ inline void compute(int batch,int progress) {
            // LocalTensor<T>vLocal = inQueueV.DeQue<T>();
            LocalTensor<T>dgrad_x_a = outQueueXa.AllocTensor<T>();
            LocalTensor<T>dgrad_x_b = outQueueXb.AllocTensor<T>();
            // LocalTensor<T>sigRightLocal = inQueueInputSigRight.AllocTensor<T>();
            float two = -2;
            float ooo = 1;

            LocalTensor<T>a = inQueueA.DeQue<T>();
            LocalTensor<T>b = inQueueB.DeQue<T>();
            LocalTensor<T>da = inQueueDA.DeQue<T>();
            LocalTensor<T>db = inQueueDB.DeQue<T>();
            LocalTensor<T>gradxa = inQueueGradxa.DeQue<T>();
            // LocalTensor<T>gradxb = inQueueGradxb.DeQue<T>();
            LocalTensor<T>dgrad_glu =  inQueuedgrad_glu.DeQue<T>();
            
            Cast(casta, a, AscendC::RoundMode::CAST_NONE, processDataNum);
            Cast(castb ,b,  AscendC::RoundMode::CAST_NONE, processDataNum);
            Cast(castda, da, AscendC::RoundMode::CAST_NONE, processDataNum);
            Cast(castdb, db, AscendC::RoundMode::CAST_NONE, processDataNum);
            Cast(castgradxa , gradxa,  AscendC::RoundMode::CAST_NONE, processDataNum);
            // Cast(castgradxb   ,gradxb, AscendC::RoundMode::CAST_NONE, processDataNum);
            Cast(castdgrad_glu, dgrad_glu, AscendC::RoundMode::CAST_NONE, processDataNum);
            
            // LocalTensor<T>v_y = inQueueVY.DeQue<T>();
            // T scalar = -1;
            float scalar = -1;

            // printf("a\n");
            // for (int i = 0; i < processDataNum; i++) {
            //     printf("%f ", a.GetValue(i));
            // }
            // printf("\n");
            // AscendC::Div(smallGlu, smallGlu, smalla, processDataNum);
            AscendC::Muls(castb, castb, scalar, processDataNum);
            AscendC::Exp(castb, castb, processDataNum);
            AscendC::Adds(castb, castb, ooo, processDataNum);
            AscendC::Div(castb, ones, castb, processDataNum);
            LocalTensor<float>sig_b = castb; // sigb
            AscendC::Mul(glu, casta, sig_b, processDataNum); // glu
            // printf("sig_b\n");
            // for (int i = 0; i < processDataNum; i++) {
            //     printf("%f ", sig_b.GetValue(i));
            // }
            // printf("\n");
            AscendC::Mul(db_neg_sig_b, castdb, sig_b, processDataNum);
            AscendC::Sub(db_neg_sig_b, castdb, db_neg_sig_b, processDataNum);
            
            // AscendC::Mul(castdgrad_glu, castdgrad_glu, sig_b, processDataNum);
            // AscendC::Mul(castdgrad_x_a, castgradxa,  db_neg_sig_b, processDataNum);
            // AscendC::Add(castdgrad_x_a, castdgrad_glu,  castdgrad_x_a, processDataNum);
            AscendC::Mul(castdgrad_x_a, castgradxa,  db_neg_sig_b, processDataNum);
            AscendC::MulAddDst(castdgrad_x_a, castdgrad_glu, sig_b, processDataNum);
            // printf("castdgrad_x_a\n");
            // for (int i = 0; i < processDataNum; i++) {
            //     printf("%f ", castdgrad_x_a.GetValue(i));
            // }
            // printf("\n");
            AscendC::Sub(castdgrad_x_b, casta,  glu, processDataNum);
            AscendC::Mul(castdgrad_x_b, castdgrad_x_a,  castdgrad_x_b, processDataNum);
            AscendC::Mul(casta, castda, sig_b, processDataNum);
            AscendC::Mul(castb, glu, db_neg_sig_b, processDataNum);
            AscendC::Sub(castda, castda,  casta, processDataNum);
            AscendC::Sub(castda, castda,  castb, processDataNum);
            // AscendC::Mul(casta, castgradxa, castda, processDataNum);
            // AscendC::Add(castdgrad_x_b, castdgrad_x_b, casta, processDataNum);
                // AscendC::Mul(casta, castgradxa, castda, processDataNum);
            // AscendC::Add(castdgrad_x_b, castdgrad_x_b, casta, processDataNum);
            AscendC::MulAddDst(castdgrad_x_b, castgradxa, castda, processDataNum);
            // printf("castdgrad_x_b\n");
            // for (int i = 0; i <processDataNum; i++) {
            //     printf("%f ", castdgrad_x_b.GetValue(i));
            // }
            // printf("\n");
            Cast(dgrad_x_a, castdgrad_x_a, AscendC::RoundMode::CAST_ROUND, processDataNum);
            Cast(dgrad_x_b, castdgrad_x_b, AscendC::RoundMode::CAST_ROUND, processDataNum);
            inQueueA.FreeTensor(a);
            inQueueB.FreeTensor(b);
            inQueueDA.FreeTensor(da);
            inQueueDB.FreeTensor(db);
            inQueueGradxa.FreeTensor(gradxa);
            // inQueueGradxb.FreeTensor(gradxb);
            inQueuedgrad_glu.FreeTensor(dgrad_glu);

            outQueueXa.EnQue(dgrad_x_a);
            outQueueXb.EnQue(dgrad_x_b);
        }
        __aicore__ inline void copyOutPad(int batch) {
            // printf("out index: %d\n", batch * iterStep/2 + progress * this->tileDataNum);
            auto term1 = outQueueXa.DeQue<T>();
            auto term2 = outQueueXb.DeQue<T>();
            // printf("term2 iterStep/2:%d\n",iterStep/2);
            // for (int i = 0; i < padTimes; i++) {
            //     for (int j = 0; j < iterStep/2 ; j++) {
            //         printf("%f ",term2.GetValue(i*iterStep/2 + j));
            //     }
            //     printf("\n");
            // }

            // printf("outindex1: %d\n",batch * iterStep + progress * this->tileDataNum);
            // printf("outindex2: %d\n",batch * iterStep + iterStep/2  + progress * this->tileDataNum);
            AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(padTimes), static_cast<uint32_t>(iterStep/2 * sizeof(T)), 0, static_cast<uint32_t>(iterStep/2 * sizeof(T)), 0};
            // printf("len :%d\n",len);
            AscendC::DataCopyPad(outGm[batch * iterStep], term1, copyParams); 
            AscendC::DataCopyPad(outGm[batch * iterStep + iterStep/2], term2, copyParams); 
            // DataCopy(outGm[batch * iterStep + progress * this->tileDataNum], term1, this->processDataNum);
            // DataCopy(outGm[batch * iterStep + iterStep/2  + progress * this->tileDataNum], term2, this->processDataNum);

            outQueueXa.FreeTensor(term1);
            outQueueXb.FreeTensor(term2);
        }
    
        __aicore__ inline void copyOut(int batch, int progress) {
            // printf("out index: %d\n", batch * iterStep/2 + progress * this->tileDataNum);
            auto term1 = outQueueXa.DeQue<T>();
            auto term2 = outQueueXb.DeQue<T>();

            DataCopy(outGm[batch * iterStep + progress * this->tileDataNum], term1, this->processDataNum);

            DataCopy(outGm[batch * iterStep + iterStep/2  + progress * this->tileDataNum], term2, this->processDataNum);

            outQueueXa.FreeTensor(term1);
            outQueueXb.FreeTensor(term2);
        }
        __aicore__ inline void process() {
            int32_t loopCount = this->tileNum;
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
                int32_t loopCount = this->tileNum;
                // printf("coreBatch:%d loopCount:%d tailDataNum:%d\n", coreBatch, loopCount, tailDataNum);
                for (int batch = 0; batch < coreBatch; batch++) {
                    this->processDataNum = this->tileDataNum;
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
        }
        // __aicore__ inline void process() {
        //     int32_t loopCount = this->tileNum;
        //     // printf("coreBatch:%d loopCount:%d tailDataNum:%d\n", coreBatch, loopCount, tailDataNum);
        //     for (int batch = 0; batch < coreBatch; batch++) {
        //         this->processDataNum = this->tileDataNum;
        //         for (int32_t i = 0; i < loopCount; i++) {
        //             if (i == this->tileNum - 1) {
        //             this->processDataNum = this->tailDataNum;
        //             }
        //             copyIn(batch, i);
        //             compute(batch, i);
        //             copyOut(batch, i);
        //         }
        //     }
        // }
    private:
        uint32_t totalLength,resLength,stride,iterStep, axesDim,coreBatch,padTimes;
        uint32_t tileDataNum, processDataNum,tailDataNum ,coreDataNum,tileNum;
        AscendC::TPipe *pipe;
        AscendC::GlobalTensor<T> gluOutGm;
        AscendC::GlobalTensor<T> xGm,vGm;
        AscendC::GlobalTensor<T> dxGm,dgradGluGm,yGradGm,gradXGm;
        AscendC::GlobalTensor<T> outGm;
        TQue<QuePosition::VECIN, BUFFER_NUM>inQueueB, inQueueA, inQueueDA, inQueueDB, inQueueGradxa,inQueueGradxb,inQueuedgrad_glu;
        TQue<QuePosition::VECOUT, BUFFER_NUM>outQueueXa,outQueueXb; 
        TBuf<QuePosition::VECCALC>QueueOnes, Queueds, Queued2s,QueueTerm2b,QueueTerm2a, QueueTerm1mid,Queuedb_neg_sig_b,Queueglu, QcastA, QcastB,QcastDA, QcastDB, QcastGradXa, QcastGradXb, QcastXa, QcastXb, QcastGradGlu,QCastdgrad_x_a,QCastdgrad_x_b; 
        AscendC::LocalTensor<float> ones, term1mid, term2a, term2b, db_neg_sig_b, glu, casta, castb, castda, castdb, castxa, castxb, castgradxa, castgradxb, castdgrad_glu,castdgrad_x_b,castdgrad_x_a;
};

