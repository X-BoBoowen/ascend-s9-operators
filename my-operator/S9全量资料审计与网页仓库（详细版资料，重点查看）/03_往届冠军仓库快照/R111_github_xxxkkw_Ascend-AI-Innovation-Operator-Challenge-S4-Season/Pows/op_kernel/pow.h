#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelPows{ // tiling key 1 正常情况不用广播
    public:
        __aicore__ inline KernelPows(){}
        __aicore__ inline void InitTiling(GM_ADDR tiling) {
            GET_TILING_DATA(tiling_data, tiling); 
            totalLength = tiling_data.totalLength;
            tileLength = tiling_data.tileLength;
            loopCount = tiling_data.loopCount;
            leftNum = tiling_data.leftNum;
        }
        
        __aicore__ inline void Init(GM_ADDR x1,GM_ADDR x2, GM_ADDR y,GM_ADDR tiling,TPipe* pipeIn){
            InitTiling(tiling);
    
            ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
            
            x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1*)x1,this->totalLength);
            x2Gm.SetGlobalBuffer((__gm__ DTYPE_X1*)x2,this->totalLength);
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y,this->totalLength);
            pipe = pipeIn;
            pipe->InitBuffer(inQueueX1, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X1));
            pipe->InitBuffer(inQueueX2, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X2));
            pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
            if constexpr (std::is_same_v<DTYPE_X1, bfloat16_t> || std::is_same_v<DTYPE_X1, half>){
                pipe->InitBuffer(tmpBuffer1, this->tileLength * sizeof(float));
                pipe->InitBuffer(tmpBuffer2, this->tileLength * sizeof(float));
                pipe->InitBuffer(tmpBuffer3, this->tileLength * sizeof(float));
            }
        }
        __aicore__ inline void Process() {
            for (int32_t i = 0; i < this->loopCount; i++) {
                CopyIn(i,  this->tileLength);       // 32B对齐后的长度                                
                Compute(i, this->tileLength);           
                CopyOut(i, this->tileLength);
            }
            if (this->leftNum > 0) {
                CopyIn(this->loopCount, this->leftNum);
                Compute(this->loopCount, this->leftNum);
                CopyOut(this->loopCount, this->leftNum);
            }
        }
    private:
        __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
            LocalTensor<DTYPE_X1> x1Local = inQueueX1.AllocTensor<DTYPE_X1>();
            LocalTensor<DTYPE_X2> x2Local = inQueueX2.AllocTensor<DTYPE_X2>();
            DataCopy(x1Local, x1Gm[progress * this->tileLength], length);
            DataCopy(x2Local, x2Gm[progress * this->tileLength], length);
            inQueueX1.EnQue(x1Local);
            inQueueX2.EnQue(x2Local);
        }
        
        __aicore__ inline void Compute(int32_t progress, uint32_t length) {
            LocalTensor<DTYPE_X1> x1Local = inQueueX1.DeQue<DTYPE_X1>();
            LocalTensor<DTYPE_X2> x2Local = inQueueX2.DeQue<DTYPE_X2>();
            LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
            if constexpr (std::is_same_v<DTYPE_X1, float>){
                Cmpfp32(x1Local,x2Local,yLocal,length);
            }
            else if constexpr (std::is_same_v<DTYPE_X1, bfloat16_t> || std::is_same_v<DTYPE_X1, half>){
                Cmp(x1Local,x2Local,yLocal,length);
            }
            outQueueY.EnQue<DTYPE_Y>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }
        __aicore__ inline void Cmpfp32(LocalTensor<DTYPE_X1> x1Local,LocalTensor<DTYPE_X2> x2Local,LocalTensor<DTYPE_Y> yLocal,uint32_t length){
            Ln(x1Local,x1Local,length);
            Mul(yLocal,x1Local,x2Local,length);
            Exp(yLocal,yLocal,length);
        }
    
        __aicore__ inline void Cmp(LocalTensor<DTYPE_X1> x1Local,LocalTensor<DTYPE_X2> x2Local,LocalTensor<DTYPE_Y> yLocal,uint32_t length){
            LocalTensor<float> x1TmpLocal = tmpBuffer1.Get<float>();
            LocalTensor<float> x2TmpLocal = tmpBuffer2.Get<float>();
            LocalTensor<float> yTmpLocal = tmpBuffer3.Get<float>();
            Cast(x1TmpLocal,x1Local,RoundMode::CAST_NONE,length);
            Cast(x2TmpLocal,x2Local,RoundMode::CAST_NONE,length);
            Ln(x1TmpLocal,x1TmpLocal,length);
            Mul(yTmpLocal,x1TmpLocal,x2TmpLocal,length);
            Exp(yTmpLocal,yTmpLocal,length);
            Cast(yLocal,yTmpLocal,RoundMode::CAST_RINT,length);
        }
        __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
            LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
            DataCopy(yGm[progress * this->tileLength], yLocal, length);
            outQueueY.FreeTensor(yLocal);
        }
        
    private:
        //一些类对象的初始化
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX2;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
        
        GlobalTensor<DTYPE_X1> x1Gm;
        GlobalTensor<DTYPE_X2> x2Gm;
        GlobalTensor<DTYPE_Y> yGm;
        TBuf<QuePosition::VECCALC> tmpBuffer1;
        TBuf<QuePosition::VECCALC> tmpBuffer2;
        TBuf<QuePosition::VECCALC> tmpBuffer3;
        
        TPipe* pipe;
        uint64_t totalLength;
        uint64_t tileLength;
        uint64_t loopCount;
        uint64_t leftNum;
    };