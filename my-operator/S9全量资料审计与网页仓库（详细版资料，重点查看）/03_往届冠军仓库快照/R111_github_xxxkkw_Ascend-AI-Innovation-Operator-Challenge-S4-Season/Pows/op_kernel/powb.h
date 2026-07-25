#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelPowsBroadCast{  // 非对齐 广播case
    public:
        __aicore__ inline KernelPowsBroadCast(){}
        __aicore__ inline void InitTiling(GM_ADDR tiling) {
            GET_TILING_DATA(tiling_data, tiling); 
            y_dimensional = tiling_data.y_dimensional;
            y_ndarray = tiling_data.y_ndarray;
            x1_ndarray = tiling_data.x1_ndarray;
            x2_ndarray = tiling_data.x2_ndarray;
            y_sumndarray = tiling_data.y_sumndarray;
            x1_sumndarray = tiling_data.x1_sumndarray;
            x2_sumndarray = tiling_data.x2_sumndarray;
            x1TotalLength = tiling_data.x1TotalLength;
            x2TotalLength = tiling_data.x2TotalLength;
            x1Size = tiling_data.x1Size;
            x2Size = tiling_data.x2Size;
            totalLength = tiling_data.totalLength;
            tileLength = tiling_data.tileLength;
            loopCount = tiling_data.loopCount;
            leftNum = tiling_data.leftNum;
    
        }
        __aicore__ inline void Init(GM_ADDR x1,GM_ADDR x2, GM_ADDR y,GM_ADDR tiling,TPipe* pipeIn){
            InitTiling(tiling);
            ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
            x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1*)x1, x1TotalLength);  
            x2Gm.SetGlobalBuffer((__gm__ DTYPE_X2*)x2, x2TotalLength); 
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, totalLength);
            pipe = pipeIn;
            pipe->InitBuffer(tmpBufferX1, this->tileLength * sizeof(DTYPE_X1));
            pipe->InitBuffer(tmpBufferX2, this->tileLength * sizeof(DTYPE_X2));
            pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
            if constexpr (std::is_same_v<DTYPE_X1, bfloat16_t> || std::is_same_v<DTYPE_X1, half>){
                pipe->InitBuffer(tmpBuffer1, this->tileLength * sizeof(float));
                pipe->InitBuffer(tmpBuffer2, this->tileLength * sizeof(float));
                pipe->InitBuffer(tmpBuffer3, this->tileLength * sizeof(float));
            }
        }
    
        __aicore__ inline void Process() {
            LocalTensor<DTYPE_X1> x1Local = tmpBufferX1.Get<DTYPE_X1>();
            LocalTensor<DTYPE_X2> x2Local = tmpBufferX2.Get<DTYPE_X2>();
            for (uint32_t i = 0; i < this->loopCount; i++) {                   
                Compute(i, this->tileLength,x1Local,x2Local);           
                CopyOut(i, this->tileLength);
            }
            if (this->leftNum > 0) {
                Compute(this->loopCount, this->leftNum,x1Local,x2Local);
                CopyOut(this->loopCount, this->leftNum);
            }
        }
    private:
        __aicore__ inline void Compute(int32_t progress, uint32_t length,LocalTensor<DTYPE_X1> x1Local,LocalTensor<DTYPE_X2> x2Local) { 
            LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
            for(uint32_t j = 0;j < length;j ++){
                uint32_t x1_start = 0;
                uint32_t x2_start = 0;
                uint32_t index = j + progress * this->tileLength;
                for (uint32_t k = 0; k < this->y_dimensional; k++){
                    if (this->x1_ndarray[k] != 1){
                        x1_start += this->x1_sumndarray[k] * (index / this->y_sumndarray[k] % this->y_ndarray[k]);
                    }
                    if(this->x2_ndarray[k] != 1){
                        x2_start += this->x2_sumndarray[k] * (index / this->y_sumndarray[k] % this->y_ndarray[k]);  
                    }
                }
                auto x1 = x1Gm.GetValue(x1_start); 
                auto x2 = x2Gm.GetValue(x2_start);
                x1Local.SetValue(j,x1);
                x2Local.SetValue(j,x2);
            }
            if constexpr (std::is_same_v<DTYPE_X1, float>){
                Cmpfp32(x1Local,x2Local,yLocal,length); 
            }
            else if constexpr (std::is_same_v<DTYPE_X1, bfloat16_t> || std::is_same_v<DTYPE_X1, half>){
                Cmp(x1Local,x2Local,yLocal,length);
            }
            outQueueY.EnQue<DTYPE_Y>(yLocal);
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
        __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {  //原始地址 输出
            LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
            DataCopy(yGm[progress * this->tileLength], yLocal, length);
            outQueueY.FreeTensor(yLocal);
        }
    
    private:
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
        
        GlobalTensor<DTYPE_X1> x1Gm;
        GlobalTensor<DTYPE_X2> x2Gm;
        GlobalTensor<DTYPE_Y> yGm;
        TBuf<QuePosition::VECCALC> tmpBuffer1;
        TBuf<QuePosition::VECCALC> tmpBuffer2;
        TBuf<QuePosition::VECCALC> tmpBuffer3;
        TBuf<QuePosition::VECCALC> tmpBufferX1;
        TBuf<QuePosition::VECCALC> tmpBufferX2;
    
        TPipe* pipe;
        uint64_t totalLength;
        uint64_t tileLength;
        uint64_t loopCount;
        uint64_t leftNum;
        uint32_t y_dimensional;
        uint32_t *x1_sumndarray;
        uint32_t *x2_sumndarray;
        uint32_t *y_ndarray;
        uint32_t *x1_ndarray;
        uint32_t *x2_ndarray;
        uint32_t *y_sumndarray;
        uint32_t x1TotalLength;
        uint32_t x2TotalLength;
        uint32_t x1Size;
        uint32_t x2Size;
    };