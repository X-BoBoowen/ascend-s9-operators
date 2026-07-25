#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template<typename TYPE_X1, typename TYPE_Y>
class KernelGelu {
    using T = TYPE_X1;
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,  
                                uint32_t block_size,
                                uint32_t core_size, uint32_t tile_num, uint32_t final_length, TPipe* pipeIn) { 
        this->blockLength = core_size;
        this->tileLength = block_size;
        this->final_length = final_length;
        this->pipe = pipeIn;
 
        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ TYPE_X1*)x , bufferlength);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y , bufferlength);

        this->tileNum = tile_num;
        pipe->InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        if constexpr(std::is_same_v<T,bfloat16_t>) {
            pipe->InitBuffer(B_int32buffer, this->tileLength * sizeof(int32_t));
            pipe->InitBuffer(B_fp32buffer, this->tileLength * sizeof(int32_t));
        } else if constexpr(std::is_same_v<T,half>) {
            pipe->InitBuffer(B_int32buffer, this->tileLength * sizeof(int32_t));
            pipe->InitBuffer(B_fp32buffer, this->tileLength * sizeof(int32_t));
        } else if constexpr(std::is_same_v<T,float>) { 
            pipe->InitBuffer(B_int32buffer, this->tileLength * sizeof(int32_t));
            pipe->InitBuffer(B_fp32buffer, this->tileLength * sizeof(int32_t)); 
            pipe->InitBuffer(B_cmpBuffer, this->tileLength * sizeof(uint8_t));
        } else if constexpr(std::is_same_v<T,int8_t> || std::is_same_v<T,uint8_t>) {
            pipe->InitBuffer(B_int16buffer, this->tileLength * sizeof(int16_t));
        }
 
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum - 1; 
        for (int32_t i = 0; i < loopCount; i+= 2) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
            CopyIn(i + 1, this->tileLength);
            Compute(i + 1, this->tileLength);
            CopyOut(i + 1, this->tileLength);
        }
        if (loopCount % 2) {
            CopyIn(loopCount - 1, this->tileLength);
            Compute(loopCount - 1, this->tileLength);
            CopyOut(loopCount - 1, this->tileLength);
        }
        CopyIn(loopCount, this->final_length);
        Compute(loopCount, this->final_length);
        CopyOut(loopCount, this->final_length);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> xLocal = inQueueX.AllocTensor<TYPE_X1>();
        DataCopy(xLocal, xGm[progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> xLocal = inQueueX.DeQue<TYPE_X1>();
        LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();
        
        if constexpr(std::is_same_v<T,int8_t> || std::is_same_v<T,uint8_t>) {
            auto fp16_tmp = B_int16buffer.Get<half>();
            Cast(fp16_tmp, xLocal, RoundMode::CAST_NONE, length);
            Cast(yLocal, fp16_tmp, RoundMode::CAST_TRUNC, length); 
        } else if constexpr(std::is_same_v<T,half>) {
            auto int32_tmp = B_int32buffer.Get<int32_t>();
            auto fp32_tmp = B_fp32buffer.Get<float>();
            Cast(int32_tmp, xLocal, RoundMode::CAST_TRUNC, length);
            Cast(fp32_tmp, int32_tmp, RoundMode::CAST_NONE, length);
            Cast(yLocal, fp32_tmp, RoundMode::CAST_NONE, length);
 
        } else if constexpr(std::is_same_v<T,float>){   
          
            Cast(yLocal, xLocal, RoundMode::CAST_TRUNC, length);
  

        } else if constexpr(std::is_same_v<T,int32_t>){ 
            Adds(yLocal, xLocal, 0, length);
        } else if constexpr(std::is_same_v<T,bfloat16_t>) {
            auto int32_tmp = B_int32buffer.Get<int32_t>();
            auto fp32_tmp = B_fp32buffer.Get<float>();
            Cast(int32_tmp, xLocal, RoundMode::CAST_TRUNC, length);
            Cast(fp32_tmp, int32_tmp, RoundMode::CAST_TRUNC, length);
            Cast(yLocal, fp32_tmp, RoundMode::CAST_TRUNC, length);
        }

        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
 
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        DataCopy(yGm[progress * this->tileLength], yLocal, length); 
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe* pipe;
    TBuf<QuePosition::VECCALC> B_int32buffer, B_int16buffer, B_fp32buffer, B_cmpBuffer;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<TYPE_X1> xGm;
    GlobalTensor<TYPE_Y> yGm;  
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t final_length;
};
  

extern "C" __global__ __aicore__ void trunc(GM_ADDR input_x, GM_ADDR output_y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe; 
    KernelGelu<DTYPE_INPUT_X,DTYPE_OUTPUT_Y> op;
    op.Init(input_x, output_y,tiling_data.block_size, tiling_data.core_size, tiling_data.tile_num, tiling_data.final_length,&pipe);
    op.Process();
      
}



 
