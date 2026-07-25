#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template<typename T> class KernalTrunc{
public:
    __aicore__ inline KernalTrunc() {}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output_y, uint32_t totalLength, 
                            uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain){
        
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        Gm_x.SetGlobalBuffer((__gm__ T*)input_x + startPointer, bufferlength);
        Gm_y.SetGlobalBuffer((__gm__ T*)output_y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(T));
        
        if constexpr (std::is_same_v<T, float>){
            pipe.InitBuffer(b_tmp1, this->tileLength * sizeof(int64_t));
        }
        else if constexpr (std::is_same_v<T, half>){
            pipe.InitBuffer(b_tmp1, this->tileLength * sizeof(int32_t));
        }
        else if constexpr (std::is_same_v<T, bfloat16_t>){
            pipe.InitBuffer(b_tmp1, this->tileLength * sizeof(int32_t));
            pipe.InitBuffer(b_tmp2, this->tileLength * sizeof(float));
        }
        else{
            pipe.InitBuffer(b_tmp1, this->tileLength * sizeof(T));
        }
        
    }

    __aicore__ inline void Process(){
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, (length + 31) / 32 * 32);
    }
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length){
        LocalTensor<T> xLocal = Q_x.AllocTensor<T>();
        DataCopy(xLocal, Gm_x[progress * this->tileLength], length);
        Q_x.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress, uint32_t length){
        LocalTensor<T> x = Q_x.DeQue<T>();
        LocalTensor<T> y = Q_y.AllocTensor<T>();
        if constexpr (std::is_same_v<T, int8_t>){
            auto int8x = b_tmp1.Get<int8_t>();
            DataCopy(int8x, x, length);
            DataCopy(y, int8x, length);
        }
        else if constexpr (std::is_same_v<T, uint8_t>){
            auto uint8x = b_tmp1.Get<uint8_t>();
            DataCopy(uint8x, x, length);
            DataCopy(y, uint8x, length);
        }
        else if constexpr (std::is_same_v<T, int32_t>){
            auto int32x = b_tmp1.Get<int32_t>();
            DataCopy(int32x, x, length);
            DataCopy(y, int32x, length);
        }
        else if constexpr (std::is_same_v<T, float>){
            auto int64x = b_tmp1.Get<int64_t>();
            Cast(int64x, x, AscendC::RoundMode::CAST_TRUNC, length);
            Cast(y, int64x, AscendC::RoundMode::CAST_RINT, length);
        }
        else if constexpr (std::is_same_v<T, half>){
            auto int32x = b_tmp1.Get<int32_t>();
            Cast(int32x, x, AscendC::RoundMode::CAST_TRUNC, length);
            half scale = 1.0;
            SetDeqScale(scale);
            Cast(y, int32x, AscendC::RoundMode::CAST_RINT, length);

        }
        else{ // bf16
            auto int32x = b_tmp1.Get<int32_t>();
            auto floatx = b_tmp1.Get<float>();
            Cast(int32x, x, AscendC::RoundMode::CAST_TRUNC, length);
            Cast(floatx, int32x, AscendC::RoundMode::CAST_RINT, length);
            Cast(y, floatx, AscendC::RoundMode::CAST_RINT, length);
        }
        Q_x.FreeTensor(x);
        Q_y.EnQue<T>(y);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length){
        LocalTensor<T> y = Q_y.DeQue<T>();
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;

    TBuf<QuePosition::VECCALC> b_tmp1, b_tmp2;

    GlobalTensor<T> Gm_x;
    GlobalTensor<T> Gm_y;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;

};


extern "C" __global__ __aicore__ void trunc(GM_ADDR input_x, GM_ADDR output_y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernalTrunc<DTYPE_INPUT_X> op;
    op.Init(input_x, output_y, tiling_data.totalLength, tiling_data.ALIGN_NUM,
            tiling_data.block_size, tiling_data.core_size,
            tiling_data.core_remain);
    op.Process();
}