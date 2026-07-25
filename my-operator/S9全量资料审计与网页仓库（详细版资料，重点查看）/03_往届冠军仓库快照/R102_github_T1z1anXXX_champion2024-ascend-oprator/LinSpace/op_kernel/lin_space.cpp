#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template<typename T> class KernalLinSpace{
public:
    __aicore__ inline KernalLinSpace() {}
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR stop, GM_ADDR num_axes, GM_ADDR output, float start_val, float stop_val, uint32_t totalLength, 
                            uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain){
        
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        this->f_start_val = start_val;
        this->f_stop_val = stop_val;
        this->totalLength = totalLength;

        Gm_start.SetGlobalBuffer((__gm__ T*)start + startPointer, 1);
        Gm_stop.SetGlobalBuffer((__gm__ T*)stop + startPointer, 1);
        Gm_num_axes.SetGlobalBuffer((__gm__ int32_t*)num_axes + startPointer, 1);
        Gm_output.SetGlobalBuffer((__gm__ T*)output + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        if constexpr (std::is_same_v<T, int16_t>){
            pipe.InitBuffer(b_tmp1, 1*sizeof(T));
        }
        else if constexpr (std::is_same_v<T, int32_t>){
            pipe.InitBuffer(b_tmp1, 1*sizeof(T));
        }
        else if constexpr (std::is_same_v<T, int8_t>){
            pipe.InitBuffer(b_tmp1, 1*sizeof(T));
        }
        // else if constexpr (std::is_same_v<T, uint8_t>){
        //     pipe.InitBuffer(b_tmp1, 1*sizeof(T));
        // }
        
        if constexpr (std::is_same_v<T, int8_t>){
            pipe.InitBuffer(b_tmp2, 1*sizeof(half));
        }
        else if constexpr (std::is_same_v<T, uint8_t>){
            pipe.InitBuffer(q_tmp, BUFFER_NUM, 1*sizeof(uint8_t));
            pipe.InitBuffer(q_tmp2, BUFFER_NUM, 1*sizeof(uint8_t));
            pipe.InitBuffer(q_tmp3, BUFFER_NUM, 1*sizeof(uint8_t));
            pipe.InitBuffer(b_tmp2, 1*sizeof(half));
            pipe.InitBuffer(b_tmp3, 1*sizeof(half));
        }
        else{
            pipe.InitBuffer(b_tmp2, 1*sizeof(float));
        }
    }

    // double 类型数据处理
    __aicore__ inline void Process(){
        T start_val = (T)Gm_start.GetValue(0);
        T stop_val = (T) Gm_stop.GetValue(0);
        int32_t num_val = (int32_t) Gm_num_axes.GetValue(0);

        float f_start_val = 0;
        float f_stop_val = 0;
        float f_diff = 0;
        if constexpr (std::is_same_v<T, double>){
            f_start_val = this->f_start_val;
            f_stop_val = this->f_stop_val;
            f_diff = (f_stop_val-f_start_val)/(num_val-1);
        }
        else if constexpr (std::is_same_v<T, bfloat16_t>){
            f_start_val = ToFloat(start_val);
            f_stop_val = ToFloat(stop_val);
            f_diff = (f_stop_val-f_start_val)/(num_val-1);
        }
        else if constexpr (std::is_same_v<T, half>){
            f_start_val = float(start_val);
            f_stop_val = float(stop_val);
            f_diff = (f_stop_val-f_start_val)/(num_val-1);
        }
        else if constexpr (std::is_same_v<T, uint8_t>){
            
        }
        else{
            f_start_val = float(start_val);
            f_stop_val = float(stop_val);
            f_diff = ((float)stop_val-(float)start_val)/((float)num_val-1);
        }

        for(int32_t i = 0; i < this->totalLength; i++){
            if constexpr (std::is_same_v<T, int8_t>){
                auto tmp1 = b_tmp1.Get<int8_t>();
                auto tmp2 = b_tmp2.Get<half>();
                Duplicate(tmp2, (half)f_diff, 1);
                Muls(tmp2, tmp2, (half)i, 1);
                Adds(tmp2, tmp2, (half)f_start_val, 1);
                Cast(tmp1, tmp2, AscendC::RoundMode::CAST_TRUNC, 1);
                Gm_output.SetValue(i, tmp1.GetValue(0));
            }
            else if constexpr (std::is_same_v<T, uint8_t> ){
                auto tmp1 = q_tmp.AllocTensor<uint8_t>();
                auto tmp4 = q_tmp2.AllocTensor<uint8_t>();
                auto tmp2 = b_tmp2.Get<half>();
                auto tmp3 = b_tmp3.Get<half>();
            
                DataCopy(tmp1, Gm_start[0], 1);
                DataCopy(tmp4, Gm_stop[0], 1);
                q_tmp.EnQue(tmp1);
                q_tmp2.EnQue(tmp4);


                auto res = q_tmp3.AllocTensor<uint8_t>();
                auto half_start = q_tmp.DeQue<uint8_t>();
                auto half_stop = q_tmp2.DeQue<uint8_t>();
                Cast(tmp2, half_start, AscendC::RoundMode::CAST_NONE, 1); //tmp2: half_start
                Cast(tmp3, half_stop, AscendC::RoundMode::CAST_NONE, 1); //tmp3: half_stop
                
                Sub(tmp2, tmp3, tmp2, 1); //tmp2: half_stop - half_start
                Duplicate(tmp3, (half)(num_val-1), 1); // tmp3: half_steps-1
                Div(tmp2, tmp2, tmp3, 1); //half_diff
                
                Muls(tmp2, tmp2, (half)i, 1);
                Adds(tmp2, tmp2, (half)f_start_val, 1);

                Cast(res, tmp2, AscendC::RoundMode::CAST_TRUNC, 1);
                q_tmp.FreeTensor(tmp1);
                q_tmp2.FreeTensor(tmp4);
                q_tmp3.EnQue<uint8_t>(res);

                auto y = q_tmp3.DeQue<uint8_t>();
                DataCopy(Gm_output[i], y, 1);
                q_tmp3.FreeTensor(y);

            }
            else if constexpr (std::is_same_v<T, int16_t>){
                float value = (float) i * f_diff + f_start_val;
                auto tmp1 = b_tmp1.Get<int16_t>();
                auto tmp2 = b_tmp2.Get<float>();
                Duplicate(tmp2, value, 1);
                Cast(tmp1, tmp2, AscendC::RoundMode::CAST_TRUNC, 1);
                Gm_output.SetValue(i, tmp1.GetValue(0));
            }
            else if constexpr (std::is_same_v<T, int32_t>){
                float value = (float) i * f_diff + f_start_val;
                auto tmp1 = b_tmp1.Get<int32_t>();
                auto tmp2 = b_tmp2.Get<float>();
                Duplicate(tmp2, value, 1);
                Cast(tmp1, tmp2, AscendC::RoundMode::CAST_TRUNC, 1);
                Gm_output.SetValue(i, tmp1.GetValue(0));
                
            }
            else if constexpr (std::is_same_v<T, float>){
                float value = (float)i * f_diff + f_start_val;
                Gm_output.SetValue(i, value);
            }
            else if constexpr (std::is_same_v<T, half>){
                float value = (float)i * f_diff + f_start_val;
                Gm_output.SetValue(i, (half)value);
                // auto h = b_tmp1.Get<half>();
                // Duplicate(h, diff, 1);
                // Muls(h, h, (half)i,1);
                // Adds(h, h, start_val, 1);
                // Gm_output.SetValue(i, h.GetValue(0));
            }
            else if constexpr (std::is_same_v<T, bfloat16_t>){
                float value = (float)i * f_diff + f_start_val;
                bfloat16_t bvalue= ToBfloat16(value);
                Gm_output.SetValue(i, bvalue);
            }
            // else{ // double

            // }

        }
        
    }


private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> q_tmp, q_tmp2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> q_tmp3;

    TBuf<QuePosition::VECCALC> b_tmp1, b_tmp2, b_tmp3;

    GlobalTensor<T> Gm_start;
    GlobalTensor<T> Gm_stop;
    GlobalTensor<int32_t> Gm_num_axes;
    GlobalTensor<T> Gm_output;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t totalLength;

    float f_start_val;
    float f_stop_val;
};


extern "C" __global__ __aicore__ void lin_space(GM_ADDR start, GM_ADDR stop, GM_ADDR num_axes, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernalLinSpace<DTYPE_START> op;
    op.Init(start, stop, num_axes, output, tiling_data.start_val, tiling_data.stop_val, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
    op.Process();
    // TODO: user kernel impl
}