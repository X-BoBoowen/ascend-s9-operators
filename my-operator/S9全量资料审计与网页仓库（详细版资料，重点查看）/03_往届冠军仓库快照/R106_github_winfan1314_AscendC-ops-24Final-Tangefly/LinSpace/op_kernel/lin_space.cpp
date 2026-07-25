#include "kernel_inc.h"

template<class DATA, int TK, class TS> class MyLinSpace{ };
/* ************************************************************************************** */
template<class DATA, class TS> class MyLinSpace<DATA, 1, TS> {                              // fp32
    GlobalTensor<DATA> gm_output;
    TQue<TPosition::VECOUT, 2> q_output;
    TS ti;
    DATA start, stop;
    int32_t step;
    __aicore__ inline void calc(const DATA &init, const DATA &mulVal, const uint32_t length) {
        ALLOC(DATA, output);
        for(int i = 0; i < length; i++)
            output.SetValue(i, DATA((init + i) * mulVal + this->start));
        ENQUE(DATA, output);
    }
    __aicore__ inline void copyUB2GM(const uint32_t prefix, const uint32_t length) {
        DEQUE(DATA, output);
        DataCopy(this->gm_output[prefix], output, length);
        FREE(output);
    }
public:
    __aicore__ inline MyLinSpace(GM_ADDR start, GM_ADDR stop, GM_ADDR num_axes, GM_ADDR output, const TS &ti, TPipe *p):ti(ti) {
        SET_GLOBAL(DATA, output, 0);
        INIT_QUEUE_2(DATA, q_output, TI.tileLength);    // tl * 4 * 2
        this->start = *CAST(__gm__ DATA *, start);
        this->stop = *CAST(__gm__ DATA *, stop);
        this->step = *CAST(__gm__ int32_t *, num_axes);
    }
    __aicore__ inline void exec() {
        DATA current_start = 0, mulVal = (stop - start) / (step - 1);
        for (uint32_t i = 0; i < TI.tileNumber; ++i) {
            calc(current_start, mulVal, TI.tileLength);
            copyUB2GM(i * TI.tileLength, TI.tileLength);
            current_start += TI.tileLength;
        }
        if (TI.reminder != 0) {
            calc(current_start, mulVal, TI.reminder);
            copyUB2GM(TI.tileNumber * TI.tileLength, TI.reminder);
        }
    }
};
/* ************************************************************************************** */
template<class DATA, class TS> class MyLinSpace<DATA, 2, TS> {                              // int32 int16 int8
    GlobalTensor<DATA> gm_output;
    TQue<TPosition::VECOUT, 2> q_output;
    TS ti;
    DATA start, stop;
    int32_t step;
    __aicore__ inline void calc(const float &init, const float &mulVal, const uint32_t length) {
        ALLOC(DATA, output);
        for(int i = 0; i < length; i++) 
            output.SetValue(i, DATA((init + i * float(1.0)) * mulVal + this->start));
        ENQUE(DATA, output);
    }
    __aicore__ inline void copyUB2GM(const uint32_t prefix, const uint32_t length) {
        DEQUE(DATA, output);
        DataCopy(this->gm_output[prefix], output, length);
        FREE(output);
    }
public:
    __aicore__ inline MyLinSpace(GM_ADDR start, GM_ADDR stop, GM_ADDR num_axes, GM_ADDR output, const TS &ti, TPipe *p):ti(ti) {
        SET_GLOBAL(DATA, output, 0);
        INIT_QUEUE_2(DATA, q_output, TI.tileLength);    // tl * type_sz * 2
        this->start = *CAST(__gm__ DATA *, start);
        this->stop = *CAST(__gm__ DATA *, stop);
        this->step = *CAST(__gm__ int32_t *, num_axes);
    }
    __aicore__ inline void exec() {
        float current_start = 0, mulVal = (stop - start) * float(1.0) / (step - 1);
        for (uint32_t i = 0; i < TI.tileNumber; ++i) {
            calc(current_start, mulVal, TI.tileLength);
            copyUB2GM(i * TI.tileLength, TI.tileLength);
            current_start += TI.tileLength;
        }
        if (TI.reminder != 0) {
            calc(current_start, mulVal, TI.reminder);
            copyUB2GM(TI.tileNumber * TI.tileLength, TI.reminder);
        }
    }
};
/* ************************************************************************************** */
template<class DATA, class TS> class MyLinSpace<DATA, 3, TS> {                              // fp16 bf16
    GlobalTensor<DATA> gm_output;
    TQue<TPosition::VECOUT, 2> q_output;
    TBuf<TPosition::VECCALC> bf_f32_1, bf_f32_2, bf_f16_1, bf_f16_2;
    TBuf<TPosition::VECCALC> bf_f32;
    TS ti;
    DATA start, stop;
    float start_32;
    int32_t step;
    __aicore__ inline void calc(const float &init, const float &mulVal, const uint32_t length) {
        ALLOC(DATA, output);
        GET(float, f32);
        for(int i = 0; i < length; i++) {
            f32.SetValue(i, DATA((init + i * float(1.0)) * mulVal + this->start_32));
        }
        TQueSync<PIPE_S, PIPE_V> sync;
        sync.SetFlag(0);
        sync.WaitFlag(0);
        Cast(output, f32, RoundMode::CAST_RINT, length);
        ENQUE(DATA, output);
    }
    __aicore__ inline void copyUB2GM(const uint32_t prefix, const uint32_t length) {
        DEQUE(DATA, output);
        DataCopy(this->gm_output[prefix], output, length);
        FREE(output);
    }
public:
    __aicore__ inline MyLinSpace(GM_ADDR start, GM_ADDR stop, GM_ADDR num_axes, GM_ADDR output, const TS &ti, TPipe *p):ti(ti) {
        SET_GLOBAL(DATA, output, 0);
        INIT_QUEUE_2(DATA, q_output, TI.tileLength);    // tl * data_sz * 2
        INIT_BUFF_N(DATA, bf_f16_1, 1);                 // 32B
        INIT_BUFF_N(DATA, bf_f16_2, 1);                 // 32B
        INIT_BUFF_N(float, bf_f32_1, 1);                // 32B
        INIT_BUFF_N(float, bf_f32_2, 1);                // 32B
        INIT_BUFF_N(float, bf_f32, TI.tileLength);      // tl * 4 * 1
        this->start = *CAST(__gm__ DATA *, start);
        this->stop = *CAST(__gm__ DATA *, stop);
        this->step = *CAST(__gm__ int32_t *, num_axes);
    }
    __aicore__ inline void exec() {
        TQueSync<PIPE_S, PIPE_V> sync;
        TQueSync<PIPE_V, PIPE_S> sync2;
        GET(DATA, f16_1);
        GET(DATA, f16_2);
        GET(float, f32_1);
        GET(float, f32_2);
        f16_1.SetValue(0, this->start);
        f16_2.SetValue(0, this->stop);
        sync.SetFlag(0);
        sync.WaitFlag(0);
        Cast(f32_1, f16_1, RoundMode::CAST_NONE, 1);
        Cast(f32_2, f16_2, RoundMode::CAST_NONE, 1);
        Sub(f32_2, f32_2, f32_1, 1); // f32_2 = end - start
        sync2.SetFlag(0);
        sync2.WaitFlag(0);
        this->start_32 = f32_1.GetValue(0); // get start 
        f32_1.TREAT_AS(int32_t).SetValue(0, this->step - 1); // set f32_1 = step - 1
        sync.SetFlag(0);
        sync.WaitFlag(0);
        Cast(f32_1, f32_1.TREAT_AS(int32_t), RoundMode::CAST_ROUND, 1); // TODO
        Div(f32_1, f32_2, f32_1, 1); // f32_1 = (end - start) / (step - 1)
        sync2.SetFlag(0);
        sync2.WaitFlag(0);
        const float mulVal = f32_1.GetValue(0);
        sync.SetFlag(0);
        sync.WaitFlag(0);
        float current_start = 0;
        for (uint32_t i = 0; i < TI.tileNumber; ++i) {
            calc(current_start, mulVal, TI.tileLength);
            copyUB2GM(i * TI.tileLength, TI.tileLength);
            current_start += TI.tileLength;
        }
        if (TI.reminder != 0) {
            calc(current_start, mulVal, TI.reminder);
            copyUB2GM(TI.tileNumber * TI.tileLength, TI.reminder);
        }
    }
};
/* ************************************************************************************** */
template<class DATA, class TS> class MyLinSpace<DATA, 4, TS> {                              // uint8
    GlobalTensor<DATA> gm_output;
    TQue<TPosition::VECOUT, 2> q_output;
    TBuf<TPosition::VECCALC> bf_f32, bf_f16;
    TS ti;
    DATA start, stop;
    int32_t step;
    __aicore__ inline void calc(const float &init, const float &mulVal, const uint32_t length) {
        ALLOC(DATA, output);
        GET(float, f32);
        GET(half, f16);
        TQueSync<PIPE_S, PIPE_V> sync;
        for(int i = 0; i < length; i++)
            f32.SetValue(i, float((init + i * float(1.0)) * mulVal + int16_t(this->start)));
        sync.SetFlag(0);
        sync.WaitFlag(0);
        Cast(f16, f32, RoundMode::CAST_FLOOR, length);
        Cast(output, f16, RoundMode::CAST_FLOOR, length);
        ENQUE(DATA, output);
    }
    __aicore__ inline void copyUB2GM(const uint32_t prefix, const uint32_t length) {
        DEQUE(DATA, output);
        DataCopy(this->gm_output[prefix], output, length);
        FREE(output);
    }
public:
    __aicore__ inline MyLinSpace(GM_ADDR start, GM_ADDR stop, GM_ADDR num_axes, GM_ADDR output, const TS &ti, TPipe *p):ti(ti) {
        SET_GLOBAL(DATA, output, 0);
        INIT_QUEUE_2(DATA, q_output, TI.tileLength);    // tl * 1 * 2
        INIT_BUFF_N(float, bf_f32, TI.tileLength);      // tl * 4 * 1
        INIT_BUFF_N(half, bf_f16, TI.tileLength);       // tl * 2 * 1
        this->start = *CAST(__gm__ DATA *, start);
        this->stop = *CAST(__gm__ DATA *, stop);
        this->step = *CAST(__gm__ int32_t *, num_axes);
    }
    __aicore__ inline void exec() {
        float current_start = 0, mulVal = (stop - start) * float(1.0) / (step - 1);
        for (uint32_t i = 0; i < TI.tileNumber; ++i) {
            calc(current_start, mulVal, TI.tileLength);
            copyUB2GM(i * TI.tileLength, TI.tileLength);
            current_start += TI.tileLength;
        }
        if (TI.reminder != 0) {
            calc(current_start, mulVal, TI.reminder);
            copyUB2GM(TI.tileNumber * TI.tileLength, TI.reminder);
        }
    }
};
/* ************************************************************************************** */
extern "C" __global__ __aicore__ void lin_space(GM_ADDR start, GM_ADDR stop, GM_ADDR num_axes, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe p;
    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(LinSpaceTIBasic, ti, tiling);
        if constexpr (is_same_v<DTYPE_OUTPUT, float>)                                                   // float32
            MyLinSpace<float, 1, decltype(ti)>(start, stop, num_axes, output, ti, &p).exec();
        else if constexpr (is_same_v<DTYPE_OUTPUT, int32_t>                                             // int16 int32 int8
            || is_same_v<DTYPE_OUTPUT, int16_t>
            || is_same_v<DTYPE_OUTPUT, int8_t>)        
            MyLinSpace<DTYPE_OUTPUT, 2, decltype(ti)>(start, stop, num_axes, output, ti, &p).exec();
        else if constexpr(is_same_v<DTYPE_OUTPUT, uint8_t>)                                             // uint8
            MyLinSpace<uint8_t, 4, decltype(ti)>(start, stop, num_axes, output, ti, &p).exec();
        else if constexpr (is_same_v<DTYPE_OUTPUT, double>) {}                                          // double: deprecated
        else MyLinSpace<half, 3, decltype(ti)>(start, stop, num_axes, output, ti, &p).exec();           // float16 bf16
    }
}