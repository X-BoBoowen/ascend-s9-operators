#include "kernel_inc.h"

constexpr static int BF_NUM = 2;

template<class DATA, int TK, class TS> class MyTrunc {};

template<class DATA, class TS> class MyTrunc<DATA, 1, TS> {
    GlobalTensor<DATA> gm_input_x, gm_output_y;
    TQue<TPosition::VECIN, BF_NUM> q_in;
    TBuf<TPosition::VECCALC> bf_tmp;
    TQue<TPosition::VECOUT, BF_NUM> q_out;
    TS ti;
    __aicore__ inline void CopyGM2UB(const uint32_t prefix, const uint32_t length) {
        ALLOC(DATA, in);
        DataCopy(in, gm_input_x[prefix], length);
        ENQUE(DATA, in);
    }
    __aicore__ inline void Calculate(const uint32_t length) {
        DEQUE(DATA, in);
        ALLOC(DATA, out);
        
        if constexpr (is_same_v<DATA, float>) { // f32
            Cast(out, in, RoundMode::CAST_TRUNC, length);
        } else { // bf16 f16
            GET(float, tmp);
            Cast(tmp, in, RoundMode::CAST_NONE, length);
            Cast(tmp, tmp, RoundMode::CAST_TRUNC, length);
            Cast(out, tmp, RoundMode::CAST_RINT, length);
        }
        FREE(in);
        ENQUE(DATA, out);
    }
    __aicore__ inline void CopyUB2GM(const uint32_t prefix, const uint32_t length) {
        DEQUE(DATA, out);
        DataCopy(gm_output_y[prefix], out, length);
        FREE(out);
    }
public:
    __aicore__ inline MyTrunc(GM_ADDR input_x, GM_ADDR output_y, const TS &ti, TPipe *p):ti(ti) {
        SET_GLOBAL(DATA, input_x, 0);
        SET_GLOBAL(DATA, output_y, 0);
        INIT_QUEUE_N(DATA, q_in, BF_NUM, TI.tileLength);
        INIT_QUEUE_N(DATA, q_out, BF_NUM, TI.tileLength);
        if constexpr (!is_same_v<DATA, float>) INIT_BUFF_N(float, bf_tmp, TI.tileLength);
    }
    __aicore__ inline void exec() {
        for (uint32_t i = 0; i < TI.tileNumber; ++i) {
            const auto ii = i * TI.tileLength;
            CopyGM2UB(ii, TI.tileLength);
            Calculate(TI.tileLength);
            CopyUB2GM(ii, TI.tileLength);
        }
        if (TI.reminder != 0) {
            const auto ii = TI.tileNumber * TI.tileLength;
            CopyGM2UB(ii, TI.reminder);
            Calculate(TI.reminder);
            CopyUB2GM(ii, TI.reminder);
        }
    }

};

template<class DATA, class TS> class MyTrunc<DATA, 2, TS> {
    GlobalTensor<DATA> gm_input_x, gm_output_y;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BF_NUM> q_data;
    TS ti;
    __aicore__ inline void CopyGM2UB(const uint32_t prefix, const uint32_t length) {
        ALLOC(DATA, data);
        DataCopy(data, gm_input_x[prefix], length);
        ENQUE(DATA, data);
    }
    __aicore__ inline void CopyUB2GM(const uint32_t prefix, const uint32_t length) {
        DEQUE(DATA, data);
        DataCopy(gm_output_y[prefix], data, length);
        FREE(data);
    }
public:
    __aicore__ inline MyTrunc(GM_ADDR input_x, GM_ADDR output_y, const TS &ti, TPipe *p):ti(ti) {
        SET_GLOBAL(DATA, input_x, 0);
        SET_GLOBAL(DATA, output_y, 0);
        INIT_QUEUE_N(DATA, q_data, BF_NUM, TI.tileLength);
    }
    __aicore__ inline void exec() {
        for (uint32_t i = 0; i < TI.tileNumber; ++i) {
            const auto ii = i * TI.tileLength;
            CopyGM2UB(ii, TI.tileLength);
            CopyUB2GM(ii, TI.tileLength);
        }
        if (TI.reminder != 0) {
            const auto ii = TI.tileNumber * TI.tileLength;
            CopyGM2UB(ii, TI.reminder);
            CopyUB2GM(ii, TI.reminder);
        }
    }

};

extern "C" __global__ __aicore__ void trunc(GM_ADDR input_x, GM_ADDR output_y, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe p;
    if(TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(TruncTIBasic, ti, tiling);
        if constexpr (!(is_same_v<DTYPE_INPUT_X, uint8_t> || is_same_v<DTYPE_INPUT_X, int32_t> || is_same_v<DTYPE_INPUT_X, int8_t>))
            MyTrunc<DTYPE_INPUT_X, 1, decltype(ti)>(input_x, output_y, ti, &p).exec();
    } else if(TILING_KEY_IS(2)) {
        GET_TILING_DATA_WITH_STRUCT(TruncTIBasic2, ti, tiling);
        if constexpr (is_same_v<DTYPE_INPUT_X, uint8_t> || is_same_v<DTYPE_INPUT_X, int32_t> || is_same_v<DTYPE_INPUT_X, int8_t>)
            MyTrunc<DTYPE_INPUT_X, 2, decltype(ti)>(input_x, output_y, ti, &p).exec();
    }
}