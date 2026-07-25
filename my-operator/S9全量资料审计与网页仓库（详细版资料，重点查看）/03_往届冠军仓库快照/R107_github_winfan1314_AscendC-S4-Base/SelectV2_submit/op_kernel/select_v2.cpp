#include "kernel_inc.h"

// tiling structs
struct tilingInfo {
    uint32_t tileLength, tileNumber, reminder;
};

template<unsigned long sz> struct _BUFFER_INFO { constexpr static uint32_t BUFFER_NUM = 2; };

template<class Tp> using BUFFER_INFO = _BUFFER_INFO<sizeof(Tp)>;

/* ************************************************************************************** */
template<class COND, class DATA, int tilingKey, class tilingStruct> class MySelectV2 { };           // base template
template<class COND, class DATA, class tilingStruct> class MySelectV2<COND, DATA, 1, tilingStruct> {// non-broadcast ver
    TQue<QuePosition::VECIN, BUFFER_INFO<DATA>::BUFFER_NUM> q_x1, q_x2, q_cond;
    TQue<QuePosition::VECOUT, BUFFER_INFO<DATA>::BUFFER_NUM> q_y;
    TBuf<QuePosition::VECCALC> bf_cmp_result, bf_casted_cond, bf_casted_x1_y, bf_casted_x2;
    GlobalTensor<DATA> gm_x1, gm_x2, gm_y;
    GlobalTensor<COND> gm_cond;
    tilingStruct ti;
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length) {
        ALLOC(COND, cond);
        DataCopy(cond, gm_cond[offset], length);
        ENQUE(COND, cond);
        ALLOC(DATA, x1);
        ALLOC(DATA, x2);
        DataCopy(x1, gm_x1[offset], length);
        DataCopy(x2, gm_x2[offset], length);
        ENQUE(DATA, x1);
        ENQUE(DATA, x2);
    }
    __aicore__ inline void Compute(uint32_t length) {
        DEQUE(uint8_t, cond);
        GET(uint8_t, cmp_result);
        GET(half, casted_cond);
        Cast(casted_cond, cond, RoundMode::CAST_NONE, length);
        FREE(cond);
        CompareScalar(cmp_result, casted_cond, half(0.0), CMPMODE::NE, length);
        DEQUE(DATA, x1);
        DEQUE(DATA, x2);
        if constexpr (std::is_same_v<DATA, int8_t>) {
            GET(half, casted_x1_y);
            GET(half, casted_x2);
            Cast(casted_x1_y, x1, RoundMode::CAST_NONE, length);
            Cast(casted_x2, x2, RoundMode::CAST_NONE, length);
            FREE(x1);
            FREE(x2);
            Select(casted_x1_y, cmp_result, casted_x1_y, casted_x2, SELMODE::VSEL_TENSOR_TENSOR_MODE, length);
            ALLOC(DATA, y);
            Cast(y, casted_x1_y, RoundMode::CAST_RINT, length);
            ENQUE(DATA, y);
        } else if constexpr (std::is_same_v<DATA, int32_t>) {
            ALLOC(DATA, y);
            Select(y.TREAT_AS(float), cmp_result, x1.TREAT_AS(float), x2.TREAT_AS(float), SELMODE::VSEL_TENSOR_TENSOR_MODE, length);
            FREE(x1);
            FREE(x2);
            ENQUE(DATA, y);
        } else { // half & float
            ALLOC(DATA, y);
            Select(y, cmp_result, x1, x2, SELMODE::VSEL_TENSOR_TENSOR_MODE, length);
            FREE(x1);
            FREE(x2);
            ENQUE(DATA, y);
        }
    }
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length) {
        DEQUE(DATA, y);
        DataCopy(gm_y[offset], y, length);
        FREE(y);
    }
public:
    __aicore__ inline MySelectV2(GM_ADDR cond, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, TPipe *p, const tilingStruct &ti):ti(ti) {
        SET_GLOBAL(COND, cond, 0);
        SET_GLOBAL(DATA, x1, 0);
        SET_GLOBAL(DATA, x2, 0);
        SET_GLOBAL(DATA, y, 0);

        INIT_QUEUE_N(COND, q_cond, BUFFER_INFO<DATA>::BUFFER_NUM, TI.tileLength);  // BUFFER_NUM * len * B
        INIT_QUEUE_N(DATA, q_x1, BUFFER_INFO<DATA>::BUFFER_NUM, TI.tileLength);    // BUFFER_NUM * len * DT * B
        INIT_QUEUE_N(DATA, q_x2, BUFFER_INFO<DATA>::BUFFER_NUM, TI.tileLength);    // BUFFER_NUM * len * DT * B
        INIT_QUEUE_N(DATA, q_y, BUFFER_INFO<DATA>::BUFFER_NUM, TI.tileLength);     // BUFFER_NUM * len * DT * B

        INIT_BUFF_N(uint8_t, bf_cmp_result, TI.tileLength);     // 1 * len * B
        INIT_BUFF_N(half, bf_casted_cond, TI.tileLength);       // 1 * len * 2 * B

        if constexpr(std::is_same_v<DATA, int8_t>) {            // for int8_t
            INIT_BUFF_N(half, bf_casted_x1_y, TI.tileLength);   // 1 * len * 2 * B
            INIT_BUFF_N(half, bf_casted_x2, TI.tileLength);     // 1 * len * 2 * B
        }
    }
    __aicore__ inline void reset_ptr(GM_ADDR cond, GM_ADDR x1, GM_ADDR x2, GM_ADDR y) {
        SET_GLOBAL(COND, cond, 0);
        SET_GLOBAL(DATA, x1, 0);
        SET_GLOBAL(DATA, x2, 0);
        SET_GLOBAL(DATA, y, 0);
    }
    __aicore__ inline void exec() {
        uint32_t idx;
        for(idx = 0u; idx < TI.tileNumber * TI.tileLength; idx += TI.tileLength) {
            CopyIn(idx, TI.tileLength);
            Compute(TI.tileLength);
            CopyOut(idx, TI.tileLength);
        }
        if(TI.reminder != 0) {
            CopyIn(idx, TI.reminder);
            Compute(TI.reminder);
            CopyOut(idx, TI.reminder);
        }
    }
};
/* ************************************************************************************** */
template<class COND, class DATA, class tilingStruct> class MySelectV2<COND, DATA, 2, tilingStruct> {// broadcast ver
    GM_ADDR cond;
    GM_ADDR x1;
    GM_ADDR x2;
    GM_ADDR y;
    tilingStruct ti;
public:
    __aicore__ MySelectV2(GM_ADDR cond, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const tilingStruct &ti):cond(cond), x1(x1), x2(x2), y(y), ti(ti) { }
    __aicore__ inline void exec() {
        for (uint32_t y_idx = 0; y_idx < TI.dataLength; y_idx++) {
            uint32_t cond_idx = 0, x1_idx = 0, x2_idx = 0;
            for (uint32_t axis = 0; axis < TI.shapeSize; axis++) { // O(M * N) M=maxLength N=ShapeSize
                cond_idx += TI.condShape[axis] == 1 ? 0 : TI.condShapeRSum[axis] * (y_idx / TI.yShapeRSum[axis] % TI.yShape[axis]);
                x1_idx += TI.x1Shape[axis] == 1 ? 0 : TI.x1ShapeRSum[axis] * (y_idx / TI.yShapeRSum[axis] % TI.yShape[axis]);
                x2_idx += TI.x2Shape[axis] == 1 ? 0 : TI.x2ShapeRSum[axis] * (y_idx / TI.yShapeRSum[axis] % TI.yShape[axis]);
            }
            *(CAST(__gm__ DATA *, this->y) + y_idx) = *(this->cond + cond_idx) ? *(CAST(__gm__ DATA *, this->x1) + x1_idx)
            : *(CAST(__gm__ DATA *, this->x2) + x2_idx);
        }
    }
};
/* ************************************************************************************** */
template<class COND, class DATA, class tilingStruct> class MySelectV2<COND, DATA, 3, tilingStruct> {// broadcast ver with Mini Batch
    GM_ADDR cond;
    GM_ADDR x1;
    GM_ADDR x2;
    GM_ADDR y;
    tilingStruct ti;
    MySelectV2<COND, DATA, 1, tilingInfo> op;
public:
    __aicore__ MySelectV2(  GM_ADDR cond,
                            GM_ADDR x1,
                            GM_ADDR x2,
                            GM_ADDR y,
                            const tilingStruct &ti,
                            TPipe *p):  cond(cond),
                                        x1(x1),
                                        x2(x2),
                                        y(y),
                                        ti(ti),
                                        op(cond, x1, x2, y, p, tilingInfo{TI.tileLength, TI.tileNumber, TI.reminder}) { }
    __aicore__ inline void exec() {
        DEF_CVAR_FROM_TI(bigBatch);
        DEF_CVAR_FROM_TI(shapeSize);
        for (uint32_t i = 0; i < bigBatch; ++i) {
            uint32_t y_idx = i * TI.yShape[0], cond_idx = 0, x1_idx = 0, x2_idx = 0;
            for (uint32_t axis = 0; axis < shapeSize; axis++) {
                cond_idx += TI.condShape[axis] == 1 ? 0 : TI.condShapeRSum[axis] * (y_idx / TI.yShapeRSum[axis] % TI.yShape[axis]);
                x1_idx += TI.x1Shape[axis] == 1 ? 0 : TI.x1ShapeRSum[axis] * (y_idx / TI.yShapeRSum[axis] % TI.yShape[axis]);
                x2_idx += TI.x2Shape[axis] == 1 ? 0 : TI.x2ShapeRSum[axis] * (y_idx / TI.yShapeRSum[axis] % TI.yShape[axis]);
            }
            op.reset_ptr(   cond + cond_idx,
                            x1 + x1_idx * sizeof(DATA), 
                            x2 + x2_idx * sizeof(DATA), 
                            y + y_idx * sizeof(DATA));
            op.exec();
            
        }
    }
};
/* ************************************************************************************** */
extern "C" __global__ __aicore__ void select_v2(GM_ADDR condition, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    // TODO: user kernel impl
    if(TILING_KEY_IS(3)) {
        TPipe p;
        GET_TILING_DATA_WITH_STRUCT(SelectV2TilingDataBCWithMiniBatch, ti, tiling);
        MySelectV2<DTYPE_CONDITION, DTYPE_Y, 3, decltype(ti)>(condition, x1, x2, y, ti, &p).exec();

    }
    else if(TILING_KEY_IS(2)) {
        GET_TILING_DATA_WITH_STRUCT(SelectV2TilingDataBC, ti, tiling);
        MySelectV2<DTYPE_CONDITION, DTYPE_Y, 2, decltype(ti)>(condition, x1, x2, y, ti).exec();
    }
    else if(TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(SelectV2TilingDataNoBC, ti, tiling);
        TPipe p;
        MySelectV2<DTYPE_CONDITION, DTYPE_Y, 1, decltype(ti)>(condition, x1, x2, y, &p, ti).exec();
    }
}