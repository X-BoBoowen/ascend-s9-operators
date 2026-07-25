#include "kernel_inc.h"

// configurations
#define SYNC_ENABLED 0

// constants
template<int tilingKey> struct Constants { };                       // base template
template<> struct Constants<1> {                                    // non-broadcast ver
    constexpr static uint32_t  BUFFER_NUM       = 2;
};
template<> struct Constants<2> {                                    // broadcast ver 16 for Kernel 2 (7 * MEM < UB Size) 512 * DB Max
    constexpr static uint32_t  BUFFER_NUM       = 1;
    constexpr static uint32_t  DB               = 32;
    constexpr static uint32_t  MEM              = 512 * DB;
    constexpr static uint32_t  MINI_BATCH16     = MEM >> 1;
    constexpr static uint32_t  MINI_BATCH32     = MEM >> 2;
};
template<> struct Constants<3> {                                    // broadcast ver 32 for Kernel 3 (3 * MEM < UB Size) 1024 * DB Max
    constexpr static uint32_t  BUFFER_NUM       = 1;
    constexpr static uint32_t  DB               = 32;
    constexpr static uint32_t  MEM              = 1024 * DB;
    constexpr static uint32_t  MINI_BATCH16     = MEM >> 1;
    constexpr static uint32_t  MINI_BATCH32     = MEM >> 2;
};

template<unsigned long> struct _BUFFER_INFO {constexpr static uint32_t BUFFER_NUM = 2;};

template<class Tp> using BUFFER_INFO = _BUFFER_INFO<sizeof(Tp)>;

// tiling structs
struct tilingInfo {
    uint32_t tileLength, maxLength, reminder;
};
/* ************************************************************************************** */
// Kernel 1 2 3 4 are in use
template<typename DATA, int tilingKey, class tilingStruct> class MyPows { };            // base template
/* ************************************************************************************** */
template<typename DATA, class tilingStruct> class MyPows<DATA, 40, tilingStruct> {       // non-broadcast ver for miniBatch Helper
    TQue<QuePosition::VECIN, BUFFER_INFO<DATA>::BUFFER_NUM> q_x1, q_x2;
    TQue<QuePosition::VECOUT, BUFFER_INFO<DATA>::BUFFER_NUM> q_y;
    TBuf<QuePosition::VECCALC> bf_casted_x1, bf_casted_x2_y;
    GlobalTensor<DATA> gm_x1, gm_x2, gm_y;
    tilingStruct ti;
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length) {
        ALLOC(DATA, x1);
        ALLOC(DATA, x2);
        DataCopy(x1, gm_x1[offset], length);
        DataCopy(x2, gm_x2[offset], length);
        ENQUE(DATA, x1);
        ENQUE(DATA, x2);
    }
    __aicore__ inline void Compute(uint32_t length) {
        DEQUE(DATA, x1);
        DEQUE(DATA, x2);
        ALLOC(DATA, y);
        if constexpr(is_same_v<DATA, bfloat16_t> || is_same_v<DATA, half>) {
            GET(float, casted_x1);
            GET(float, casted_x2_y);
            Cast(casted_x1, x1, RoundMode::CAST_NONE, length);
            Cast(casted_x2_y, x2, RoundMode::CAST_NONE, length);
            Ln(casted_x1, casted_x1, length);
            Mul(casted_x2_y, casted_x2_y, casted_x1, length);
            Exp(casted_x2_y, casted_x2_y, length);
            Cast(y, casted_x2_y, RoundMode::CAST_RINT, length);
        } else {
            Ln(x1, x1, length);
            Mul(y, x2, x1, length);
            Exp(y, y, length);
        }
        ENQUE(DATA, y);
        FREE(x1);
        FREE(x2);
    }
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length) {
        DEQUE(DATA, y);
        DataCopy(gm_y[offset], y, length);
        FREE(y);
    }
public:
    __aicore__ inline MyPows(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const tilingStruct &ti, TPipe *p):ti(ti) {
        SET_GLOBAL(DATA, x1, 0);
        SET_GLOBAL(DATA, x2, 0);
        SET_GLOBAL(DATA, y, 0);
        INIT_QUEUE_N(DATA, q_x1, BUFFER_INFO<DATA>::BUFFER_NUM, TI.tileLength);   // BUFFER_NUM * len * DT * B
        INIT_QUEUE_N(DATA, q_x2, BUFFER_INFO<DATA>::BUFFER_NUM, TI.tileLength);   // BUFFER_NUM * len * DT * B
        INIT_QUEUE_N(DATA, q_y, BUFFER_INFO<DATA>::BUFFER_NUM, TI.tileLength);    // BUFFER_NUM * len * DT * B 
        if constexpr(is_same_v<DATA, bfloat16_t> || is_same_v<DATA, half>) {    // extra buff
            INIT_BUFF_N(float, bf_casted_x1, TI.tileLength);              // 1 * len * 4 * B
            INIT_BUFF_N(float, bf_casted_x2_y, TI.tileLength);            // 1 * len * 4 * B
        }
    }
    __aicore__ inline void reset_ptr(GM_ADDR x1, GM_ADDR x2, GM_ADDR y) {
        SET_GLOBAL(DATA, x1, 0);
        SET_GLOBAL(DATA, x2, 0);
        SET_GLOBAL(DATA, y, 0);
    }
    __aicore__ inline void exec() {
        uint32_t idx;
        for(idx = 0u; idx < TI.maxLength; idx += TI.tileLength) {
            CopyIn(idx, TI.tileLength);
            Compute(TI.tileLength);
            CopyOut(idx, TI.tileLength);
        }
        CopyIn(idx, TI.reminder);
        Compute(TI.reminder);
        CopyOut(idx, TI.reminder);
    }
};
/* ************************************************************************************** */
template<typename DATA, class tilingStruct> class MyPows<DATA, 1, tilingStruct> {       // non-broadcast ver 16 sp
    TQue<QuePosition::VECIN, BUFFER_INFO<DATA>::BUFFER_NUM> q_x1, q_x2;
    TQue<QuePosition::VECOUT, BUFFER_INFO<DATA>::BUFFER_NUM> q_y;
    TBuf<QuePosition::VECCALC> bf_casted_x1, bf_casted_x2_y;
    GlobalTensor<DATA> gm_x1, gm_x2, gm_y;
    tilingStruct ti;
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length) {
        ALLOC(DATA, x1);
        DataCopy(x1, gm_x1[offset], length);
        ENQUE(DATA, x1);
        ALLOC(DATA, x2);
        DataCopy(x2, gm_x2[offset], length);
        ENQUE(DATA, x2);
    }
    __aicore__ inline void Compute(uint32_t length) {
        GET(float, casted_x1);
        GET(float, casted_x2_y);
        DEQUE(DATA, x1);
        Cast(casted_x1, x1, RoundMode::CAST_NONE, length);
        FREE(x1);
        DEQUE(DATA, x2);
        Cast(casted_x2_y, x2, RoundMode::CAST_NONE, length);
        FREE(x2);
        Ln(casted_x1, casted_x1, length);
        Mul(casted_x2_y, casted_x2_y, casted_x1, length);
        Exp(casted_x2_y, casted_x2_y, length);
        ALLOC(DATA, y);
        Cast(y, casted_x2_y, RoundMode::CAST_RINT, length);
        ENQUE(DATA, y);
    }
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length) {
        DEQUE(DATA, y);
        DataCopy(gm_y[offset], y, length);
        FREE(y);
    }
public:
    __aicore__ inline MyPows(   GM_ADDR x1,
                                GM_ADDR x2,
                                GM_ADDR y,
                                const tilingStruct &ti,
                                TPipe *p): ti(ti) {
        SET_GLOBAL(DATA, x1, 0);
        SET_GLOBAL(DATA, x2, 0);
        SET_GLOBAL(DATA, y, 0);
        INIT_QUEUE_N(DATA, q_x1, BUFFER_INFO<DATA>::BUFFER_NUM, TI.tileLength); // BUFFER_NUM * len * DT * B
        INIT_QUEUE_N(DATA, q_x2, BUFFER_INFO<DATA>::BUFFER_NUM, TI.tileLength); // BUFFER_NUM * len * DT * B
        INIT_QUEUE_N(DATA, q_y, BUFFER_INFO<DATA>::BUFFER_NUM, TI.tileLength);  // BUFFER_NUM * len * DT * B 
        INIT_BUFF_N(float, bf_casted_x1, TI.tileLength);              // 1 * len * 4 * B
        INIT_BUFF_N(float, bf_casted_x2_y, TI.tileLength);            // 1 * len * 4 * B
    }
    __aicore__ inline void reset_ptr(GM_ADDR x1, GM_ADDR x2, GM_ADDR y) {
        SET_GLOBAL(DATA, x1, 0);
        SET_GLOBAL(DATA, x2, 0);
        SET_GLOBAL(DATA, y, 0);
    }
    __aicore__ inline void exec() {
        uint32_t idx;
        for(idx = 0u; idx < TI.maxLength; idx += TI.tileLength) {
            CopyIn(idx, TI.tileLength);
            Compute(TI.tileLength);
            CopyOut(idx, TI.tileLength);
        }
        CopyIn(idx, TI.reminder);
        Compute(TI.reminder);
        CopyOut(idx, TI.reminder);
    }
};
/* ************************************************************************************** */
template<class tilingStruct> class MyPows<float, -1, tilingStruct> {                    // non-broadcast ver 32 sp(result: 13200us) deprecated
    using DATA = float;
    TQue<QuePosition::VECIN, BUFFER_INFO<DATA>::BUFFER_NUM> q_x1, q_x2;
    TQue<QuePosition::VECOUT, BUFFER_INFO<DATA>::BUFFER_NUM> q_y;
    GlobalTensor<DATA> gm_x1, gm_x2, gm_y;
    tilingStruct ti;
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length) {
        ALLOC(DATA, x1);
        ALLOC(DATA, x2);
        DataCopy(x1, gm_x1[offset], length);
        DataCopy(x2, gm_x2[offset], length);
        ENQUE(DATA, x1);
        ENQUE(DATA, x2); 
    }
    __aicore__ inline void Compute(uint32_t length) {
        DEQUE(DATA, x1);
        Ln(x1, x1, length);
        DEQUE(DATA, x2);
        ALLOC(DATA, y);
        Mul(y, x2, x1, length);
        FREE(x1);
        FREE(x2);
        Exp(y, y, length);
        ENQUE(DATA, y);
    }
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length) {
        DEQUE(DATA, y);
        DataCopy(gm_y[offset], y, length);
        FREE(y);
    }
public:
    __aicore__ inline MyPows(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const tilingStruct &ti, TPipe *p):ti(ti) {
        SET_GLOBAL(DATA, x1, 0);
        SET_GLOBAL(DATA, x2, 0);
        SET_GLOBAL(DATA, y, 0);
        INIT_QUEUE_N(DATA, q_x1, BUFFER_INFO<DATA>::BUFFER_NUM, TI.tileLength);   // BUFFER_NUM * len * DT * B
        INIT_QUEUE_N(DATA, q_x2, BUFFER_INFO<DATA>::BUFFER_NUM, TI.tileLength);   // BUFFER_NUM * len * DT * B
        INIT_QUEUE_N(DATA, q_y, BUFFER_INFO<DATA>::BUFFER_NUM, TI.tileLength);    // BUFFER_NUM * len * DT * B 
    }
    __aicore__ inline void reset_ptr(GM_ADDR x1, GM_ADDR x2, GM_ADDR y) {
        SET_GLOBAL(DATA, x1, 0);
        SET_GLOBAL(DATA, x2, 0);
        SET_GLOBAL(DATA, y, 0);
    }
    __aicore__ inline void exec() {
        uint32_t idx;
        for(idx = 0u; idx < TI.maxLength; idx += TI.tileLength) {
            CopyIn(idx, TI.tileLength);
            Compute(TI.tileLength);
            CopyOut(idx, TI.tileLength);
        }
        CopyIn(idx, TI.reminder);
        Compute(TI.reminder);
        CopyOut(idx, TI.reminder);
    }
};
/* ************************************************************************************** */
template<class tilingStruct> class MyPows<float, 1, tilingStruct> {                     // non-broadcast ver 32 sp(result: 12400us)
    using DATA = float;
    constexpr static auto BUFFER_NUM = 4;
    GlobalTensor<DATA> gm_x1, gm_x2, gm_y;
    TBuf<TPosition::VECCALC> bf_x1[BUFFER_NUM], bf_y[BUFFER_NUM];
    LocalTensor<DATA> x1[BUFFER_NUM], y[BUFFER_NUM];
    tilingStruct ti;
    TQueSync<PIPE_MTE2, PIPE_V> syncIn1;
    // TQueSync<PIPE_V, PIPE_MTE2> syncIn2;
    TQueSync<PIPE_V, PIPE_MTE3> syncOut1;
    TQueSync<PIPE_MTE3, PIPE_V> syncOut2;
    TQueSync<PIPE_MTE3, PIPE_MTE2> sync3;
    __aicore__ inline void Process(uint8_t eventId, uint32_t offset, uint32_t length) {
        this->sync3.WaitFlag(eventId);
        DataCopy(this->x1[eventId & 3], this->gm_x1[offset], length);
        DataCopy(this->y[eventId & 3], this->gm_x2[offset], length); // copy x2 to yLocal
        this->syncIn1.SetFlag(0); // Vecter after MTE2
        this->syncIn1.WaitFlag(0); // Vecter after MTE2
        this->syncOut2.WaitFlag(eventId & 3); // Next Vector after This Vector
        Ln(this->x1[eventId & 3], this->x1[eventId & 3], length); // x1 Self Ln
        Mul(this->y[eventId & 3], this->y[eventId & 3], this->x1[eventId & 3], length); // y = x2 * Ln(x1)
        Exp(this->y[eventId & 3], this->y[eventId & 3], length); // y Self Exp
        this->syncOut1.SetFlag(0); // MTE3 after Vector
        this->syncOut1.WaitFlag(0); // MTE3 after Vector
        this->syncOut2.SetFlag((eventId + 1) & 3); // Next Vector after This Vector
        DataCopy(this->gm_y[offset], this->y[eventId & 3], length);
        this->sync3.SetFlag((eventId + 4) & 7);// Next same buffer after MTE3
        
    }
public:
    __aicore__ inline MyPows(   GM_ADDR x1,
                                GM_ADDR x2,
                                GM_ADDR y,
                                const tilingStruct &ti,
                                TPipe *p): ti(ti) {
        this->reset_ptr(x1, x2, y);
        for(uint8_t i = 0; i < BUFFER_NUM; i++) {
            INIT_BUFF_N(DATA, this->bf_x1[i], TI.tileLength);
            _GET(DATA, this->bf_x1[i], this->x1[i]);
            INIT_BUFF_N(DATA, this->bf_y[i], TI.tileLength);
            _GET(DATA, this->bf_y[i], this->y[i]);
        }
    }
    __aicore__ inline void reset_ptr(GM_ADDR x1, GM_ADDR x2, GM_ADDR y) {
        SET_GLOBAL(DATA, x1, 0);
        SET_GLOBAL(DATA, x2, 0);
        SET_GLOBAL(DATA, y, 0);
        for(uint8_t i = 0; i < BUFFER_NUM; i++) {
            this->sync3.SetFlag(i);
        }
        this->syncOut2.SetFlag(0);
    }
    __aicore__ inline void exec() {
        uint32_t idx;
        uint8_t eventId = 0;
        for(idx = 0u; idx < TI.maxLength; idx += TI.tileLength) {
            Process(eventId++ & 7, idx, TI.tileLength);
        }
        if(TI.reminder != 0) {
            Process(eventId++ & 7, TI.maxLength, TI.reminder);
        }
        for(uint8_t i = 0; i < BUFFER_NUM; i++) {
            this->sync3.WaitFlag((eventId + i) & 7);
        }
        this->syncOut2.WaitFlag(eventId & 3);
    }
};
/* ************************************************************************************** */
template<typename DATA, class tilingStruct> class MyPows<DATA, 2, tilingStruct> {       // broadcast ver 16 sp
    TQue<QuePosition::VECIN, Constants<2>::BUFFER_NUM> q_x1, q_x2;
    TQue<QuePosition::VECOUT, Constants<2>::BUFFER_NUM> q_y;
    TBuf<QuePosition::VECCALC> bf_casted_x1, bf_casted_x2_y;
    GlobalTensor<DATA> gm_y;
    LocalTensor<DATA> x1, x2;
    LocalTensor<float> casted_x1, casted_x2_y;
    tilingStruct ti;
#if SYNC_ENABLED
    TQueSync<PIPE_V, PIPE_S> sync{};
#endif
    GM_ADDR gm_x1;
    GM_ADDR gm_x2;
    __aicore__ inline void CopyIn(const int32_t i, const uint32_t x1_idx, const uint32_t x2_idx) {
        this->x1.SetValue(i, *(CAST(__gm__ DATA *, gm_x1) + x1_idx));
        this->x2.SetValue(i, *(CAST(__gm__ DATA *, gm_x2) + x2_idx));
    }
    __aicore__ inline void Compute(const int32_t reminder = Constants<2>::MINI_BATCH16) {
        ALLOC(DATA, y);
        Cast(casted_x1, x1, RoundMode::CAST_NONE, reminder);
        Cast(casted_x2_y, x2, RoundMode::CAST_NONE, reminder);
#if SYNC_ENABLED
        sync.SetFlag(0); // Vector Pipe SetFlag after Calc
#endif
        Ln(casted_x1, casted_x1, reminder);
        Mul(casted_x2_y, casted_x2_y, casted_x1, reminder);
        Exp(casted_x2_y, casted_x2_y, reminder);
        Cast(y, casted_x2_y, RoundMode::CAST_RINT, reminder);
        ENQUE(DATA, y);
    }
    __aicore__ inline void CopyOut(const uint32_t prefix, const int32_t reminder = Constants<2>::MINI_BATCH16) {
        DEQUE(DATA, y);
        DataCopy(gm_y[prefix], y, reminder);
        FREE(y);
    }
public:
    __aicore__ inline MyPows(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const tilingStruct &ti, TPipe *p):gm_x1(x1), gm_x2(x2), ti(ti) {
        SET_GLOBAL(DATA, y, 0);
        INIT_QUEUE_1(uint8_t, q_x1, Constants<2>::MEM);                     // 2 * miniBatch
        INIT_QUEUE_1(uint8_t, q_x2, Constants<2>::MEM);                     // 2 * miniBatch
        INIT_QUEUE_1(uint8_t, q_y, Constants<2>::MEM);                      // 2 * miniBatch
        INIT_BUFF_N(DATA, bf_casted_x1, Constants<2>::MINI_BATCH16 * 2);    // 4 * miniBatch
        INIT_BUFF_N(DATA, bf_casted_x2_y, Constants<2>::MINI_BATCH16 * 2);  // 4 * miniBatch
        this->casted_x1 = this->bf_casted_x1.template Get<float>();
        this->casted_x2_y = this->bf_casted_x2_y.template Get<float>();
    }
    __aicore__ inline void exec() { 
        DEF_CVAR_FROM_TI(shapeSize);
        DEF_CVAR_FROM_TI(dataLength);
        uint32_t trigger = 0;
        _ALLOC(DATA, q_x1, this->x1);
        _ALLOC(DATA, q_x2, this->x2);
        for (uint32_t y_idx = 0; y_idx < dataLength; y_idx++) {
            uint32_t x1_idx = 0, x2_idx = 0;
            for (uint32_t axis = 0; axis < shapeSize; axis++) {
                x1_idx += TI.x1Shape[axis] != 1 ? TI.x1ShapeRSum[axis] * (y_idx / TI.yShapeRSum[axis] % TI.yShape[axis]) : 0;
                x2_idx += TI.x2Shape[axis] != 1 ? TI.x2ShapeRSum[axis] * (y_idx / TI.yShapeRSum[axis] % TI.yShape[axis]) : 0;
            }
            CopyIn(trigger, x1_idx, x2_idx);      
            if(++trigger == Constants<2>::MINI_BATCH16) {
                Compute();
                CopyOut(y_idx + 1 - Constants<2>::MINI_BATCH16);
#if SYNC_ENABLED
                sync.WaitFlag(0); // Scalar Pipe wait until Vector Pipe SetFlag
#endif
                trigger = 0;
            }
        }
        Compute(trigger);
        CopyOut(dataLength - trigger, CEIL(trigger, 32 / sizeof(DATA)));
#if SYNC_ENABLED
        sync.WaitFlag(0); // Scalar Pipe wait until Vector Pipe SetFlag
#endif
        _FREE(q_x1, this->x1);
        _FREE(q_x2, this->x2);
    }
};
/* ************************************************************************************** */
template<typename DATA, class tilingStruct> class MyPows<DATA, 3, tilingStruct> {       // broadcast ver 32 sp
    TQue<QuePosition::VECIN, Constants<3>::BUFFER_NUM> q_x1, q_x2;
    TQue<QuePosition::VECOUT, Constants<3>::BUFFER_NUM> q_y;
    GM_ADDR gm_x1;
    GM_ADDR gm_x2;
    LocalTensor<DATA> x1, x2;
    GlobalTensor<DATA> gm_y;
    tilingStruct ti;
#if SYNC_ENABLED
    TQueSync<PIPE_V, PIPE_S> sync{};
#endif
    __aicore__ inline void CopyIn(const int32_t i, const uint32_t x1_idx, const uint32_t x2_idx) {
        this->x1.SetValue(i, *(CAST(__gm__ DATA *, gm_x1) + x1_idx));
        this->x2.SetValue(i, *(CAST(__gm__ DATA *, gm_x2) + x2_idx));
    }
    __aicore__ inline void Compute(const int32_t reminder = Constants<3>::MINI_BATCH32) {
        ALLOC(DATA, y);
        Ln(x1, x1, reminder);
        Mul(y, x2, x1, reminder);
#if SYNC_ENABLED
        sync.SetFlag(0); // Vector Pipe SetFlag after Calc
#endif
        Exp(y, y, reminder);
        ENQUE(DATA, y);
    }
    __aicore__ inline void CopyOut(const uint32_t prefix, const int32_t reminder = Constants<3>::MINI_BATCH32) {
        DEQUE(DATA, y);
        DataCopy(gm_y[prefix], y, reminder);
        FREE(y);
    }
public:
    __aicore__ inline MyPows(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const tilingStruct &ti, TPipe *p):gm_x1(x1), gm_x2(x2), ti(ti) {
        SET_GLOBAL(DATA, y, 0);
        INIT_QUEUE_1(uint8_t, q_x1, Constants<3>::MEM); // 4 * miniBatch
        INIT_QUEUE_1(uint8_t, q_x2, Constants<3>::MEM); // 4 * miniBatch
        INIT_QUEUE_1(uint8_t, q_y, Constants<3>::MEM);  // 4 * miniBatch
    }
    __aicore__ inline void exec() {
        DEF_CVAR_FROM_TI(shapeSize);
        DEF_CVAR_FROM_TI(dataLength);
        uint32_t trigger = 0;
        _ALLOC(DATA, q_x1, this->x1);
        _ALLOC(DATA, q_x2, this->x2);
        for (uint32_t y_idx = 0; y_idx < dataLength; y_idx++) {
            uint32_t x1_idx = 0, x2_idx = 0;
            for (uint32_t axis = 0; axis < shapeSize; axis++) {
                x1_idx += TI.x1Shape[axis] != 1 ? TI.x1ShapeRSum[axis] * (y_idx / TI.yShapeRSum[axis] % TI.yShape[axis]) : 0;
                x2_idx += TI.x2Shape[axis] != 1 ? TI.x2ShapeRSum[axis] * (y_idx / TI.yShapeRSum[axis] % TI.yShape[axis]) : 0;
            }
            CopyIn(trigger, x1_idx, x2_idx);
            if(++trigger == Constants<3>::MINI_BATCH32) { // Calculate and copy out every MINI_BATCH32 elements
                Compute();
                CopyOut(y_idx + 1 - Constants<3>::MINI_BATCH32);
#if SYNC_ENABLED
                sync.WaitFlag(0); // Scalar Pipe wait until Vector Pipe SetFlag
#endif
                trigger = 0;
            }
        }
        Compute(trigger);
        CopyOut(dataLength - trigger, CEIL(trigger, 32 / sizeof(DATA)));
#if SYNC_ENABLED
        sync.WaitFlag(0); // Scalar Pipe wait until Vector Pipe SetFlag
#endif
        _FREE(q_x1, this->x1);
        _FREE(q_x2, this->x2);
    }
};
/* ************************************************************************************** */
template<typename DATA, class tilingStruct> class MyPows<DATA, 4, tilingStruct> {       // broadcast sp minibatch ver    
    GM_ADDR gm_x1;
    GM_ADDR gm_x2;
    GM_ADDR gm_y;
    tilingStruct ti;
    MyPows<DATA, 40, tilingInfo> op;
public:
    __aicore__ inline MyPows(   GM_ADDR x1,
                                GM_ADDR x2,
                                GM_ADDR y,
                                const tilingStruct &ti,
                                TPipe *p):  gm_x1(x1),
                                            gm_x2(x2),
                                            gm_y(y), 
                                            ti(ti), 
                                            op(x1, x2, y, tilingInfo{TI.tileLength, TI.maxLength, TI.reminder}, p) { }
    __aicore__ inline void exec() { 
        DEF_CVAR_FROM_TI(bigBatch);
        DEF_CVAR_FROM_TI(shapeSize);
        for (uint32_t i = 0; i < bigBatch; ++i) {
            uint32_t y_idx = i * TI.yShape[0], x1_idx = 0, x2_idx = 0;
            for (uint32_t axis = 0; axis < shapeSize; axis++) {
                x1_idx += TI.x1Shape[axis] != 1 ? TI.x1ShapeRSum[axis] * (y_idx / TI.yShapeRSum[axis] % TI.yShape[axis]) : 0;
                x2_idx += TI.x2Shape[axis] != 1 ? TI.x2ShapeRSum[axis] * (y_idx / TI.yShapeRSum[axis] % TI.yShape[axis]) : 0;
            }
            op.reset_ptr(   gm_x1 + x1_idx * sizeof(DATA), 
                            gm_x2 + x2_idx * sizeof(DATA), 
                            gm_y + y_idx * sizeof(DATA));
            op.exec();
        }
    }
};
/* ************************************************************************************** */
extern "C" __global__ __aicore__ void pows(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe p;
    /* ************************************************************** */
    if(TILING_KEY_IS(1)) {          // non-broadcast
        GET_TILING_DATA_WITH_STRUCT(PowsTilingDataNoBC, ti, tiling);
        MyPows<DTYPE_Y, 1, decltype(ti)>(x1, x2, y, ti, &p).exec();
    }
    /* ************************************************************** */
    else if(TILING_KEY_IS(2)) {     // broadcast 16
        GET_TILING_DATA_WITH_STRUCT(PowsTilingDataBC, ti, tiling);
        MyPows<DTYPE_Y, 2, decltype(ti)>(x1, x2, y, ti, &p).exec();
    }
    /* ************************************************************** */
    else if(TILING_KEY_IS(3)) {     // broadcast 32
        GET_TILING_DATA_WITH_STRUCT(PowsTilingDataBC, ti, tiling);
        MyPows<DTYPE_Y, 3, decltype(ti)>(x1, x2, y, ti, &p).exec();
    } 
    /* ************************************************************** */
    else if(TILING_KEY_IS(4)) {     // broadcast minibatch ver
        GET_TILING_DATA_WITH_STRUCT(PowsTilingDataBCWithMiniBatch, ti, tiling);
        MyPows<DTYPE_Y, 4, decltype(ti)>(x1, x2, y, ti, &p).exec();
    }
    
}