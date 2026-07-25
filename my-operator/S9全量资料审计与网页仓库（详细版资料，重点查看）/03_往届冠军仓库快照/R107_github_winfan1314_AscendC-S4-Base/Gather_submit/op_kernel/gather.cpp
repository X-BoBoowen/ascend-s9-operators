#include "kernel_inc.h"

// tiling structs
struct tilingInfo {
    uint32_t maxLength, tileLength, reminder;
};
/* ******************************************************************************************************* */
template<class DATA, int tilingKey, class tilingStruct> class DataCopyHelper { };                           // DataCopyHelper base template
/* ******************************************************************************************************* */
template<class DATA, class tilingStruct> class DataCopyHelper<DATA, 2, tilingStruct> {                      // DataCopyHelper double buffer version
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> q_local;
    GlobalTensor<DATA> gm_x, gm_y;
    tilingStruct ti;
    __aicore__ inline void CopyIn(uint32_t prefix, uint32_t length) {
        ALLOC(DATA, local);
        DataCopy(local, gm_x[prefix], length);
        ENQUE(DATA, local);
    }
    __aicore__ inline void CopyOut(uint32_t prefix, uint32_t length) {
        DEQUE(DATA, local);
        DataCopy(gm_y[prefix], local, length);
        FREE(local);
    }
public:
    __aicore__ inline DataCopyHelper(GM_ADDR x, GM_ADDR y, const tilingStruct &ti, TPipe *p):ti(ti) {
        this->set_ptr(x, y);
        INIT_QUEUE_2(DATA, q_local, TI.tileLength);
    }
    __aicore__ inline DataCopyHelper(const tilingStruct &ti, TPipe *p):ti(ti) {
        INIT_QUEUE_2(DATA, q_local, TI.tileLength);
    }
    __aicore__ inline void set_ptr(GM_ADDR x, GM_ADDR y) {
        SET_GLOBAL(DATA, x, 0);
        SET_GLOBAL(DATA, y, 0);
    }
    __aicore__ inline void exec(GM_ADDR x, GM_ADDR y) {
        this->set_ptr(x, y);
        this->exec();
    }
    __aicore__ inline void exec() {
        uint32_t i;
        for(i = 0; i < TI.maxLength; i += TI.tileLength) {
            CopyIn(i, TI.tileLength);
            CopyOut(i, TI.tileLength);
        }
        CopyIn(i, TI.reminder);
        CopyOut(i, TI.reminder);
    }
};
/* ******************************************************************************************************* */
template<class DATA, class INDICES, int tilingKey, class tilingStruct> class MyGather { };                  // Gather base template
/* ******************************************************************************************************* */
template<class DATA, class INDICES, class tilingStruct> class MyGather<DATA, INDICES, 0, tilingStruct> {    // Gather DataCopy Version
    GM_ADDR x;
    GM_ADDR indices;
    GM_ADDR y;
    tilingStruct ti;
    DataCopyHelper<DATA, 2, tilingInfo> helper;
public:
    __aicore__ inline MyGather( GM_ADDR x, 
                                GM_ADDR indices,
                                GM_ADDR y,
                                const tilingStruct &ti,
                                TPipe *p):  x(x),
                                            indices(indices),
                                            y(y),
                                            ti(ti),
                                            helper({ TI.maxLength, TI.tileLength, TI.reminder }, p) { }
    __aicore__ inline void exec() {
        INDICES index, basePrefix = 0, indicesPrefix = 0, inPrefix = 0, outPrefix = 0;
        for (uint32_t bigBatch = 0; bigBatch < TI.batchNumber; ++bigBatch) {
            for (uint32_t indiceIdx = 0; indiceIdx < TI.indicesLength; ++indiceIdx) {
                index = *(CAST(__gm__ INDICES *, indices) + indicesPrefix + indiceIdx);
                inPrefix = basePrefix + index * TI.sliceLength;
                helper.exec(this->x + inPrefix * sizeof(DATA), this->y + outPrefix * sizeof(DATA));
                // SetFlag<HardEvent::MTE3_S>(0);
                // WaitFlag<HardEvent::MTE3_S>(0);
                // PipeBarrier<PIPE_MTE2>();
                PipeBarrier<PIPE_MTE3>();
                outPrefix += TI.sliceLength;
            }
            basePrefix += TI.batchLength;
            indicesPrefix += TI.indicesLength;
        }
    }
};
/* ******************************************************************************************************* */
template<class DATA, class INDICES, class tilingStruct> class MyGather<DATA, INDICES, 1, tilingStruct> {    // Gather ScalarCopy Version
    GM_ADDR x;
    GM_ADDR indices;
    GM_ADDR y;
    tilingStruct ti;
    __aicore__ inline void CopyScalar(__gm__ DATA *x, __gm__ DATA *y) {
        for(uint32_t i = 0; i < TI.sliceLength; ++i) {
            *(y + i) = *(x + i);
        }
    }
public:
    __aicore__ inline MyGather( GM_ADDR x, 
                                GM_ADDR indices,
                                GM_ADDR y,
                                const tilingStruct &ti):x(x),
                                                        indices(indices),
                                                        y(y),
                                                        ti(ti) { }
    __aicore__ inline void exec() {
        INDICES index, basePrefix = 0, indicesPrefix = 0, inPrefix = 0, outPrefix = 0;
        for (uint32_t bigBatch = 0; bigBatch < TI.batchNumber; ++bigBatch) {
            for (uint32_t indiceIdx = 0; indiceIdx < TI.indicesLength; ++indiceIdx) {
                index = *(CAST(__gm__ INDICES *, indices) + indicesPrefix + indiceIdx);
                inPrefix = basePrefix + index * TI.sliceLength;
                CopyScalar(CAST(__gm__ DATA *, this->x) + inPrefix, CAST(__gm__ DATA *, this->y) + outPrefix);
                outPrefix += TI.sliceLength;
            }
            basePrefix += TI.batchLength;
            indicesPrefix += TI.indicesLength;
        }
    }
};
/* ******************************************************************************************************* */
extern "C" __global__ __aicore__ void gather(GM_ADDR x1, GM_ADDR indices, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    if(TILING_KEY_IS(0)) {      // DataCopy Ver
        GET_TILING_DATA_WITH_STRUCT(GatherTilingDataWithDataCopy, ti, tiling);
        TPipe p;
        MyGather<DTYPE_Y, DTYPE_INDICES, 0, decltype(ti)>(x1, indices, y, ti, &p).exec();
    }
    else if(TILING_KEY_IS(1)) { // ScalarCopy Ver
        GET_TILING_DATA_WITH_STRUCT(GatherTilingDataScalarCopy, ti, tiling);
        MyGather<DTYPE_Y, DTYPE_INDICES, 1, decltype(ti)>(x1, indices, y, ti).exec();
    }
}