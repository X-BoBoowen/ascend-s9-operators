#include "kernel_inc.h"

template<int tilingKey> struct ReshapeTilingInfo { };

template<> struct ReshapeTilingInfo<1> { 
    uint32_t tileLength, tileNumber, reminder;
    // int32_t axis, numAxes;
};
template<> struct ReshapeTilingInfo<2> { 
    uint32_t tileLength, tileNumber, reminder;
    // int32_t axis, numAxes;
};

template<class DATA, class SHAPE, int tilingKey> class MyReshape {};

template<class DATA, class SHAPE> class MyReshape<DATA, SHAPE, 1> {
    GlobalTensor<DATA> gm_x, gm_y;
    GlobalTensor<SHAPE> gm_shape;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> q_data;
    ReshapeTilingInfo<1> ti;
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length) {
        ALLOC(DATA, data);
        DataCopy(data, gm_x[offset], length);
        ENQUE(DATA, data);
    }
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length) {
        DEQUE(DATA, data);
        DataCopy(gm_y[offset], data, length);
        FREE(data);
    }
public:
    __aicore__ inline MyReshape(GM_ADDR x, GM_ADDR shape, GM_ADDR y, const ReshapeTilingInfo<1> &ti, TPipe *pipe):ti(ti) {
        SET_GLOBAL(DATA, x, 0);
        SET_GLOBAL(SHAPE, shape, 0);
        SET_GLOBAL(DATA, y, 0);
        INIT_QUEUE_2(DATA, q_data, TI.tileLength);
    }
    __aicore__ inline void exec() {
        for(uint32_t i = 0; i < TI.tileNumber; i++) {
            CopyIn(i * TI.tileLength, TI.tileLength);
            CopyOut(i * TI.tileLength, TI.tileLength);
        }
        const uint32_t miniBatch = 32 / sizeof(DATA);
        const uint32_t lastLength = CEIL(TI.reminder, miniBatch);
        CopyIn(TI.tileNumber * TI.tileLength, lastLength);
        CopyOut(TI.tileNumber * TI.tileLength, lastLength);
    }
};

template<class DATA, class SHAPE> class MyReshape<DATA, SHAPE, 2> {
    GM_ADDR x;
    GM_ADDR y;
    ReshapeTilingInfo<2> ti;
    __aicore__ inline void Copy(const uint32_t offset) {
        gm_y.SetValue(offset, (DATA)gm_x.GetValue(offset));
    }
public:
    __aicore__ inline MyReshape(GM_ADDR x, GM_ADDR shape, GM_ADDR y, const ReshapeTilingInfo<2> &ti):x(x), y(y), ti(ti) { }
    __aicore__ inline void exec() {
        const uint32_t lastStart = TI.tileNumber * TI.tileLength;
        for(uint32_t batch = 0; batch < TI.tileNumber; batch++) 
            for(uint32_t miniBatch = 0; miniBatch < TI.tileLength; miniBatch++) 
                Copy(batch * TI.tileLength + miniBatch);
        for(uint32_t miniBatch = 0; miniBatch < TI.reminder; miniBatch++) 
            Copy(lastStart + miniBatch);
    }
};
extern "C" __global__ __aicore__ void reshape(GM_ADDR x, GM_ADDR shape, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    if (TILING_KEY_IS(1)) {
        ReshapeTilingInfo<1> ti{
            TI_COPY_VAR(tileLength),
            TI_COPY_VAR(tileNumber),
            TI_COPY_VAR(reminder),
        };
        TPipe p;
        MyReshape<DTYPE_Y, DTYPE_SHAPE, 1>(x, shape, y, ti, &p).exec();
    } else if(TILING_KEY_IS(2)) {
        ReshapeTilingInfo<2> ti{
            TI_COPY_VAR(tileLength),
            TI_COPY_VAR(tileNumber),
            TI_COPY_VAR(reminder),
        };
        MyReshape<DTYPE_Y, DTYPE_SHAPE, 2>(x, shape, y, ti).exec();
    }
}