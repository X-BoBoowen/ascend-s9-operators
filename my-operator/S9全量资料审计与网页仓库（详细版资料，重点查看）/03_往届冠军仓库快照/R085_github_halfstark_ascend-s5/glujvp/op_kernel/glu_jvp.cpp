#include "kernel_operator.h"
// #include "nobcast.h"
#include "nobcastFast.h"

extern "C" __global__ __aicore__ void glu_jvp(GM_ADDR glu_out, GM_ADDR input, GM_ADDR v, GM_ADDR jvp_out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 增加这一行

    if constexpr (std::is_same_v<DTYPE_V, float>) {
            NoBcastFastFloat<DTYPE_V>op;
            op.Init(glu_out, input, v, jvp_out,
                tiling_data.finalTileNum,tiling_data.tileDataNum,tiling_data.tailDataNum,
                tiling_data.iterStep, tiling_data.smallBatch,  tiling_data.tail,tiling_data.padtimes,&pipe );
            op.process();
    } else if constexpr (std::is_same_v<DTYPE_V, half>) {
            // printf("enter half\n");
            NoBcastFastHalf<half>op;
            op.Init(glu_out, input, v, jvp_out,
                tiling_data.finalTileNum,tiling_data.tileDataNum,tiling_data.tailDataNum,
                tiling_data.iterStep, tiling_data.smallBatch,  tiling_data.tail,tiling_data.padtimes,&pipe );
            op.process();
    }else if constexpr (std::is_same_v<DTYPE_V, bfloat16_t>) {
            NoBcastFastHalf<bfloat16_t>op;
            op.Init(glu_out, input, v, jvp_out,
                tiling_data.finalTileNum,tiling_data.tileDataNum,tiling_data.tailDataNum,
                tiling_data.iterStep, tiling_data.smallBatch, tiling_data.tail, tiling_data.padtimes,&pipe );
            op.process();
    }
}