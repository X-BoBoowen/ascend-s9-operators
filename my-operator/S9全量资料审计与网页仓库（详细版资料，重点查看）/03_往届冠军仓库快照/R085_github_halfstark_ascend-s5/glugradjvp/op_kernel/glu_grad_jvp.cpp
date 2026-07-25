#include "kernel_operator.h"
#include "nobcastFast.h"

extern "C" __global__ __aicore__ void glu_grad_jvp(GM_ADDR grad_x, GM_ADDR y_grad, GM_ADDR x, GM_ADDR v_y, GM_ADDR v_x, GM_ADDR jvp_out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 增加这一行
    // printf("tiling_data.padtimes:%d\n", tiling_data.padtimes);
    if constexpr (std::is_same_v<DTYPE_X, float>) {
            NoBcastFastFloat<DTYPE_X>op;
            op.Init(grad_x, y_grad, x, v_y, v_x, jvp_out,
                tiling_data.finalTileNum,tiling_data.tileDataNum,tiling_data.tailDataNum,
                tiling_data.iterStep, tiling_data.smallBatch, tiling_data.tail,tiling_data.padtimes,&pipe );
            op.process();
    } else if constexpr (std::is_same_v<DTYPE_X, half>) {
        NoBcastFastHalf<DTYPE_X>op;
        op.Init(grad_x,y_grad, x, v_y, v_x, jvp_out,
            tiling_data.finalTileNum,tiling_data.tileDataNum,tiling_data.tailDataNum,
            tiling_data.iterStep, tiling_data.smallBatch, tiling_data.tail,tiling_data.padtimes,&pipe );
            op.process();
    }else if constexpr (std::is_same_v<DTYPE_X, bfloat16_t>) {
        NoBcastFastHalf<DTYPE_X>op;
        op.Init(grad_x, y_grad, x, v_y, v_x, jvp_out,
            tiling_data.finalTileNum,tiling_data.tileDataNum,tiling_data.tailDataNum,
            tiling_data.iterStep, tiling_data.smallBatch, tiling_data.tail,tiling_data.padtimes,&pipe );
            op.process();
    }
}    // TODO: user kernel impl