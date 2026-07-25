#include "kernel_operator.h"
#include "pow.h"
#include "powb.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void pows(GM_ADDR x1,GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    if(TILING_KEY_IS(1)){ // fp16非广播
        KernelPows op;
        op.Init(x1,x2, y, tiling,&pipe);
        op.Process();
    }else if(TILING_KEY_IS(2)){ // fp32非广播
         op;
        op.Init(x1,x2, y, tiling,&pipe);
        op.Process();
    }else if(TILING_KEY_IS(3)){ // fp16广播
        KernelPowsBroadCastB op;
        op.Init(x1,x2, y, tiling,&pipe);
        op.Process();
    }else if(TILING_KEY_IS(4)){ // fp32广播
        KernelPowsBroadCastBB op;
}
