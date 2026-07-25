#include "kernel_operator.h"
#include "glu_grad_jvp_sca.h"
#include "glu_grad_jvp_fp32.h"
#include "glu_grad_jvp_16.h"
#include "glu_grad_jvp_fp32_small.h"
extern "C" __global__ __aicore__ void glu_grad_jvp(GM_ADDR x_grad, GM_ADDR y_grad, GM_ADDR x, GM_ADDR v_y, GM_ADDR v_x, GM_ADDR jvp_out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    // TODO: user kernel impl
    TPipe pipe;

    if (TILING_KEY_IS(1)) {
        KernelGluGradJvp16<DTYPE_X> op;
        op.Init(x_grad, y_grad, x, v_y, v_x, jvp_out, tiling_data.HI, tiling_data.J, tiling_data.KS, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelGluGradJvpFp32 op;
        op.Init(x_grad, y_grad, x, v_y, v_x, jvp_out, tiling_data.HI, tiling_data.J, tiling_data.KS, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelGluGradJvpFp32Small op;
        op.Init(x_grad, y_grad, x, v_y, v_x, jvp_out, tiling_data.HI, tiling_data.J, tiling_data.KS, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, &pipe);
        op.Process();
    }
}