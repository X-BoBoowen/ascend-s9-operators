#include "../op_host/common.h"
#include "copysign_mult_sca.h"
#include "copysign_spec0.h"
#include "copysign_spec1.h"
#include "copysign_spec2.h"
#include "copysign_spec3.h"
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void copysign(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace,
                                               GM_ADDR tiling) {
    TPipe pipe;

    if (TILING_KEY_IS(0)) {
        GET_TILING_DATA_WITH_STRUCT(CopysignTilingData_0, tiling_data, tiling);
        CopysignSpec0<DTYPE_X1> spec0;
        spec0.Init(x1, x2, y, tiling_data.totalNum, &pipe);
        spec0.Process();
    } else if (TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(CopysignTilingData_0, tiling_data, tiling);
        CopysignSpec1<DTYPE_X1> spec1;
        spec1.Init(x1, x2, y, tiling_data.totalNum, &pipe);
        spec1.Process();

    } else if (TILING_KEY_IS(2)) {
        GET_TILING_DATA_WITH_STRUCT(CopysignTilingData_2, tiling_data, tiling);
        CopysignSpec2<DTYPE_X1> spec2;
        spec2.Init(x1, x2, y, tiling_data.shapeInf, &pipe);
        spec2.Process();
    } else if (TILING_KEY_IS(3)) {
        GET_TILING_DATA_WITH_STRUCT(CopysignTilingData_2, tiling_data, tiling);
        CopysignSpec3<DTYPE_X1> spec3;
        spec3.Init(x1, x2, y, tiling_data.shapeInf, &pipe);
        spec3.Process();
    } else if (TILING_KEY_IS(4)) {
        GET_TILING_DATA(tiling_data, tiling);
        CopysignMultSca<DTYPE_X1> sca;
        sca.Init(x1, x2, y, tiling_data.maxDim, tiling_data.shapeInf1, tiling_data.shapeInf2,
                 &pipe);
        sca.Process();
    }
}