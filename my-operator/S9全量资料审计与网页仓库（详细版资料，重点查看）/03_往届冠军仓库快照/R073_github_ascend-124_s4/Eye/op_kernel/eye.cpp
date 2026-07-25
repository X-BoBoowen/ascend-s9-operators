#include "kernel_operator.h"

using namespace AscendC;

template<typename T>
__aicore__ inline void eye(GM_ADDR y, EyeTilingData &tiling)
{
    int block_index = GetBlockIdx();
    int block_dim = GetBlockNum();
    int batch_offset = tiling.batch_size * block_index / block_dim;
    int batch_size = tiling.batch_size * (block_index + 1) / block_dim - batch_offset;
    int num_rows = tiling.num_rows;
    int num_columns = tiling.num_columns;
    int num = min(num_rows, num_columns);

    GlobalTensor<T> y_global_tensor;
    y_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));

    int length = num_rows * num_columns * sizeof(T);
    if (length > 192 << 10)
    {
        TPipe t_pipe;
        TQue<TPosition::VECOUT, 1> y_t_que;
        constexpr int BLOCK_LEN = 128;
        t_pipe.InitBuffer(y_t_que, 1, BLOCK_LEN * BLOCK_LEN * sizeof(T));

        //
        {
            LocalTensor<T> y_local_tensor = y_t_que.AllocTensor<T>();
            Duplicate(y_local_tensor.template ReinterpretCast<short>(), static_cast<short>(0), BLOCK_LEN * BLOCK_LEN * sizeof(T) / sizeof(short));
            for (int i = 0; i < BLOCK_LEN; i++)
                y_local_tensor.SetValue(i * (BLOCK_LEN + 1), static_cast<T>(1));
            y_t_que.EnQue(y_local_tensor);
        }

        LocalTensor<T> y_local_tensor = y_t_que.DeQue<T>();
        DataCopyParams data_copy_params_1;
        data_copy_params_1.blockCount = BLOCK_LEN;
        data_copy_params_1.blockLen = BLOCK_LEN * sizeof(T);
        data_copy_params_1.dstStride = (num_columns - BLOCK_LEN) * sizeof(T);
        DataCopyPad(y_global_tensor, y_local_tensor, data_copy_params_1);
        DataCopyParams data_copy_params_2;
        data_copy_params_2.blockCount = 1;
        for (int i = 0; i < batch_size; i++)
        {
            int j;
            for (j = 0; j < num - BLOCK_LEN; j += BLOCK_LEN)
                DataCopyPad(y_global_tensor[(batch_offset + i) * num_rows * num_columns + j * (num_columns + 1)], y_local_tensor, data_copy_params_1);
            if (i == 0)
            {
                data_copy_params_2.blockCount = num - j;
                data_copy_params_2.blockLen = (num - j) * sizeof(T);
                data_copy_params_2.srcStride = (data_copy_params_1.blockLen - data_copy_params_2.blockLen) / 32;
                data_copy_params_2.dstStride = num_columns * sizeof(T) - data_copy_params_2.blockLen;
            }
            DataCopyPad(y_global_tensor[(batch_offset + i) * num_rows * num_columns + j * (num_columns + 1)], y_local_tensor, data_copy_params_2);
        }
        y_t_que.FreeTensor(y_local_tensor);
    }
    else
    {
        TPipe t_pipe;
        TBuf<> y_t_buf;
        t_pipe.InitBuffer(y_t_buf, length);

        LocalTensor<T> y_local_tensor = y_t_buf.AllocTensor<T>();
        Duplicate(y_local_tensor.template ReinterpretCast<short>(), static_cast<short>(0), num_rows * num_columns * sizeof(T) / sizeof(short));
        for (int i = 0; i < num; i++)
            y_local_tensor.SetValue(i * (num_columns + 1), static_cast<T>(1));

        if (num_rows * num_columns * sizeof(T) % 32 == 0)
        {
            for (int i = 0; i < batch_size; i++)
                DataCopy(y_global_tensor[(batch_offset + i) * num_rows * num_columns], y_local_tensor, num_columns * num_columns);
        }
        else
        {
            DataCopyExtParams data_copy_ext_params;
            data_copy_ext_params.blockCount = 1;
            data_copy_ext_params.blockLen = length;
            for (int i = 0; i < batch_size; i++)
                DataCopyPad(y_global_tensor[(batch_offset + i) * num_rows * num_columns], y_local_tensor, data_copy_ext_params);
        }
    }
}

extern "C" __global__ __aicore__ void eye(GM_ADDR y, GM_ADDR y_ref, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA_WITH_STRUCT(EyeTilingData, tiling_data, tiling);
    if (TILING_KEY_IS(0))
        eye<float>(y, tiling_data);
    else if (TILING_KEY_IS(1))
        eye<half>(y, tiling_data);
    else if (TILING_KEY_IS(2))
        eye<double>(y, tiling_data);
    else if (TILING_KEY_IS(3))
        eye<int>(y, tiling_data);
    else
        Trap();
}
