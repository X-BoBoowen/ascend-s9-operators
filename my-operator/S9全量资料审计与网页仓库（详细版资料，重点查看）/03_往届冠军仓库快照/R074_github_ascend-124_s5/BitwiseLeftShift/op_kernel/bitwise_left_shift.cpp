#include "kernel_operator.h"

using namespace AscendC;

__aicore__ inline constexpr int ceil_div(int x, int y)
{
    return (x - 1) / y + 1;
}

__aicore__ inline constexpr int ceil_round(int x, int y)
{
    return ceil_div(x, y) * y;
}

__aicore__ inline void bitwise_left_shift_int8(GM_ADDR input, GM_ADDR other, GM_ADDR out, BitwiseLeftShiftTilingData &tiling)
{
    using T = int8_t;
    using U = short;

    int block_index = GetBlockIdx();
    int block_dim = GetBlockNum();
    constexpr int DATA_BLOCK_SIZE = 32 / sizeof(T);
    int compute_blocks = ceil_div(tiling.size, DATA_BLOCK_SIZE);
    int compute_start = compute_blocks * block_index / block_dim * DATA_BLOCK_SIZE;
    int compute_end = compute_blocks * (block_index + 1) / block_dim * DATA_BLOCK_SIZE;

    GlobalTensor<T> input_global_tensor, other_global_tensor, out_global_tensor;
    input_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
    other_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(other));
    out_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));

    TPipe t_pipe;
    TQue<TPosition::VECIN, 1> input_t_que, other_t_que;
    TQue<TPosition::VECOUT, 1> out_t_que;
    TBuf<> bit_t_buf;

    constexpr int MAX_TILE_SIZE = (20 << 10) / sizeof(U);
    t_pipe.InitBuffer(input_t_que, 2, MAX_TILE_SIZE * sizeof(U));
    t_pipe.InitBuffer(other_t_que, 2, MAX_TILE_SIZE * sizeof(U));
    t_pipe.InitBuffer(out_t_que, 2, MAX_TILE_SIZE * sizeof(int));
    t_pipe.InitBuffer(bit_t_buf, 512);

    bool first = true;
    LocalTensor<U> bit_local_tensor = bit_t_buf.Get<U>();

    for (int i = compute_start; i < compute_end; i += MAX_TILE_SIZE)
    {
        int tile_size = min(compute_end - i, MAX_TILE_SIZE);
        //
        {
            LocalTensor<T> input_local_tensor = input_t_que.AllocTensor<T>();
            LocalTensor<T> other_local_tensor = other_t_que.AllocTensor<T>();
            DataCopy(input_local_tensor, input_global_tensor[i], tile_size);
            DataCopy(other_local_tensor, other_global_tensor[i], tile_size);
            input_t_que.EnQue(input_local_tensor);
            other_t_que.EnQue(other_local_tensor);
        }
        //
        {
            LocalTensor<U> out_local_tensor = out_t_que.AllocTensor<U>();
            LocalTensor<T> out_local_tensor_t = out_local_tensor.ReinterpretCast<T>();
            LocalTensor<uint16_t> out_local_tensor_uint16 = out_local_tensor.ReinterpretCast<uint16_t>();
            LocalTensor<int> out_local_tensor_int = out_local_tensor.ReinterpretCast<int>();
            LocalTensor<unsigned> out_local_tensor_unsigned = out_local_tensor.ReinterpretCast<unsigned>();
            LocalTensor<float> out_local_tensor_float = out_local_tensor.ReinterpretCast<float>();
            LocalTensor<half> out_local_tensor_half = out_local_tensor.ReinterpretCast<half>();
            if (first)
            {
                first = false;
                for (int j = 0; j < 8 * sizeof(T); j++)
                    bit_local_tensor.SetValue(j, static_cast<U>(1 << j));
            }
            LocalTensor<U> input_local_tensor = input_t_que.DeQue<U>();
            LocalTensor<T> input_local_tensor_t = input_local_tensor.ReinterpretCast<T>();
            LocalTensor<U> other_local_tensor = other_t_que.DeQue<U>();
            LocalTensor<T> other_local_tensor_t = other_local_tensor.ReinterpretCast<T>();
            Cast(out_local_tensor_half, input_local_tensor_t, RoundMode::CAST_NONE, tile_size);
            Cast(input_local_tensor, out_local_tensor_half, RoundMode::CAST_RINT, tile_size);
            Cast(out_local_tensor_half, other_local_tensor_t, RoundMode::CAST_NONE, tile_size);
            Cast(other_local_tensor, out_local_tensor_half, RoundMode::CAST_RINT, tile_size);
            Muls(other_local_tensor, other_local_tensor, static_cast<U>(sizeof(U)), tile_size);
            Cast(out_local_tensor_float, other_local_tensor, RoundMode::CAST_NONE, tile_size);
            Cast(out_local_tensor_int, out_local_tensor_float, RoundMode::CAST_RINT, tile_size);
            Gather(out_local_tensor, bit_local_tensor, out_local_tensor_unsigned, 0, tile_size);
            Mul(out_local_tensor, out_local_tensor, input_local_tensor, tile_size);
            ShiftLeft(out_local_tensor, out_local_tensor, static_cast<U>(8 * sizeof(U) / 2), tile_size);
            ShiftRight(out_local_tensor, out_local_tensor, static_cast<U>(8 * sizeof(U) / 2), tile_size);
            Cast(out_local_tensor_half, out_local_tensor, RoundMode::CAST_RINT, tile_size);
            Cast(out_local_tensor_t, out_local_tensor_half, RoundMode::CAST_RINT, tile_size);
            input_t_que.FreeTensor(input_local_tensor);
            other_t_que.FreeTensor(other_local_tensor);
            out_t_que.EnQue(out_local_tensor);
        }
        //
        {
            LocalTensor<T> out_local_tensor = out_t_que.DeQue<T>();
            DataCopy(out_global_tensor[i], out_local_tensor, tile_size);
            out_t_que.FreeTensor(out_local_tensor);
        }
    }
}

__aicore__ inline void bitwise_left_shift_int(GM_ADDR input, GM_ADDR other, GM_ADDR out, BitwiseLeftShiftTilingData &tiling)
{
    using T = int;
    using U = int;

    int block_index = GetBlockIdx();
    int block_dim = GetBlockNum();
    constexpr int DATA_BLOCK_SIZE = 32 / sizeof(T);
    int compute_blocks = ceil_div(tiling.size, DATA_BLOCK_SIZE);
    int compute_start = compute_blocks * block_index / block_dim * DATA_BLOCK_SIZE;
    int compute_end = compute_blocks * (block_index + 1) / block_dim * DATA_BLOCK_SIZE;

    GlobalTensor<T> input_global_tensor, other_global_tensor, out_global_tensor;
    input_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
    other_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(other));
    out_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));

    TPipe t_pipe;
    TQue<TPosition::VECIN, 1> input_t_que, other_t_que;
    TQue<TPosition::VECOUT, 1> out_t_que;
    TBuf<> bit_t_buf;

    constexpr int MAX_TILE_SIZE = (28 << 10) / sizeof(U);
    t_pipe.InitBuffer(input_t_que, 2, MAX_TILE_SIZE * sizeof(U));
    t_pipe.InitBuffer(other_t_que, 2, MAX_TILE_SIZE * sizeof(U));
    t_pipe.InitBuffer(out_t_que, 2, MAX_TILE_SIZE * sizeof(U));
    t_pipe.InitBuffer(bit_t_buf, 512);

    bool first = true;
    LocalTensor<U> bit_local_tensor = bit_t_buf.Get<U>();

    for (int i = compute_start; i < compute_end; i += MAX_TILE_SIZE)
    {
        int tile_size = min(compute_end - i, MAX_TILE_SIZE);
        //
        {
            LocalTensor<T> input_local_tensor = input_t_que.AllocTensor<T>();
            LocalTensor<T> other_local_tensor = other_t_que.AllocTensor<T>();
            DataCopy(input_local_tensor, input_global_tensor[i], tile_size);
            DataCopy(other_local_tensor, other_global_tensor[i], tile_size);
            input_t_que.EnQue(input_local_tensor);
            other_t_que.EnQue(other_local_tensor);
        }
        //
        {
            LocalTensor<U> out_local_tensor = out_t_que.AllocTensor<U>();
            if (first)
            {
                first = false;
                for (int j = 0; j < 8 * sizeof(T); j++)
                    bit_local_tensor.SetValue(j, static_cast<U>(1 << j));
            }
            LocalTensor<U> input_local_tensor = input_t_que.DeQue<U>();
            LocalTensor<U> other_local_tensor = other_t_que.DeQue<U>();
            LocalTensor<unsigned> other_local_tensor_unsigned = other_local_tensor.ReinterpretCast<unsigned>();
            Muls(other_local_tensor, other_local_tensor, static_cast<U>(sizeof(U)), tile_size);
            Gather(out_local_tensor, bit_local_tensor, other_local_tensor_unsigned, 0, tile_size);
            Mul(out_local_tensor, out_local_tensor, input_local_tensor, tile_size);
            input_t_que.FreeTensor(input_local_tensor);
            other_t_que.FreeTensor(other_local_tensor);
            out_t_que.EnQue(out_local_tensor);
        }
        //
        {
            LocalTensor<T> out_local_tensor = out_t_que.DeQue<T>();
            DataCopy(out_global_tensor[i], out_local_tensor, tile_size);
            out_t_que.FreeTensor(out_local_tensor);
        }
    }
}

__aicore__ inline void bitwise_left_shift_short(GM_ADDR input, GM_ADDR other, GM_ADDR out, BitwiseLeftShiftTilingData &tiling)
{
    using T = short;
    using U = short;

    int block_index = GetBlockIdx();
    int block_dim = GetBlockNum();
    constexpr int DATA_BLOCK_SIZE = 32 / sizeof(T);
    int compute_blocks = ceil_div(tiling.size, DATA_BLOCK_SIZE);
    int compute_start = compute_blocks * block_index / block_dim * DATA_BLOCK_SIZE;
    int compute_end = compute_blocks * (block_index + 1) / block_dim * DATA_BLOCK_SIZE;

    GlobalTensor<T> input_global_tensor, other_global_tensor, out_global_tensor;
    input_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
    other_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(other));
    out_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));

    TPipe t_pipe;
    TQue<TPosition::VECIN, 1> input_t_que, other_t_que;
    TQue<TPosition::VECOUT, 1> out_t_que;
    TBuf<> bit_t_buf;

    constexpr int MAX_TILE_SIZE = (20 << 10) / sizeof(U);
    t_pipe.InitBuffer(input_t_que, 2, MAX_TILE_SIZE * sizeof(U));
    t_pipe.InitBuffer(other_t_que, 2, MAX_TILE_SIZE * sizeof(U));
    t_pipe.InitBuffer(out_t_que, 2, MAX_TILE_SIZE * sizeof(int));
    t_pipe.InitBuffer(bit_t_buf, 512);

    bool first = true;
    LocalTensor<U> bit_local_tensor = bit_t_buf.Get<U>();

    for (int i = compute_start; i < compute_end; i += MAX_TILE_SIZE)
    {
        int tile_size = min(compute_end - i, MAX_TILE_SIZE);
        //
        {
            LocalTensor<T> input_local_tensor = input_t_que.AllocTensor<T>();
            LocalTensor<T> other_local_tensor = other_t_que.AllocTensor<T>();
            DataCopy(input_local_tensor, input_global_tensor[i], tile_size);
            DataCopy(other_local_tensor, other_global_tensor[i], tile_size);
            input_t_que.EnQue(input_local_tensor);
            other_t_que.EnQue(other_local_tensor);
        }
        //
        {
            LocalTensor<U> out_local_tensor = out_t_que.AllocTensor<U>();
            LocalTensor<int> out_local_tensor_int = out_local_tensor.ReinterpretCast<int>();
            LocalTensor<unsigned> out_local_tensor_unsigned = out_local_tensor.ReinterpretCast<unsigned>();
            LocalTensor<float> out_local_tensor_float = out_local_tensor.ReinterpretCast<float>();
            if (first)
            {
                first = false;
                for (int j = 0; j < 8 * sizeof(T); j++)
                    bit_local_tensor.SetValue(j, static_cast<U>(1 << j));
            }
            LocalTensor<U> input_local_tensor = input_t_que.DeQue<U>();
            LocalTensor<U> other_local_tensor = other_t_que.DeQue<U>();
            Muls(other_local_tensor, other_local_tensor, static_cast<U>(sizeof(U)), tile_size);
            Cast(out_local_tensor_float, other_local_tensor, RoundMode::CAST_NONE, tile_size);
            Cast(out_local_tensor_int, out_local_tensor_float, RoundMode::CAST_RINT, tile_size);
            Gather(out_local_tensor, bit_local_tensor, out_local_tensor_unsigned, 0, tile_size);
            Mul(out_local_tensor, out_local_tensor, input_local_tensor, tile_size);
            input_t_que.FreeTensor(input_local_tensor);
            other_t_que.FreeTensor(other_local_tensor);
            out_t_que.EnQue(out_local_tensor);
        }
        //
        {
            LocalTensor<T> out_local_tensor = out_t_que.DeQue<T>();
            DataCopy(out_global_tensor[i], out_local_tensor, tile_size);
            out_t_que.FreeTensor(out_local_tensor);
        }
    }
}

__aicore__ inline void bitwise_left_shift_long(GM_ADDR input, GM_ADDR other, GM_ADDR out, BitwiseLeftShiftTilingData &tiling)
{
    using T = long;
    using U = int;

    int block_index = GetBlockIdx();
    int block_dim = GetBlockNum();
    constexpr int DATA_BLOCK_SIZE = 32 / sizeof(T);
    int compute_blocks = ceil_div(tiling.size, DATA_BLOCK_SIZE);
    int compute_start = compute_blocks * block_index / block_dim * DATA_BLOCK_SIZE;
    int compute_end = compute_blocks * (block_index + 1) / block_dim * DATA_BLOCK_SIZE;

    GlobalTensor<T> input_global_tensor, other_global_tensor, out_global_tensor;
    input_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
    other_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(other));
    out_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));

    TPipe t_pipe;
    TQue<TPosition::VECIN, 1> input_t_que, other_t_que;
    TQue<TPosition::VECOUT, 1> out_t_que;
    TBuf<> bit_t_buf;
    TBuf<> index_t_buf;

    constexpr int MAX_TILE_SIZE = (20 << 10) / sizeof(T);
    t_pipe.InitBuffer(input_t_que, 2, MAX_TILE_SIZE * sizeof(T));
    t_pipe.InitBuffer(other_t_que, 2, MAX_TILE_SIZE * sizeof(T));
    t_pipe.InitBuffer(out_t_que, 2, MAX_TILE_SIZE * sizeof(T));
    t_pipe.InitBuffer(bit_t_buf, MAX_TILE_SIZE * sizeof(T));
    t_pipe.InitBuffer(index_t_buf, MAX_TILE_SIZE * sizeof(T));

    LocalTensor<unsigned> index_local_tensor = index_t_buf.Get<unsigned>();
    CreateVecIndex(index_local_tensor.ReinterpretCast<int>(), 0, MAX_TILE_SIZE * 2);
    ShiftRight(index_local_tensor, index_local_tensor, 1u, MAX_TILE_SIZE * 2);
    ShiftLeft(index_local_tensor, index_local_tensor, 3u, MAX_TILE_SIZE * 2);

    for (int j = compute_start; j < compute_end; j += MAX_TILE_SIZE)
    {
        int tile_size = min(compute_end - j, MAX_TILE_SIZE);
        //
        {
            LocalTensor<T> input_local_tensor = input_t_que.AllocTensor<T>();
            LocalTensor<T> other_local_tensor = other_t_que.AllocTensor<T>();
            DataCopy(input_local_tensor, input_global_tensor[j], tile_size);
            DataCopy(other_local_tensor, other_global_tensor[j], tile_size);
            input_t_que.EnQue(input_local_tensor);
            other_t_que.EnQue(other_local_tensor);
        }
        //
        {
            LocalTensor<U> input_local_tensor = input_t_que.DeQue<U>();
            LocalTensor<float> input_local_tensor_float = input_local_tensor.ReinterpretCast<float>();
            LocalTensor<U> other_local_tensor = other_t_que.DeQue<U>();
            LocalTensor<U> bit_local_tensor = bit_t_buf.Get<U>();
            LocalTensor<unsigned> bit_local_tensor_unsigned = bit_local_tensor.ReinterpretCast<unsigned>();
            LocalTensor<short> bit_local_tensor_short = bit_local_tensor.ReinterpretCast<short>();
            LocalTensor<uint8_t> bit_local_tensor_uint8 = bit_local_tensor.ReinterpretCast<uint8_t>();
            LocalTensor<U> out_local_tensor = out_t_que.AllocTensor<U>();
            LocalTensor<float> out_local_tensor_float = out_local_tensor.ReinterpretCast<float>();
            LocalTensor<short> out_local_tensor_short = out_local_tensor.ReinterpretCast<short>();
            Gather(other_local_tensor, other_local_tensor, index_local_tensor, 0, tile_size * 2);
            constexpr int U_BIT_WIDTH = 8 * sizeof(U);
            int l = 1;
            for (int k = 1; k < U_BIT_WIDTH; k <<= 1, l++)
            {
                ShiftLeft(out_local_tensor, input_local_tensor, static_cast<U>(k), tile_size * 2);
                Gather(bit_local_tensor, input_local_tensor, index_local_tensor, 0, tile_size * 2);
                ShiftRight(bit_local_tensor_unsigned, bit_local_tensor_unsigned, static_cast<unsigned>(U_BIT_WIDTH - k), tile_size * 2);
                Or(out_local_tensor_short, out_local_tensor_short, bit_local_tensor_short, (uint64_t[]){0xcccccccccccccccc, 0xcccccccccccccccc}, ceil_div(tile_size, 32), {});
                ShiftLeft(bit_local_tensor, other_local_tensor, static_cast<U>(U_BIT_WIDTH - l), tile_size * 2);
                ShiftRight(bit_local_tensor, bit_local_tensor, static_cast<U>(U_BIT_WIDTH - 1), tile_size * 2);
                CompareScalar(bit_local_tensor_uint8, bit_local_tensor, 0, CMPMODE::EQ, ceil_round(tile_size * 2, 256 / sizeof(U)));
                Select(input_local_tensor_float, bit_local_tensor_uint8, input_local_tensor_float, out_local_tensor_float, SELMODE::VSEL_TENSOR_TENSOR_MODE, tile_size * 2);
            }
            Duplicate(out_local_tensor, 0, tile_size * 2);
            Gather(out_local_tensor, input_local_tensor, index_local_tensor, 0, (uint64_t[]){0xaaaaaaaaaaaaaaaa, 0}, ceil_div(tile_size, 32), 8);
            ShiftLeft(bit_local_tensor, other_local_tensor, static_cast<U>(U_BIT_WIDTH - l), tile_size * 2);
            ShiftRight(bit_local_tensor, bit_local_tensor, static_cast<U>(U_BIT_WIDTH - 1), tile_size * 2);
            CompareScalar(bit_local_tensor_uint8, bit_local_tensor, 0, CMPMODE::EQ, ceil_round(tile_size * 2, 256 / sizeof(U)));
            Select(out_local_tensor_float, bit_local_tensor_uint8, input_local_tensor_float, out_local_tensor_float, SELMODE::VSEL_TENSOR_TENSOR_MODE, tile_size * 2);
            input_t_que.FreeTensor(input_local_tensor);
            other_t_que.FreeTensor(other_local_tensor);
            out_t_que.EnQue(out_local_tensor);
        }
        //
        {
            LocalTensor<T> out_local_tensor = out_t_que.DeQue<T>();
            DataCopy(out_global_tensor[j], out_local_tensor, tile_size);
            out_t_que.FreeTensor(out_local_tensor);
        }
    }
}

template<typename T>
__aicore__ inline void broadcast(GM_ADDR other, GM_ADDR out, GM_ADDR workspace, BitwiseLeftShiftTilingData &tiling)
{
    int block_index = GetBlockIdx();
    int block_dim = GetBlockNum();
    constexpr int DATA_BLOCK_SIZE = 32 / sizeof(T);

    int broadcast_count = 0;
    for (int i = 0; i < 3; i++)
    {
        if (tiling.input_n[i] != tiling.other_n[i])
            broadcast_count++;
    }

    GlobalTensor<T> out_global_tensor, workspace_global_tensor;
    out_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(broadcast_count % 2 == 0 ? out : workspace));
    workspace_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(other));

    TPipe t_pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> t_que_bind;
    constexpr int MAX_TILE_SIZE = (92 << 10) / sizeof(T);
    t_pipe.InitBuffer(t_que_bind, 2, MAX_TILE_SIZE * sizeof(T));

    int high_size = tiling.other_n[0] * tiling.other_n[1] * tiling.other_n[2];
    int low_size = 1;
    for (int i = 0; i < 3; i++)
    {
        high_size /= tiling.other_n[i];
        if (tiling.input_n[i] != tiling.other_n[i])
        {
            int mid_size = tiling.input_n[i];
            int high_from, high_to, low_from, low_to;
            if (low_size > MAX_TILE_SIZE)
            {
                high_from = 0;
                high_to = high_size;
                int low_blocks = ceil_div(low_size, DATA_BLOCK_SIZE);
                low_from = low_blocks * block_index / block_dim * DATA_BLOCK_SIZE;
                low_to = min(low_blocks * (block_index + 1) / block_dim * DATA_BLOCK_SIZE, low_size);
            }
            else
            {
                high_from = high_size * block_index / block_dim;
                high_to = high_size * (block_index + 1) / block_dim;
                low_from = 0;
                low_to = low_size;
            }
            for (int j = high_from; j < high_to; j++)
            {
                for (int k = low_from; k < low_to; k += MAX_TILE_SIZE)
                {
                    int tile_size = min(low_to - k, MAX_TILE_SIZE);
                    //
                    {
                        LocalTensor<T> workspace_local_tensor = t_que_bind.AllocTensor<T>();
                        DataCopyExtParams data_copy_ext_params;
                        data_copy_ext_params.blockLen = tile_size * sizeof(T);
                        DataCopyPad(workspace_local_tensor, workspace_global_tensor[low_size * j + k], data_copy_ext_params, {});
                        t_que_bind.EnQue(workspace_local_tensor);
                    }
                    //
                    {
                        LocalTensor<T> out_local_tensor = t_que_bind.DeQue<T>();
                        DataCopyExtParams data_copy_ext_params;
                        data_copy_ext_params.blockLen = tile_size * sizeof(T);
                        for (int l = 0; l < mid_size; l++)
                            DataCopyPad(out_global_tensor[(j * mid_size + l) * low_size + k], out_local_tensor, data_copy_ext_params);
                        t_que_bind.FreeTensor(out_local_tensor);
                    }
                }
            }
            if (--broadcast_count % 2 == 0)
            {
                out_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));
                workspace_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace));
            }
            else
            {
                out_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace));
                workspace_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));
            }
            CrossCoreSetFlag<0, PIPE_MTE3>(0);
            CrossCoreWaitFlag(0);
        }
        low_size *= tiling.input_n[i];
    }

    t_pipe.Destroy();
}

extern "C" __global__ __aicore__ void bitwise_left_shift(GM_ADDR input, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA_WITH_STRUCT(BitwiseLeftShiftTilingData, tiling_data, tiling);
    if (TILING_KEY_IS(2))
        bitwise_left_shift_int8(input, other, out, tiling_data);
    else if (TILING_KEY_IS(3))
        bitwise_left_shift_int(input, other, out, tiling_data);
    else if (TILING_KEY_IS(6))
        bitwise_left_shift_short(input, other, out, tiling_data);
    else if (TILING_KEY_IS(9))
        bitwise_left_shift_long(input, other, out, tiling_data);
    else if (TILING_KEY_IS(102))
    {
        KERNEL_TASK_TYPE(102, KERNEL_TYPE_MIX_AIV_1_0)
        broadcast<int8_t>(other, out, workspace, tiling_data);
        bitwise_left_shift_int8(input, workspace, out, tiling_data);
    }
    else if (TILING_KEY_IS(103))
    {
        KERNEL_TASK_TYPE(103, KERNEL_TYPE_MIX_AIV_1_0)
        broadcast<int>(other, out, workspace, tiling_data);
        bitwise_left_shift_int(input, workspace, out, tiling_data);
    }
    else if (TILING_KEY_IS(106))
    {
        KERNEL_TASK_TYPE(106, KERNEL_TYPE_MIX_AIV_1_0)
        broadcast<short>(other, out, workspace, tiling_data);
        bitwise_left_shift_short(input, workspace, out, tiling_data);
    }
    else if (TILING_KEY_IS(109))
    {
        KERNEL_TASK_TYPE(109, KERNEL_TYPE_MIX_AIV_1_0)
        broadcast<long>(other, out, workspace, tiling_data);
        bitwise_left_shift_long(input, workspace, out, tiling_data);
    }
    else
        Trap();
}
