#include "kernel_operator.h"

using namespace AscendC;

template<typename T, typename... Ts>
struct is_one_of : std::false_type {};

template<typename T, typename U, typename... Ts>
struct is_one_of<T, U, Ts...> : std::conditional_t<std::is_same_v<T, U>, std::true_type, is_one_of<T, Ts...>> {};

template<typename T, typename... Ts>
constexpr bool is_one_of_v = is_one_of<T, Ts...>::value;

template<typename T>
__aicore__ inline constexpr T ceil_div(T x, T y)
{
    return (x - 1) / y + 1;
}

template<typename T>
__aicore__ inline constexpr T ceil_round(T x, T y)
{
    return ceil_div(x, y) * y;
}

template<typename T>
__aicore__ inline std::enable_if_t<is_one_of_v<T, float, half>> hypot(GM_ADDR input, GM_ADDR other, GM_ADDR out, HypotTilingData &tiling)
{
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

    constexpr int MAX_TILE_SIZE = (31 << 10) / sizeof(T);
    t_pipe.InitBuffer(input_t_que, 2, MAX_TILE_SIZE * sizeof(T));
    t_pipe.InitBuffer(other_t_que, 2, MAX_TILE_SIZE * sizeof(T));
    t_pipe.InitBuffer(out_t_que, 2, MAX_TILE_SIZE * sizeof(T));

    int loop_start, loop_end, loop_step;
    uint16_t atomic_type, atomic_op;
    GetStoreAtomicConfig(atomic_type, atomic_op);
    if (atomic_type)
    {
        loop_start = compute_start + ceil_round(compute_end - compute_start, MAX_TILE_SIZE) - MAX_TILE_SIZE;
        loop_end = compute_start - MAX_TILE_SIZE;
        loop_step = -MAX_TILE_SIZE;
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_NONE, AtomicOp::ATOMIC_SUM>();
    }
    else
    {
        loop_start = compute_start;
        loop_end = compute_start + ceil_round(compute_end - compute_start, MAX_TILE_SIZE);
        loop_step = MAX_TILE_SIZE;
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_S32, AtomicOp::ATOMIC_SUM>();
    }

    for (int i = loop_start, j = 0; i != loop_end; i += loop_step, j++)
    {
        int _ = min(compute_end - i, MAX_TILE_SIZE);
        //
        {
            LocalTensor<T> input = input_t_que.AllocTensor<T>();
            LocalTensor<T> other = other_t_que.AllocTensor<T>();
            DataCopy(input, input_global_tensor[i], _);
            DataCopy(other, other_global_tensor[i], _);
            input_t_que.EnQue(input);
            other_t_que.EnQue(other);
        }
        //
        {
            LocalTensor<T> input = input_t_que.DeQue<T>();
            LocalTensor<T> other = other_t_que.DeQue<T>();
            LocalTensor<T> out = out_t_que.AllocTensor<T>();
            Mul(out, input, input, _);
            MulAddDst(out, other, other, _);
            Sqrt(out, out, _);
            input_t_que.FreeTensor(input);
            other_t_que.FreeTensor(other);
            out_t_que.EnQue(out);
        }
        //
        {
            LocalTensor<T> out = out_t_que.DeQue<T>();
            DataCopy(out_global_tensor[i], out, _);
            out_t_que.FreeTensor(out);
        }
    }
}

template<typename T>
__aicore__ inline std::enable_if_t<std::is_same_v<T, bfloat16_t>> hypot(GM_ADDR input, GM_ADDR other, GM_ADDR out, HypotTilingData &tiling)
{
    using U = float;
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

    constexpr int MAX_TILE_SIZE = (31 << 10) / sizeof(U);
    t_pipe.InitBuffer(input_t_que, 2, MAX_TILE_SIZE * sizeof(U));
    t_pipe.InitBuffer(other_t_que, 2, MAX_TILE_SIZE * sizeof(U));
    t_pipe.InitBuffer(out_t_que, 2, MAX_TILE_SIZE * sizeof(U));

    int loop_start, loop_end, loop_step;
    uint16_t atomic_type, atomic_op;
    GetStoreAtomicConfig(atomic_type, atomic_op);
    if (atomic_type)
    {
        loop_start = compute_start + ceil_round(compute_end - compute_start, MAX_TILE_SIZE) - MAX_TILE_SIZE;
        loop_end = compute_start - MAX_TILE_SIZE;
        loop_step = -MAX_TILE_SIZE;
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_NONE, AtomicOp::ATOMIC_SUM>();
    }
    else
    {
        loop_start = compute_start;
        loop_end = compute_start + ceil_round(compute_end - compute_start, MAX_TILE_SIZE);
        loop_step = MAX_TILE_SIZE;
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_S32, AtomicOp::ATOMIC_SUM>();
    }

    for (int i = loop_start, j = 0; i != loop_end; i += loop_step, j++)
    {
        int _ = min(compute_end - i, MAX_TILE_SIZE);
        //
        {
            LocalTensor<T> input = input_t_que.AllocTensor<T>();
            LocalTensor<T> other = other_t_que.AllocTensor<T>();
            DataCopy(input, input_global_tensor[i], _);
            DataCopy(other, other_global_tensor[i], _);
            input_t_que.EnQue(input);
            other_t_que.EnQue(other);
        }
        //
        {
            LocalTensor<U> input = input_t_que.DeQue<U>();
            LocalTensor<U> other = other_t_que.DeQue<U>();
            LocalTensor<U> out = out_t_que.AllocTensor<U>();
            Cast(out, input.ReinterpretCast<T>(), RoundMode::CAST_NONE, _);
            Mul(out, out, out, _);
            Cast(input, other.ReinterpretCast<T>(), RoundMode::CAST_NONE, _);
            MulAddDst(out, input, input, _);
            Sqrt(out, out, _);
            Cast(out.ReinterpretCast<T>(), out, RoundMode::CAST_RINT, _);
            input_t_que.FreeTensor(input);
            other_t_que.FreeTensor(other);
            out_t_que.EnQue(out);
        }
        //
        {
            LocalTensor<T> out = out_t_que.DeQue<T>();
            DataCopy(out_global_tensor[i], out, _);
            out_t_que.FreeTensor(out);
        }
    }
}

template<typename T>
__aicore__ inline void do_broadcast(GM_ADDR input_other, GM_ADDR out, GM_ADDR workspace, int *input_other_n, int *out_n, TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> &t_que_bind)
{
    int block_index = GetBlockIdx();
    int block_dim = GetBlockNum();
    constexpr int DATA_BLOCK_SIZE = 512 / sizeof(T);
    constexpr int MAX_TILE_SIZE = (95 << 10) / sizeof(T);

    int broadcast_count = 0;
    for (int i = 0; i < 4; i++)
    {
        if (input_other_n[i] != out_n[i])
            broadcast_count++;
    }

    GlobalTensor<T> out_global_tensor, workspace_global_tensor;
    out_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(broadcast_count % 2 == 0 ? out : workspace));
    workspace_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input_other));

    int high_size = input_other_n[0] * input_other_n[1] * input_other_n[2] * input_other_n[3];
    int low_size = 1;
    for (int i = 0; i < 4; i++)
    {
        high_size /= input_other_n[i];
        if (input_other_n[i] != out_n[i])
        {
            int mid_size = out_n[i];
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
                    int _ = min(low_to - k, MAX_TILE_SIZE);
                    //
                    {
                        LocalTensor<T> workspace = t_que_bind.AllocTensor<T>();
                        DataCopyExtParams data_copy_ext_params;
                        data_copy_ext_params.blockLen = _ * sizeof(T);
                        DataCopyPad(workspace, workspace_global_tensor[low_size * j + k], data_copy_ext_params, {});
                        t_que_bind.EnQue(workspace);
                    }
                    //
                    {
                        LocalTensor<T> out = t_que_bind.DeQue<T>();
                        DataCopyExtParams data_copy_ext_params;
                        data_copy_ext_params.blockLen = _ * sizeof(T);
                        for (int l = 0; l < mid_size; l++)
                            DataCopyPad(out_global_tensor[(j * mid_size + l) * low_size + k], out, data_copy_ext_params);
                        t_que_bind.FreeTensor(out);
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
        low_size *= out_n[i];
    }
}

template<typename T>
__aicore__ inline void broadcast(GM_ADDR &input, GM_ADDR &other, GM_ADDR out, GM_ADDR workspace, HypotTilingData &tiling)
{
    TPipe t_pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> t_que_bind;
    constexpr int MAX_TILE_SIZE = (95 << 10) / sizeof(T);
    t_pipe.InitBuffer(t_que_bind, 2, MAX_TILE_SIZE * sizeof(T));

    if (tiling.broadcast_offset[0] >= 0)
    {
        do_broadcast<T>(input, out, workspace + tiling.broadcast_offset[0], tiling.input_n, tiling.out_n, t_que_bind);
        input = workspace + tiling.broadcast_offset[0];
    }
    if (tiling.broadcast_offset[1] >= 0)
    {
        do_broadcast<T>(other, out, workspace + tiling.broadcast_offset[1], tiling.other_n, tiling.out_n, t_que_bind);
        other = workspace + tiling.broadcast_offset[1];
    }

    t_pipe.Destroy();
}

template<typename T>
__aicore__ inline std::enable_if_t<is_one_of_v<T, float, half>> hypot_broadcast(GM_ADDR input, GM_ADDR other, GM_ADDR out, HypotTilingData &tiling)
{
    int block_index = GetBlockIdx();
    int block_dim = GetBlockNum();
    constexpr int DATA_BLOCK_SIZE = 32 / sizeof(T);
    int high_cut, low_cut;
    if (tiling.high_size == 1)
    {
        high_cut = 1;
        low_cut = block_dim;
    }
    else
    {
        high_cut = block_dim;
        low_cut = 1;
    }
    int high_index = block_index / low_cut;
    int low_index = block_index % low_cut;
    long high_start = tiling.high_size * high_index / high_cut;
    long high_end = tiling.high_size * (high_index + 1) / high_cut;
    long low_blocks = tiling.low_size / DATA_BLOCK_SIZE;
    long low_start = low_blocks * low_index / low_cut * DATA_BLOCK_SIZE;
    long low_end = low_blocks * (low_index + 1) / low_cut * DATA_BLOCK_SIZE;

    GlobalTensor<T> input_global_tensor, other_global_tensor, out_global_tensor;
    input_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(tiling.swap_input_other ? other : input));
    other_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(tiling.swap_input_other ? input : other));
    out_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));

    TPipe t_pipe;
    TQue<TPosition::VECIN, 1> input_t_que, other_t_que;
    TQue<TPosition::VECOUT, 1> out_t_que;

    constexpr long MAX_TILE_SIZE = (31 << 10) / sizeof(T);
    long max_mid_tile_size = MAX_TILE_SIZE / min(low_end - low_start, MAX_TILE_SIZE);
    t_pipe.InitBuffer(input_t_que, 2, MAX_TILE_SIZE * sizeof(T));
    t_pipe.InitBuffer(other_t_que, 2, MAX_TILE_SIZE * sizeof(T));
    t_pipe.InitBuffer(out_t_que, 2, MAX_TILE_SIZE * sizeof(T));

    long high_loop_start, high_loop_end, high_loop_step, low_loop_start, low_loop_end, low_loop_step, mid_loop_start, mid_loop_end, mid_loop_step;
    uint16_t atomic_type, atomic_op;
    GetStoreAtomicConfig(atomic_type, atomic_op);
    if (atomic_type)
    {
        high_loop_start = high_end - 1;
        high_loop_end = high_start - 1;
        high_loop_step = -1;
        low_loop_start = low_start + ceil_round(low_end - low_start, MAX_TILE_SIZE) - MAX_TILE_SIZE;
        low_loop_end = low_start - MAX_TILE_SIZE;
        low_loop_step = -MAX_TILE_SIZE;
        mid_loop_start = ceil_round(static_cast<long>(tiling.size), max_mid_tile_size) - max_mid_tile_size;
        mid_loop_end = -max_mid_tile_size;
        mid_loop_step = -max_mid_tile_size;
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_NONE, AtomicOp::ATOMIC_SUM>();
    }
    else
    {
        high_loop_start = high_start;
        high_loop_end = high_end;
        high_loop_step = 1;
        low_loop_start = low_start;
        low_loop_end = low_start + ceil_round(low_end - low_start, MAX_TILE_SIZE);
        low_loop_step = MAX_TILE_SIZE;
        mid_loop_start = 0;
        mid_loop_end = ceil_round(static_cast<long>(tiling.size), max_mid_tile_size);
        mid_loop_step = max_mid_tile_size;
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_S32, AtomicOp::ATOMIC_SUM>();
    }

    for (long i = high_loop_start; i != high_loop_end; i += high_loop_step)
    {
        for (long j = low_loop_start; j != low_loop_end; j += low_loop_step)
        {
            int low_tile_size = min(low_end - j, MAX_TILE_SIZE);
            //
            {
                LocalTensor<T> input = input_t_que.AllocTensor<T>();
                DataCopy(input, input_global_tensor[i * tiling.low_size + j], low_tile_size);
                input_t_que.EnQue(input);
            }
            LocalTensor<T> input;
            for (long k = mid_loop_start; k != mid_loop_end; k += mid_loop_step)
            {
                int mid_tile_size = min(tiling.size - k, max_mid_tile_size);
                //
                {
                    LocalTensor<T> other = other_t_que.AllocTensor<T>();
                    DataCopyParams data_copy_params(mid_tile_size, low_tile_size * sizeof(T) / 32, (tiling.low_size - low_tile_size) * sizeof(T) / 32, 0);
                    DataCopy(other, other_global_tensor[i * tiling.size * tiling.low_size + j + k * tiling.low_size], data_copy_params);
                    other_t_que.EnQue(other);
                }
                //
                {
                    int _ = low_tile_size * mid_tile_size;
                    if (k == mid_loop_start)
                    {
                        input = input_t_que.DeQue<T>();
                        Mul(input, input, input, _);
                        if (mid_tile_size > 1)
                        {
                            CopyRepeatParams copy_repeat_params{1, 1, 8, 8};
                            int current_size;
                            for (current_size = low_tile_size; current_size < min(_, static_cast<int>(2560 / sizeof(T))); current_size *= 2)
                                Copy(input[current_size], input, 256 / sizeof(T), ceil_div(static_cast<int>(current_size * sizeof(T)), 256), copy_repeat_params);
                            if (current_size < _)
                                Copy(input[current_size], input, 256 / sizeof(T), ceil_div(static_cast<int>((_ - current_size) * sizeof(T)), 256), copy_repeat_params);
                        }
                    }
                    LocalTensor<T> other = other_t_que.DeQue<T>();
                    LocalTensor<T> out = out_t_que.AllocTensor<T>();
                    FusedMulAdd(other, other, input, _);
                    Sqrt(out, other, _);
                    other_t_que.FreeTensor(other);
                    out_t_que.EnQue(out);
                }
                //
                {
                    LocalTensor<T> out = out_t_que.DeQue<T>();
                    DataCopyParams data_copy_params(mid_tile_size, low_tile_size * sizeof(T) / 32, 0, (tiling.low_size - low_tile_size) * sizeof(T) / 32);
                    DataCopy(out_global_tensor[i * tiling.size * tiling.low_size + j + k * tiling.low_size], out, data_copy_params);
                    out_t_que.FreeTensor(out);
                }
            }
            input_t_que.FreeTensor(input);
        }
    }
}

template<typename T>
__aicore__ inline std::enable_if_t<std::is_same_v<T, bfloat16_t>> hypot_broadcast(GM_ADDR input, GM_ADDR other, GM_ADDR out, HypotTilingData &tiling)
{
    using U = float;
    int block_index = GetBlockIdx();
    int block_dim = GetBlockNum();
    constexpr int DATA_BLOCK_SIZE = 32 / sizeof(T);
    int high_cut, low_cut;
    if (tiling.high_size == 1)
    {
        high_cut = 1;
        low_cut = block_dim;
    }
    else
    {
        high_cut = block_dim;
        low_cut = 1;
    }
    int high_index = block_index / low_cut;
    int low_index = block_index % low_cut;
    long high_start = tiling.high_size * high_index / high_cut;
    long high_end = tiling.high_size * (high_index + 1) / high_cut;
    long low_blocks = tiling.low_size / DATA_BLOCK_SIZE;
    long low_start = low_blocks * low_index / low_cut * DATA_BLOCK_SIZE;
    long low_end = low_blocks * (low_index + 1) / low_cut * DATA_BLOCK_SIZE;

    GlobalTensor<T> input_global_tensor, other_global_tensor, out_global_tensor;
    input_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(tiling.swap_input_other ? other : input));
    other_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(tiling.swap_input_other ? input : other));
    out_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));

    TPipe t_pipe;
    TQue<TPosition::VECIN, 1> input_t_que, other_t_que;
    TQue<TPosition::VECOUT, 1> out_t_que;

    constexpr long MAX_TILE_SIZE = (31 << 10) / sizeof(U);
    long max_mid_tile_size = MAX_TILE_SIZE / min(low_end - low_start, MAX_TILE_SIZE);
    t_pipe.InitBuffer(input_t_que, 2, MAX_TILE_SIZE * sizeof(U));
    t_pipe.InitBuffer(other_t_que, 2, MAX_TILE_SIZE * sizeof(U));
    t_pipe.InitBuffer(out_t_que, 2, MAX_TILE_SIZE * sizeof(U));

    long high_loop_start, high_loop_end, high_loop_step, low_loop_start, low_loop_end, low_loop_step, mid_loop_start, mid_loop_end, mid_loop_step;
    uint16_t atomic_type, atomic_op;
    GetStoreAtomicConfig(atomic_type, atomic_op);
    if (atomic_type)
    {
        high_loop_start = high_end - 1;
        high_loop_end = high_start - 1;
        high_loop_step = -1;
        low_loop_start = low_start + ceil_round(low_end - low_start, MAX_TILE_SIZE) - MAX_TILE_SIZE;
        low_loop_end = low_start - MAX_TILE_SIZE;
        low_loop_step = -MAX_TILE_SIZE;
        mid_loop_start = ceil_round(static_cast<long>(tiling.size), max_mid_tile_size) - max_mid_tile_size;
        mid_loop_end = -max_mid_tile_size;
        mid_loop_step = -max_mid_tile_size;
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_NONE, AtomicOp::ATOMIC_SUM>();
    }
    else
    {
        high_loop_start = high_start;
        high_loop_end = high_end;
        high_loop_step = 1;
        low_loop_start = low_start;
        low_loop_end = low_start + ceil_round(low_end - low_start, MAX_TILE_SIZE);
        low_loop_step = MAX_TILE_SIZE;
        mid_loop_start = 0;
        mid_loop_end = ceil_round(static_cast<long>(tiling.size), max_mid_tile_size);
        mid_loop_step = max_mid_tile_size;
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_S32, AtomicOp::ATOMIC_SUM>();
    }

    for (long i = high_loop_start; i != high_loop_end; i += high_loop_step)
    {
        for (long j = low_loop_start; j != low_loop_end; j += low_loop_step)
        {
            int low_tile_size = min(low_end - j, MAX_TILE_SIZE);
            //
            {
                LocalTensor<T> input = input_t_que.AllocTensor<T>();
                DataCopy(input, input_global_tensor[i * tiling.low_size + j], low_tile_size);
                input_t_que.EnQue(input);
            }
            LocalTensor<U> input;
            for (long k = mid_loop_start; k != mid_loop_end; k += mid_loop_step)
            {
                int mid_tile_size = min(tiling.size - k, max_mid_tile_size);
                //
                {
                    LocalTensor<T> other = other_t_que.AllocTensor<T>();
                    DataCopyParams data_copy_params(mid_tile_size, low_tile_size * sizeof(T) / 32, (tiling.low_size - low_tile_size) * sizeof(T) / 32, 0);
                    DataCopy(other, other_global_tensor[i * tiling.size * tiling.low_size + j + k * tiling.low_size], data_copy_params);
                    other_t_que.EnQue(other);
                }
                //
                {
                    int _ = low_tile_size * mid_tile_size;
                    LocalTensor<U> out = out_t_que.AllocTensor<U>();
                    if (k == mid_loop_start)
                    {
                        input = input_t_que.DeQue<U>();
                        Cast(out, input.ReinterpretCast<T>(), RoundMode::CAST_NONE, _);
                        Mul(input, out, out, _);
                        if (mid_tile_size > 1)
                        {
                            CopyRepeatParams copy_repeat_params{1, 1, 8, 8};
                            int current_size;
                            for (current_size = low_tile_size; current_size < min(_, static_cast<int>(2560 / sizeof(U))); current_size *= 2)
                                Copy(input[current_size], input, 256 / sizeof(U), ceil_div(static_cast<int>(current_size * sizeof(U)), 256), copy_repeat_params);
                            if (current_size < _)
                                Copy(input[current_size], input, 256 / sizeof(U), ceil_div(static_cast<int>((_ - current_size) * sizeof(U)), 256), copy_repeat_params);
                        }
                    }
                    LocalTensor<U> other = other_t_que.DeQue<U>();
                    Cast(out, other.ReinterpretCast<T>(), RoundMode::CAST_NONE, _);
                    FusedMulAdd(out, out, input, _);
                    Sqrt(out, out, _);
                    Cast(out.ReinterpretCast<T>(), out, RoundMode::CAST_RINT, _);
                    other_t_que.FreeTensor(other);
                    out_t_que.EnQue(out);
                }
                //
                {
                    LocalTensor<T> out = out_t_que.DeQue<T>();
                    DataCopyParams data_copy_params(mid_tile_size, low_tile_size * sizeof(T) / 32, 0, (tiling.low_size - low_tile_size) * sizeof(T) / 32);
                    DataCopy(out_global_tensor[i * tiling.size * tiling.low_size + j + k * tiling.low_size], out, data_copy_params);
                    out_t_que.FreeTensor(out);
                }
            }
            input_t_que.FreeTensor(input);
        }
    }
}

extern "C" __global__ __aicore__ void hypot(GM_ADDR input, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(0))
        hypot<float>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(1))
        hypot<half>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(27))
        hypot<bfloat16_t>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(100))
    {
        KERNEL_TASK_TYPE(100, KERNEL_TYPE_MIX_AIV_1_0);
        broadcast<float>(input, other, out, workspace, tiling_data);
        hypot<float>(input, other, out, tiling_data);
    }
    else if (TILING_KEY_IS(101))
    {
        KERNEL_TASK_TYPE(101, KERNEL_TYPE_MIX_AIV_1_0);
        broadcast<half>(input, other, out, workspace, tiling_data);
        hypot<half>(input, other, out, tiling_data);
    }
    else if (TILING_KEY_IS(127))
    {
        KERNEL_TASK_TYPE(127, KERNEL_TYPE_MIX_AIV_1_0);
        broadcast<bfloat16_t>(input, other, out, workspace, tiling_data);
        hypot<bfloat16_t>(input, other, out, tiling_data);
    }
    else if (TILING_KEY_IS(200))
        hypot_broadcast<float>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(201))
        hypot_broadcast<half>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(227))
        hypot_broadcast<bfloat16_t>(input, other, out, tiling_data);
}
