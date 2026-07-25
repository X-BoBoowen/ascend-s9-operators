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
__aicore__ inline std::enable_if_t<is_one_of_v<T, float, half>> fmin(GM_ADDR input, GM_ADDR other, GM_ADDR out, FminTilingData &tiling)
{
    int block_index = GetBlockIdx();
    int block_dim = GetBlockNum();
    constexpr int DATA_BLOCK_SIZE = 512 / sizeof(T);
    long compute_blocks = ceil_div(tiling.size, DATA_BLOCK_SIZE);
    long compute_start = compute_blocks * block_index / block_dim * DATA_BLOCK_SIZE;
    long compute_end = compute_blocks * (block_index + 1) / block_dim * DATA_BLOCK_SIZE;

    GlobalTensor<T> input_global_tensor, other_global_tensor, out_global_tensor;
    input_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
    other_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(other));
    out_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));

    TPipe t_pipe;
    TQue<TPosition::VECIN, 1> input_t_que, other_t_que;
    TQue<TPosition::VECOUT, 1> out_t_que;

    constexpr long MAX_TILE_SIZE = (30 << 10) / sizeof(T);
    t_pipe.InitBuffer(input_t_que, 2, MAX_TILE_SIZE * sizeof(T));
    t_pipe.InitBuffer(other_t_que, 2, MAX_TILE_SIZE * sizeof(T));
    t_pipe.InitBuffer(out_t_que, 2, MAX_TILE_SIZE * sizeof(T));

    long loop_start, loop_end, loop_step;
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

    for (long i = loop_start; i != loop_end; i += loop_step)
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
            LocalTensor<uint8_t> out_uint8_0 = out.template ReinterpretCast<uint8_t>();
            LocalTensor<uint8_t> out_uint8_1 = out[MAX_TILE_SIZE / 2].template ReinterpretCast<uint8_t>();
            LocalTensor<uint16_t> out_uint16_0 = out_uint8_0.ReinterpretCast<uint16_t>();
            LocalTensor<uint16_t> out_uint16_1 = out_uint8_1.ReinterpretCast<uint16_t>();
            int compare_size = ceil_round(_, static_cast<int>(256 / sizeof(T)));
            Compare(out_uint8_0, input, other, CMPMODE::LT, compare_size);
            Compare(out_uint8_1, other, other, CMPMODE::EQ, compare_size);
            int logical_size = compare_size / 16;
            Not(out_uint16_1, out_uint16_1, logical_size);
            Or(out_uint16_0, out_uint16_0, out_uint16_1, logical_size);
            Select(input, out_uint8_0, input, other, SELMODE::VSEL_TENSOR_TENSOR_MODE, _);
            Min(out, input, input, _);
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
__aicore__ inline std::enable_if_t<is_one_of_v<T, int8_t, uint8_t>> fmin(GM_ADDR input, GM_ADDR other, GM_ADDR out, FminTilingData &tiling)
{
    using U = half;
    int block_index = GetBlockIdx();
    int block_dim = GetBlockNum();
    constexpr int DATA_BLOCK_SIZE = 512 / sizeof(T);
    long compute_blocks = ceil_div(tiling.size, DATA_BLOCK_SIZE);
    long compute_start = compute_blocks * block_index / block_dim * DATA_BLOCK_SIZE;
    long compute_end = compute_blocks * (block_index + 1) / block_dim * DATA_BLOCK_SIZE;

    GlobalTensor<T> input_global_tensor, other_global_tensor, out_global_tensor;
    input_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
    other_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(other));
    out_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));

    TPipe t_pipe;
    TQue<TPosition::VECIN, 1> input_t_que, other_t_que;
    TQue<TPosition::VECOUT, 1> out_t_que;

    constexpr long MAX_TILE_SIZE = (31 << 10) / sizeof(U);
    t_pipe.InitBuffer(input_t_que, 2, MAX_TILE_SIZE * sizeof(U));
    t_pipe.InitBuffer(other_t_que, 2, MAX_TILE_SIZE * sizeof(U));
    t_pipe.InitBuffer(out_t_que, 2, MAX_TILE_SIZE * sizeof(U));

    long loop_start, loop_end, loop_step;
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
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_F32, AtomicOp::ATOMIC_SUM>();
    }

    for (long i = loop_start; i != loop_end; i += loop_step)
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
            Cast(input, other.ReinterpretCast<T>(), RoundMode::CAST_NONE, _);
            Min(out, out, input, _);
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
__aicore__ inline std::enable_if_t<is_one_of_v<T, int, short>> fmin(GM_ADDR input, GM_ADDR other, GM_ADDR out, FminTilingData &tiling)
{
    int block_index = GetBlockIdx();
    int block_dim = GetBlockNum();
    constexpr int DATA_BLOCK_SIZE = 512 / sizeof(T);
    long compute_blocks = ceil_div(tiling.size, DATA_BLOCK_SIZE);
    long compute_start = compute_blocks * block_index / block_dim * DATA_BLOCK_SIZE;
    long compute_end = compute_blocks * (block_index + 1) / block_dim * DATA_BLOCK_SIZE;

    GlobalTensor<T> input_global_tensor, other_global_tensor, out_global_tensor;
    input_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
    other_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(other));
    out_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));

    TPipe t_pipe;
    TQue<TPosition::VECIN, 1> input_t_que, other_t_que;
    TQue<TPosition::VECOUT, 1> out_t_que;

    constexpr long MAX_TILE_SIZE = (31 << 10) / sizeof(T);
    t_pipe.InitBuffer(input_t_que, 2, MAX_TILE_SIZE * sizeof(T));
    t_pipe.InitBuffer(other_t_que, 2, MAX_TILE_SIZE * sizeof(T));
    t_pipe.InitBuffer(out_t_que, 2, MAX_TILE_SIZE * sizeof(T));

    long loop_start, loop_end, loop_step;
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
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_F32, AtomicOp::ATOMIC_SUM>();
    }

    for (long i = loop_start; i != loop_end; i += loop_step)
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
            Min(out, input, other, _);
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
__aicore__ inline std::enable_if_t<std::is_same_v<T, long>> fmin(GM_ADDR input, GM_ADDR other, GM_ADDR out, FminTilingData &tiling)
{
    using U = int;
    using V = float;
    int block_index = GetBlockIdx();
    int block_dim = GetBlockNum();
    constexpr int DATA_BLOCK_SIZE = 512 / sizeof(T);
    long compute_blocks = ceil_div(tiling.size, DATA_BLOCK_SIZE);
    long compute_start = compute_blocks * block_index / block_dim * DATA_BLOCK_SIZE;
    long compute_end = compute_blocks * (block_index + 1) / block_dim * DATA_BLOCK_SIZE;

    GlobalTensor<T> input_global_tensor, other_global_tensor, out_global_tensor;
    input_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
    other_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(other));
    out_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));

    TPipe t_pipe;
    TQue<TPosition::VECIN, 1> input_t_que, other_t_que;
    TQue<TPosition::VECOUT, 1> out_t_que;
    TBuf<> t_bufs[2];

    constexpr long MAX_TILE_SIZE = (22 << 10) / sizeof(T);
    t_pipe.InitBuffer(input_t_que, 2, MAX_TILE_SIZE * sizeof(T));
    t_pipe.InitBuffer(other_t_que, 2, MAX_TILE_SIZE * sizeof(T));
    t_pipe.InitBuffer(out_t_que, 2, MAX_TILE_SIZE * sizeof(T));
    for (auto &t_buf : t_bufs)
        t_pipe.InitBuffer(t_buf, MAX_TILE_SIZE * sizeof(T));

    long loop_start, loop_end, loop_step;
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
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_F32, AtomicOp::ATOMIC_SUM>();
    }

    for (long i = loop_start; i != loop_end; i += loop_step)
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
            LocalTensor<uint64_t> input_uint64 = input.ReinterpretCast<uint64_t>();
            LocalTensor<U> other = other_t_que.DeQue<U>();
            LocalTensor<U> out = out_t_que.AllocTensor<U>();
            LocalTensor<uint8_t> out_uint8_0 = out.ReinterpretCast<uint8_t>();
            LocalTensor<uint8_t> out_uint8_1 = out[MAX_TILE_SIZE].ReinterpretCast<uint8_t>();
            LocalTensor<uint16_t> out_uint16_0 = out_uint8_0.ReinterpretCast<uint16_t>();
            LocalTensor<uint16_t> out_uint16_1 = out_uint8_1.ReinterpretCast<uint16_t>();
            LocalTensor<unsigned> out_unsigned = out.ReinterpretCast<unsigned>();
            LocalTensor<U> temp_0_0 = t_bufs[0].Get<U>();
            LocalTensor<V> temp_V_0_0 = temp_0_0.ReinterpretCast<V>();
            LocalTensor<U> temp_0_1 = temp_0_0[MAX_TILE_SIZE];
            LocalTensor<V> temp_V_0_1 = temp_0_1.ReinterpretCast<V>();
            LocalTensor<U> temp_1_0 = t_bufs[1].Get<U>();
            LocalTensor<V> temp_V_1_0 = temp_1_0.ReinterpretCast<V>();
            LocalTensor<U> temp_1_1 = temp_1_0[MAX_TILE_SIZE];
            LocalTensor<V> temp_V_1_1 = temp_1_1.ReinterpretCast<V>();
            GatherMaskParams gather_mask_params;
            gather_mask_params.repeatTimes = ceil_div(static_cast<int>(_ * sizeof(T)), 256);
            uint64_t rsvd;
            GatherMask(temp_0_0, input, 1, false, 0, gather_mask_params, rsvd);
            GatherMask(temp_0_1, input, 2, false, 0, gather_mask_params, rsvd);
            GatherMask(temp_1_0, other, 1, false, 0, gather_mask_params, rsvd);
            GatherMask(temp_1_1, other, 2, false, 0, gather_mask_params, rsvd);
            Min(input, temp_0_0, temp_1_0, _);
            Min(other, temp_0_1, temp_1_1, _);
            int compare_size = ceil_round(_, static_cast<int>(256 / sizeof(U)));
            Compare(out_uint8_0, temp_0_0, input, CMPMODE::EQ, compare_size);
            Compare(out_uint8_1, temp_1_1, other, CMPMODE::EQ, compare_size);
            int logical_size = compare_size / 16;
            Not(out_uint16_1, out_uint16_1, logical_size);
            Or(out_uint16_0, out_uint16_0, out_uint16_1, logical_size);
            Compare(out_uint8_1, temp_0_1, other, CMPMODE::EQ, compare_size);
            And(out_uint16_0, out_uint16_0, out_uint16_1, logical_size);
            Select(input.ReinterpretCast<V>(), out_uint8_0, temp_V_0_0, temp_V_1_0, SELMODE::VSEL_TENSOR_TENSOR_MODE, _);
            Select(other.ReinterpretCast<V>(), out_uint8_0, temp_V_0_1, temp_V_1_1, SELMODE::VSEL_TENSOR_TENSOR_MODE, _);
            CreateVecIndex(out, 0, _ * 2);
            ShiftRight(out, out, 1, _ * 2);
            Muls(out, out, static_cast<U>(sizeof(U)), _ * 2);
            Gather(temp_0_0, input, out_unsigned, 0, _ * 2);
            Gather(temp_1_0, other, out_unsigned, 0, _ * 2);
            input_uint64.SetValue(0, 0x5555555555555555);
            Select(out.ReinterpretCast<V>(), input_uint64, temp_V_0_0, temp_V_1_0, SELMODE::VSEL_CMPMASK_SPR, _ * 2);
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
__aicore__ inline std::enable_if_t<std::is_same_v<T, bfloat16_t>> fmin(GM_ADDR input, GM_ADDR other, GM_ADDR out, FminTilingData &tiling)
{
    using U = float;
    int block_index = GetBlockIdx();
    int block_dim = GetBlockNum();
    constexpr int DATA_BLOCK_SIZE = 512 / sizeof(T);
    long compute_blocks = ceil_div(tiling.size, DATA_BLOCK_SIZE);
    long compute_start = compute_blocks * block_index / block_dim * DATA_BLOCK_SIZE;
    long compute_end = compute_blocks * (block_index + 1) / block_dim * DATA_BLOCK_SIZE;

    GlobalTensor<T> input_global_tensor, other_global_tensor, out_global_tensor;
    input_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
    other_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(other));
    out_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));

    TPipe t_pipe;
    TQue<TPosition::VECIN, 1> input_t_que, other_t_que;
    TQue<TPosition::VECOUT, 1> out_t_que;

    constexpr long MAX_TILE_SIZE = (31 << 10) / sizeof(U);
    t_pipe.InitBuffer(input_t_que, 2, MAX_TILE_SIZE * sizeof(U));
    t_pipe.InitBuffer(other_t_que, 2, MAX_TILE_SIZE * sizeof(U));
    t_pipe.InitBuffer(out_t_que, 2, MAX_TILE_SIZE * sizeof(U));

    long loop_start, loop_end, loop_step;
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

    for (long i = loop_start; i != loop_end; i += loop_step)
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
            LocalTensor<uint8_t> other_uint8_0 = other.ReinterpretCast<uint8_t>();
            LocalTensor<uint8_t> other_uint8_1 = other[MAX_TILE_SIZE / 2].ReinterpretCast<uint8_t>();
            LocalTensor<uint16_t> other_uint16_0 = other_uint8_0.ReinterpretCast<uint16_t>();
            LocalTensor<uint16_t> other_uint16_1 = other_uint8_1.ReinterpretCast<uint16_t>();
            LocalTensor<U> out = out_t_que.AllocTensor<U>();
            Cast(out, input.ReinterpretCast<T>(), RoundMode::CAST_NONE, _);
            Cast(input, other.ReinterpretCast<T>(), RoundMode::CAST_NONE, _);
            int compare_size = ceil_round(_, static_cast<int>(256 / sizeof(U)));
            Compare(other_uint8_0, out, input, CMPMODE::LT, compare_size);
            Compare(other_uint8_1, input, input, CMPMODE::EQ, compare_size);
            int logical_size = compare_size / 16;
            Not(other_uint16_1, other_uint16_1, logical_size);
            Or(other_uint16_0, other_uint16_0, other_uint16_1, logical_size);
            Select(out, other_uint8_0, out, input, SELMODE::VSEL_TENSOR_TENSOR_MODE, _);
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
__aicore__ inline void broadcast(GM_ADDR &input, GM_ADDR &other, GM_ADDR out, GM_ADDR workspace, FminTilingData &tiling)
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
__aicore__ inline std::enable_if_t<is_one_of_v<T, float, half>> fmin_broadcast(GM_ADDR input, GM_ADDR other, GM_ADDR out, FminTilingData &tiling)
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

    constexpr long MAX_TILE_SIZE = (30 << 10) / sizeof(T);
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
                    LocalTensor<uint8_t> out_uint8_0 = out.template ReinterpretCast<uint8_t>();
                    LocalTensor<uint8_t> out_uint8_1 = out[MAX_TILE_SIZE / 2].template ReinterpretCast<uint8_t>();
                    LocalTensor<uint16_t> out_uint16_0 = out_uint8_0.ReinterpretCast<uint16_t>();
                    LocalTensor<uint16_t> out_uint16_1 = out_uint8_1.ReinterpretCast<uint16_t>();
                    int compare_size = ceil_round(_, static_cast<int>(256 / sizeof(T)));
                    Compare(out_uint8_0, input, other, CMPMODE::LT, compare_size);
                    Compare(out_uint8_1, other, other, CMPMODE::EQ, compare_size);
                    int logical_size = compare_size / 16;
                    Not(out_uint16_1, out_uint16_1, logical_size);
                    Or(out_uint16_0, out_uint16_0, out_uint16_1, logical_size);
                    Select(other, out_uint8_0, input, other, SELMODE::VSEL_TENSOR_TENSOR_MODE, _);
                    Min(out, other, other, _);
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
__aicore__ inline std::enable_if_t<is_one_of_v<T, int8_t, uint8_t>> fmin_broadcast(GM_ADDR input, GM_ADDR other, GM_ADDR out, FminTilingData &tiling)
{
    using U = half;
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
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_F32, AtomicOp::ATOMIC_SUM>();
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
                        Adds(input, out, static_cast<U>(0), _);
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
                    Min(out, out, input, _);
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

template<typename T>
__aicore__ inline std::enable_if_t<is_one_of_v<T, int, short>> fmin_broadcast(GM_ADDR input, GM_ADDR other, GM_ADDR out, FminTilingData &tiling)
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
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_F32, AtomicOp::ATOMIC_SUM>();
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
                    Min(out, input, other, _);
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
__aicore__ inline std::enable_if_t<std::is_same_v<T, long>> fmin_broadcast(GM_ADDR input, GM_ADDR other, GM_ADDR out, FminTilingData &tiling)
{
    using U = int;
    using V = float;
    int block_index = GetBlockIdx();
    int block_dim = GetBlockNum();
    constexpr int DATA_BLOCK_SIZE = 32 / sizeof(U);
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
    TBuf<> t_bufs[2];

    constexpr long MAX_TILE_SIZE = (22 << 10) / sizeof(T);
    long max_mid_tile_size = MAX_TILE_SIZE / min(low_end - low_start, MAX_TILE_SIZE);
    t_pipe.InitBuffer(input_t_que, 2, MAX_TILE_SIZE * sizeof(T));
    t_pipe.InitBuffer(other_t_que, 2, MAX_TILE_SIZE * sizeof(T));
    t_pipe.InitBuffer(out_t_que, 2, MAX_TILE_SIZE * sizeof(T));
    for (auto &t_buf : t_bufs)
        t_pipe.InitBuffer(t_buf, MAX_TILE_SIZE * sizeof(T));

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
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_F32, AtomicOp::ATOMIC_SUM>();
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
            LocalTensor<U> input_0, input_1;
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
                    LocalTensor<U> temp_0_0 = t_bufs[0].Get<U>();
                    LocalTensor<U> temp_0_1 = temp_0_0[MAX_TILE_SIZE];
                    GatherMaskParams gather_mask_params;
                    gather_mask_params.repeatTimes = ceil_div(static_cast<int>(_ * sizeof(T)), 256);
                    uint64_t rsvd;
                    if (k == mid_loop_start)
                    {
                        input_0 = input_t_que.DeQue<U>();
                        input_1 = input_0[MAX_TILE_SIZE];
                        GatherMask(temp_0_0, input_0, 1, false, 0, gather_mask_params, rsvd);
                        GatherMask(temp_0_1, input_0, 2, false, 0, gather_mask_params, rsvd);
                        Adds(input_0, temp_0_0, static_cast<U>(0), _);
                        Adds(input_1, temp_0_1, static_cast<U>(0), _);
                        if (mid_tile_size > 1)
                        {
                            CopyRepeatParams copy_repeat_params{1, 1, 8, 8};
                            int current_size;
                            for (current_size = low_tile_size; current_size < min(_, static_cast<int>(2560 / sizeof(U))); current_size *= 2)
                            {
                                Copy(input_0[current_size], input_0, 256 / sizeof(U), ceil_div(static_cast<int>(current_size * sizeof(U)), 256), copy_repeat_params);
                                Copy(input_1[current_size], input_1, 256 / sizeof(U), ceil_div(static_cast<int>(current_size * sizeof(U)), 256), copy_repeat_params);
                            }
                            if (current_size < _)
                            {
                                Copy(input_0[current_size], input_0, 256 / sizeof(U), ceil_div(static_cast<int>((_ - current_size) * sizeof(U)), 256), copy_repeat_params);
                                Copy(input_1[current_size], input_1, 256 / sizeof(U), ceil_div(static_cast<int>((_ - current_size) * sizeof(U)), 256), copy_repeat_params);
                            }
                        }
                    }
                    LocalTensor<U> other_0 = other_t_que.DeQue<U>();
                    LocalTensor<U> other_1 = other_0[MAX_TILE_SIZE];
                    LocalTensor<uint64_t> other_uint64 = other_0.ReinterpretCast<uint64_t>();
                    LocalTensor<U> out = out_t_que.AllocTensor<U>();
                    LocalTensor<uint8_t> out_uint8_0 = out.ReinterpretCast<uint8_t>();
                    LocalTensor<uint8_t> out_uint8_1 = out[MAX_TILE_SIZE].ReinterpretCast<uint8_t>();
                    LocalTensor<uint16_t> out_uint16_0 = out_uint8_0.ReinterpretCast<uint16_t>();
                    LocalTensor<uint16_t> out_uint16_1 = out_uint8_1.ReinterpretCast<uint16_t>();
                    LocalTensor<unsigned> out_unsigned = out.ReinterpretCast<unsigned>();
                    LocalTensor<U> temp_1_0 = t_bufs[1].Get<U>();
                    LocalTensor<V> temp_V_1_0 = temp_1_0.ReinterpretCast<V>();
                    LocalTensor<U> temp_1_1 = temp_1_0[MAX_TILE_SIZE];
                    GatherMask(temp_1_0, other_0, 1, false, 0, gather_mask_params, rsvd);
                    GatherMask(temp_1_1, other_0, 2, false, 0, gather_mask_params, rsvd);
                    Min(other_0, input_0, temp_1_0, _);
                    Min(other_1, input_1, temp_1_1, _);
                    int compare_size = ceil_round(_, static_cast<int>(256 / sizeof(U)));
                    Compare(out_uint8_0, input_0, other_0, CMPMODE::EQ, compare_size);
                    Compare(out_uint8_1, temp_1_1, other_1, CMPMODE::EQ, compare_size);
                    int logical_size = compare_size / 16;
                    Not(out_uint16_1, out_uint16_1, logical_size);
                    Or(out_uint16_0, out_uint16_0, out_uint16_1, logical_size);
                    Compare(out_uint8_1, input_1, other_1, CMPMODE::EQ, compare_size);
                    And(out_uint16_0, out_uint16_0, out_uint16_1, logical_size);
                    Select(other_0.ReinterpretCast<V>(), out_uint8_0, input_0.ReinterpretCast<V>(), temp_V_1_0, SELMODE::VSEL_TENSOR_TENSOR_MODE, _);
                    Select(other_1.ReinterpretCast<V>(), out_uint8_0, input_1.ReinterpretCast<V>(), temp_1_1.ReinterpretCast<V>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, _);
                    CreateVecIndex(out, 0, _ * 2);
                    ShiftRight(out, out, 1, _ * 2);
                    Muls(out, out, static_cast<U>(sizeof(U)), _ * 2);
                    Gather(temp_0_0, other_0, out_unsigned, 0, _ * 2);
                    Gather(temp_1_0, other_1, out_unsigned, 0, _ * 2);
                    other_uint64.SetValue(0, 0x5555555555555555);
                    Select(out.ReinterpretCast<V>(), other_uint64, temp_0_0.ReinterpretCast<V>(), temp_V_1_0, SELMODE::VSEL_CMPMASK_SPR, _ * 2);
                    other_t_que.FreeTensor(other_0);
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
            input_t_que.FreeTensor(input_0);
        }
    }
}

template<typename T>
__aicore__ inline std::enable_if_t<std::is_same_v<T, bfloat16_t>> fmin_broadcast(GM_ADDR input, GM_ADDR other, GM_ADDR out, FminTilingData &tiling)
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

    constexpr long MAX_TILE_SIZE = (30 << 10) / sizeof(U);
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
                        Adds(input, out, static_cast<U>(0), _);
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
                    LocalTensor<uint8_t> other_uint8_0 = other.ReinterpretCast<uint8_t>();
                    LocalTensor<uint8_t> other_uint8_1 = other[MAX_TILE_SIZE / 2].ReinterpretCast<uint8_t>();
                    LocalTensor<uint16_t> other_uint16_0 = other_uint8_0.ReinterpretCast<uint16_t>();
                    LocalTensor<uint16_t> other_uint16_1 = other_uint8_1.ReinterpretCast<uint16_t>();
                    Cast(out, other.ReinterpretCast<T>(), RoundMode::CAST_NONE, _);
                    int compare_size = ceil_round(_, static_cast<int>(256 / sizeof(U)));
                    Compare(other_uint8_0, out, input, CMPMODE::LT, compare_size);
                    Compare(other_uint8_1, out, out, CMPMODE::EQ, compare_size);
                    int logical_size = compare_size / 16;
                    Not(other_uint16_1, other_uint16_1, logical_size);
                    Or(other_uint16_0, other_uint16_0, other_uint16_1, logical_size);
                    Select(out, other_uint8_0, out, input, SELMODE::VSEL_TENSOR_TENSOR_MODE, _);
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

extern "C" __global__ __aicore__ void fmin(GM_ADDR input, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(0))
        fmin<float>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(1))
        fmin<half>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(2))
        fmin<int8_t>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(3))
        fmin<int>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(4))
        fmin<uint8_t>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(6))
        fmin<short>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(9))
        fmin<long>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(27))
        fmin<bfloat16_t>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(100))
    {
        KERNEL_TASK_TYPE(100, KERNEL_TYPE_MIX_AIV_1_0);
        broadcast<float>(input, other, out, workspace, tiling_data);
        fmin<float>(input, other, out, tiling_data);
    }
    else if (TILING_KEY_IS(101))
    {
        KERNEL_TASK_TYPE(101, KERNEL_TYPE_MIX_AIV_1_0);
        broadcast<half>(input, other, out, workspace, tiling_data);
        fmin<half>(input, other, out, tiling_data);
    }
    else if (TILING_KEY_IS(102))
    {
        KERNEL_TASK_TYPE(102, KERNEL_TYPE_MIX_AIV_1_0);
        broadcast<int8_t>(input, other, out, workspace, tiling_data);
        fmin<int8_t>(input, other, out, tiling_data);
    }
    else if (TILING_KEY_IS(103))
    {
        KERNEL_TASK_TYPE(103, KERNEL_TYPE_MIX_AIV_1_0);
        broadcast<int>(input, other, out, workspace, tiling_data);
        fmin<int>(input, other, out, tiling_data);
    }
    else if (TILING_KEY_IS(104))
    {
        KERNEL_TASK_TYPE(104, KERNEL_TYPE_MIX_AIV_1_0);
        broadcast<uint8_t>(input, other, out, workspace, tiling_data);
        fmin<uint8_t>(input, other, out, tiling_data);
    }
    else if (TILING_KEY_IS(106))
    {
        KERNEL_TASK_TYPE(106, KERNEL_TYPE_MIX_AIV_1_0);
        broadcast<short>(input, other, out, workspace, tiling_data);
        fmin<short>(input, other, out, tiling_data);
    }
    else if (TILING_KEY_IS(109))
    {
        KERNEL_TASK_TYPE(109, KERNEL_TYPE_MIX_AIV_1_0);
        broadcast<long>(input, other, out, workspace, tiling_data);
        fmin<long>(input, other, out, tiling_data);
    }
    else if (TILING_KEY_IS(127))
    {
        KERNEL_TASK_TYPE(127, KERNEL_TYPE_MIX_AIV_1_0);
        broadcast<bfloat16_t>(input, other, out, workspace, tiling_data);
        fmin<bfloat16_t>(input, other, out, tiling_data);
    }
    else if (TILING_KEY_IS(200))
        fmin_broadcast<float>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(201))
        fmin_broadcast<half>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(202))
        fmin_broadcast<int8_t>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(203))
        fmin_broadcast<int>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(204))
        fmin_broadcast<uint8_t>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(206))
        fmin_broadcast<short>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(209))
        fmin_broadcast<long>(input, other, out, tiling_data);
    else if (TILING_KEY_IS(227))
        fmin_broadcast<bfloat16_t>(input, other, out, tiling_data);
}
