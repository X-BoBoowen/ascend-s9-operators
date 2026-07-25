#include "kernel_operator.h"
#include <cmath>
#include <limits>

using namespace AscendC;

// =================================================================================================
// TILING_KEY is 0: dim is the last dimension (Continuous Case)
// REWRITTEN FOR NUMERICAL STABILITY & OPTIMIZED WITH VECTOR-SCALAR OPERATIONS
// =================================================================================================

constexpr int32_t BLOCK_SIZE = 1024;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t CALC_BUFFER_SIZE = 6 * BLOCK_SIZE;

// 用于安全地进行 float 和 int 之间的位操作
union FloatIntUnion {
    float f;
    uint32_t i;
};

/**
 * @brief 使用范围缩减和泰勒级数逼近来计算 exp(x)，专为 x <= 0 优化。
 *        exp(x) = 2^k * exp(r), 其中 x = k*ln(2) + r
 * @param x 输入值，必须小于等于 0。
 * @return exp(x) 的高精度近似值。
 */
__aicore__ inline float exp_approx(float x) {
    // 检查下溢出边界。log(FLT_MIN) 大约是 -87.3
    if (x < -87.3f) {
        return 0.0f;
    }

    // 常量
    constexpr float LN2 = 0.69314718056f;       // ln(2)
    constexpr float INV_LN2 = 1.44269504089f;   // 1 / ln(2)

    // 1. 范围缩减
    // k = round(x / ln(2))
    int32_t k = static_cast<int32_t>(x * INV_LN2 - 0.5f); // 对于负数，减0.5等效于round
    // r = x - k * ln(2)
    float r = x - k * LN2;

    // 2. 使用泰勒级数（以霍纳法则评估）逼近 exp(r)
    // exp(r) ≈ 1 + r + r²/2! + r³/3! + r⁴/4! + r⁵/5!
    // P(r) = 1 + r(1 + r(1/2 + r(1/6 + r(1/24 + r/120))))
    constexpr float C5 = 1.0f / 120.0f; // 1/5!
    constexpr float C4 = 1.0f / 24.0f;  // 1/4!
    constexpr float C3 = 1.0f / 6.0f;   // 1/3!
    constexpr float C2 = 1.0f / 2.0f;   // 1/2!
    
    float exp_r = r * (C5 * r + C4);
    exp_r = r * (exp_r + C3);
    exp_r = r * (exp_r + C2);
    exp_r = r * (exp_r + 1.0f);
    exp_r = exp_r + 1.0f;

    // 3. 重构结果: exp_r * 2^k
    // 通过直接操作IEEE-754浮点数的指数位来实现 2^k
    FloatIntUnion scale;
    // 1.0f 的整数表示是 0x3F800000。指数位从第23位开始。
    // (k << 23) 将 k 加到指数上（考虑到偏置）。
    scale.i = (k << 23) + 0x3F800000;
    
    return exp_r * scale.f;
}

/**
 * @brief 使用 atanh 恒等式和泰勒级数逼近来计算 log1p(f)，专为 0 < f <= 1 优化。
 *        log1p(f) = 2 * atanh(f / (2+f))
 * @param f 输入值，必须在 (0, 1] 范围内。
 * @return log1p(f) 的高精度近似值。
 */
__aicore__ inline float log1p_approx(float f) {
    // 当 f 非常小时, log1p(f) ≈ f
    if (f < 1e-5f) {
        return f;
    }

    // 1. 使用恒等式进行范围缩减
    float y = f / (2.0f + f);
    float y2 = y * y; // y^2

    // 2. 使用 atanh(y) 的泰勒级数（以霍纳法则评估）
    // atanh(y) ≈ y + y³/3 + y⁵/5 + y⁷/7 + y⁹/9
    // P(y) = y * (1 + y²(1/3 + y²(1/5 + y²(1/7 + y²/9))))
    constexpr float C9 = 1.0f / 9.0f;
    constexpr float C7 = 1.0f / 7.0f;
    constexpr float C5 = 1.0f / 5.0f;
    constexpr float C3 = 1.0f / 3.0f;
    
    float atanh_y = y2 * (C9 * y2 + C7);
    atanh_y = y2 * (atanh_y + C5);
    atanh_y = y2 * (atanh_y + C3);
    atanh_y = y * (atanh_y + 1.0f);
    
    // 3. 重构结果
    return 2.0f * atanh_y;
}

#define MIN(a, b) ((a) < (b) ? (a) : (b))

template<typename T>
class KernelLogcumsumexp {
public:
    __aicore__ inline KernelLogcumsumexp() {}

    // 辅助函数：矢量化前缀和（当前使用简单的串行累加实现）
    __aicore__ inline void VectorizedPrefixSum(LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t size) {
        float cumulative_sum = 0.0;
        for (uint32_t i = 0; i < size; ++i) {
            cumulative_sum += src.GetValue(i);
            dst.SetValue(i, cumulative_sum);
        }
    }

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR out, uint32_t batch_size, uint32_t length) {
        this->batch_size = batch_size;
        this->length = length;
        
        this->input_gm.SetGlobalBuffer((__gm__ T*)input);
        this->out_gm.SetGlobalBuffer((__gm__ T*)out);
        
        pipe.InitBuffer(in_queue, BUFFER_NUM, BLOCK_SIZE * sizeof(T));
        pipe.InitBuffer(out_queue, BUFFER_NUM, BLOCK_SIZE * sizeof(T));
        
        pipe.InitBuffer(calc_buf, CALC_BUFFER_SIZE * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t b = GetBlockIdx(); b < this->batch_size; b += GetBlockNum()) {
            ProcessSingleBatch(b);
        }
    }

private:
    __aicore__ inline void IterateAndCalculateLSE(
        LocalTensor<float>& in_data,
        LocalTensor<float>& out_data,
        uint32_t size,
        float& running_lse)
    {
        for (uint32_t i = 0; i < size; ++i) {
            float current_x = in_data.GetValue(i);
        
            // 计算 new_lse = LogSumExp(running_lse, current_x)
            // 使用数值稳定的公式: max(a,b) + log(1 + exp(min(a,b) - max(a,b)))
            float max_val = (running_lse > current_x) ? running_lse : current_x;
            float min_val = (running_lse < current_x) ? running_lse : current_x;
        
            if (min_val == -1e30f) {
                running_lse = max_val;
            } else {
                float diff = min_val - max_val; // diff is guaranteed to be <= 0
            
                // 计算 log(1 + exp(diff))，即 log1p(exp(diff))
                float exp_diff = exp_approx(diff);
                float log_add = log1p_approx(exp_diff);
            
                running_lse = max_val + log_add;
            }
        
            // 将当前步骤的结果存入输出缓冲区
            out_data.SetValue(i, running_lse);
        }
    }

    __aicore__ inline void ProcessSingleBatch(uint32_t batch_idx) {
        // running_lse 是跨块传递的累积状态变量
        float running_lse = -1e30f;

        // 仅为标量计算分配一个小的固定工作区，所有路径共用

        for (uint32_t offset = 0; offset < this->length; offset += BLOCK_SIZE) {
            uint32_t current_block_size = MIN(BLOCK_SIZE, this->length - offset);
        
            // 1. 数据加载，并提前分配好输出Tensor
            CopyIn(batch_idx, offset, current_block_size);
            LocalTensor<T> in_local = in_queue.DeQue<T>();
            LocalTensor<T> out_local = out_queue.AllocTensor<T>();

            constexpr uint32_t base = 64;

            // 2. 根据输入类型，选择最高效的计算路径
            if constexpr (std::is_same_v<T, float>) {
                // ========================= OPTIMIZED PATH for float =========================
                // 输入和输出类型均为 float, 直接将 in_local 的数据计算后写入 out_local。
                // 这种方式避免了两次内部 DataCopy (in_local -> in_fp32 和 result_fp32 -> out_local)，
                // 实现了零中间拷贝的最高效数据流。
                IterateAndCalculateLSE(in_local, out_local, current_block_size, running_lse);

            } else {
                // ========================= STANDARD PATH for non-float ======================
                // 输入类型不是 float, 需要临时的 float 缓冲区进行 Cast 和计算。
                LocalTensor<float> in_fp32 = calc_buf.GetWithOffset<float>(BLOCK_SIZE, base);
                LocalTensor<float> result_fp32 = calc_buf.GetWithOffset<float>(BLOCK_SIZE, (BLOCK_SIZE) * sizeof(float) + base);
            
                // a. 将输入 Cast 为 float
                Cast(in_fp32, in_local, RoundMode::CAST_NONE, current_block_size);
            
                // b. 在 float 缓冲区上执行计算
                IterateAndCalculateLSE(in_fp32, result_fp32, current_block_size, running_lse);
            
                // c. 将 float 结果 Cast 回原始类型到输出缓冲区
                Cast(out_local, result_fp32, RoundMode::CAST_RINT, current_block_size);
            }
        
            // 3. 释放输入 Tensor
            in_queue.FreeTensor(in_local);

            // 4. 将填充好数据的输出 Tensor 入队并搬运回 GM
            out_queue.EnQue(out_local);
            CopyOut(batch_idx, offset, current_block_size);
        }
    }    
    __aicore__ inline void CopyIn(uint32_t batch_idx, uint32_t offset, uint32_t size) {
        LocalTensor<T> in_local = in_queue.AllocTensor<T>();
        uint32_t gm_offset = batch_idx * this->length + offset;
        
        DataCopyExtParams copy_params{1, static_cast<uint32_t>(size * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad_params{false, 0, 0, (T)0};
        DataCopyPad(in_local, input_gm[gm_offset], copy_params, pad_params);
        
        in_queue.EnQue(in_local);
    }

    __aicore__ inline void CopyOut(uint32_t batch_idx, uint32_t offset, uint32_t size) {
        LocalTensor<T> out_local = out_queue.DeQue<T>();
        uint32_t gm_offset = batch_idx * this->length + offset;
        
        DataCopyExtParams copy_params{1, static_cast<uint32_t>(size * sizeof(T)), 0, 0, 0};
        DataCopyPad(out_gm[gm_offset], out_local, copy_params);
        
        out_queue.FreeTensor(out_local);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> in_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> out_queue;
    TBuf<TPosition::VECCALC> calc_buf;

    GlobalTensor<T> input_gm;
    GlobalTensor<T> out_gm;
    
    uint32_t batch_size;
    uint32_t length;
};

// =================================================================================================
// TILING_KEY is 1: dim is NOT the last dimension (Non-Continuous Case)
// REWRITTEN FOR NUMERICAL STABILITY
// =================================================================================================

template<typename T>
class KernelLogcumsumexpNonCont {
public:
    // TILE_SIZE defines the number of vectors we process in parallel
    static constexpr int32_t TILE_SIZE = 1024;
    
    __aicore__ inline KernelLogcumsumexpNonCont() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR out, uint32_t batch_size, uint32_t reduce_len, uint32_t batch_size_vec) {
        this->batch_size = batch_size;
        this->reduce_len = reduce_len;
        this->batch_size_vec = batch_size_vec;
        
        this->input_gm.SetGlobalBuffer((__gm__ T*)input);
        this->out_gm.SetGlobalBuffer((__gm__ T*)out);
        
        pipe.InitBuffer(in_queue, BUFFER_NUM, TILE_SIZE * sizeof(T));
        pipe.InitBuffer(out_queue, BUFFER_NUM, TILE_SIZE * sizeof(T));
        
        // Allocate buffer for fp32 calculations.
        // We need space for: prefix_lse, in_fp32, result, max_vec, min_vec
        // 5 vectors of TILE_SIZE should be sufficient.
        pipe.InitBuffer(calc_buf, 5 * TILE_SIZE * sizeof(float));
    }

    __aicore__ inline void Process() {
        uint32_t tiles_count = (this->batch_size_vec + TILE_SIZE - 1) / TILE_SIZE;
        uint32_t total_tasks = this->batch_size * tiles_count;

        for (uint32_t task_id = GetBlockIdx(); task_id < total_tasks; task_id += GetBlockNum()) {
            uint32_t p_idx = task_id / tiles_count;
            uint32_t tiles_idx = task_id % tiles_count;
            uint32_t tile_base = tiles_idx * TILE_SIZE;
            uint32_t current_tile_size = MIN(TILE_SIZE, this->batch_size_vec - tile_base);
            ProcessSingleTile(p_idx, tile_base, current_tile_size);
        }
    }

private:
    __aicore__ inline void ProcessSingleTile(uint32_t batch_idx, uint32_t tile_base, uint32_t tile_size) {
        // State variable is now a single vector in the log-domain
        LocalTensor<float> prefix_lse_vec = calc_buf.Get<float>(TILE_SIZE);
        
        // Buffers for intermediate calculations
        LocalTensor<float> in_vec_fp32 = calc_buf.GetWithOffset<float>(TILE_SIZE, 1 * TILE_SIZE * sizeof(float));
        LocalTensor<float> result_vec = calc_buf.GetWithOffset<float>(TILE_SIZE, 2 * TILE_SIZE * sizeof(float));
        LocalTensor<float> max_vec = calc_buf.GetWithOffset<float>(TILE_SIZE, 3 * TILE_SIZE * sizeof(float));
        LocalTensor<float> min_vec = calc_buf.GetWithOffset<float>(TILE_SIZE, 4 * TILE_SIZE * sizeof(float));

        // Initialize state vector to -infinity
        Duplicate(prefix_lse_vec, -1e30f, tile_size);
        
        uint32_t base_gm_offset = batch_idx * this->reduce_len * this->batch_size_vec;
        
        // Innermost loop: Iterate along the 'dim' axis
        for (uint32_t l = 0; l < this->reduce_len; ++l) {
            uint32_t current_gm_offset = base_gm_offset + l * this->batch_size_vec + tile_base;
            
            // 1. Copy data in and cast to fp32
            CopyInNonCont(current_gm_offset, tile_size);
            LocalTensor<T> in_local = in_queue.DeQue<T>();
            if constexpr (std::is_same_v<T, float>) {
                DataCopy(in_vec_fp32, in_local, TILE_SIZE);
            } else {
                Cast(in_vec_fp32, in_local, RoundMode::CAST_NONE, tile_size);
            }
            in_queue.FreeTensor(in_local);

            // 2. Compute the new LSE: result = LogSumExp(prefix_lse_vec, in_vec_fp32)
            // This is the vectorized, numerically stable update step.
            Max(max_vec, prefix_lse_vec, in_vec_fp32, tile_size);
            Min(min_vec, prefix_lse_vec, in_vec_fp32, tile_size);

            Sub(min_vec, min_vec, max_vec, tile_size);           // min - max
            Exp(min_vec, min_vec, tile_size);                   // exp(min - max)
            Adds(min_vec, min_vec, 1.0f, tile_size);            // 1 + exp(...)
            Log(min_vec, min_vec, tile_size);                   // log(1 + exp(...))
            
            Add(result_vec, max_vec, min_vec, tile_size);
            
            // 3. Update state for the next iteration
            // The current result becomes the prefix for the next step.
            DataCopy(prefix_lse_vec, result_vec, TILE_SIZE);

            // 4. Cast back the current result and copy out
            LocalTensor<T> out_local = out_queue.AllocTensor<T>();
            if constexpr (std::is_same_v<T, float>) {
                DataCopy(out_local, result_vec, TILE_SIZE);
            } else {
                Cast(out_local, result_vec, RoundMode::CAST_RINT, tile_size);
            }
            out_queue.EnQue(out_local);
            CopyOutNonCont(current_gm_offset, tile_size);
        }
    }
    
    __aicore__ inline void CopyInNonCont(uint32_t gm_offset, uint32_t size) {
        LocalTensor<T> in_local = in_queue.AllocTensor<T>();
        DataCopyExtParams copy_params{1, static_cast<uint32_t>(size * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad_params{false, 0, 0, (T)0};
        DataCopyPad(in_local, input_gm[gm_offset], copy_params, pad_params);
        in_queue.EnQue(in_local);
    }

    __aicore__ inline void CopyOutNonCont(uint32_t gm_offset, uint32_t size) {
        LocalTensor<T> out_local = out_queue.DeQue<T>();
        DataCopyExtParams copy_params{1, static_cast<uint32_t>(size * sizeof(T)), 0, 0, 0};
        DataCopyPad(out_gm[gm_offset], out_local, copy_params);
        out_queue.FreeTensor(out_local);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> in_queue;
    TQue<TPosition::VECOUT, BUFFER_NUM> out_queue;
    TBuf<TPosition::VECCALC> calc_buf;

    GlobalTensor<T> input_gm;
    GlobalTensor<T> out_gm;
    
    uint32_t batch_size;
    uint32_t reduce_len;
    uint32_t batch_size_vec;
};

// =================================================================================================
// Main Kernel Entry Point
// =================================================================================================
extern "C" __global__ __aicore__ void logcumsumexp(GM_ADDR input, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (TILING_KEY_IS(0)) {
        // Case: dim is the last dimension. Elements are continuous.
        KernelLogcumsumexp<DTYPE_INPUT> op;
        op.Init(input, out, tiling_data.batch_size, tiling_data.length);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        // Case: dim is NOT the last dimension. Elements are non-continuous.
        KernelLogcumsumexpNonCont<DTYPE_INPUT> op;
        op.Init(input, out, tiling_data.batch_size, tiling_data.length, tiling_data.batch_size_vec);
        op.Process();
    }
}