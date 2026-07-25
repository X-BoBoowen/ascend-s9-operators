// #ifdef DUMMY
// #define __CCE_AICORE__ 220
// #define __DAV_C220_CUBE__
// #define __DAV_C220_VEC__
// #include <tikicpulib.h>
// #include <__clang_cce_aicore_functions.h>
// #endif

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

__aicore__ inline constexpr int ceil_div(int x, int y)
{
    return (x - 1) / y + 1;
}

__aicore__ inline constexpr int ceil_round(int x, int y)
{
    return ceil_div(x, y) * y;
}

constexpr uint32_t COMPLEX_FLOAT_SIZE_U32 = sizeof(float) * 2;

// 计算步长 (32位)
__aicore__ inline bool GetStrides_u32(const int32_t dims[8], uint32_t ndim, uint32_t strides[8]) {
    if (ndim == 0) return true;
    const uint32_t max_dims = 8;
    for(uint32_t i=ndim; i<max_dims; ++i) strides[i] = UINT32_MAX;

    strides[ndim - 1] = 1;
    for (int32_t i = static_cast<int32_t>(ndim) - 2; i >= 0; --i) {
        uint32_t next_dim_size = (dims[i + 1] <= 0) ? 1 : static_cast<uint32_t>(dims[i+1]);
        // 简单的 32位溢出检查
        if (next_dim_size > 0 && strides[i + 1] > UINT32_MAX / next_dim_size) {
             for (int32_t k=i; k >=0; --k) strides[k] = UINT32_MAX;
             return false; // 溢出
        }
        strides[i] = strides[i + 1] * next_dim_size;
    }
    return true;
}

// 线性批索引 -> 多维批索引 (32位)
__aicore__ inline void GetBatchMultiDimIndex_u32(
    uint32_t linear_batch_idx,
    const int32_t batch_dims[8],
    uint32_t batch_ndim,
    uint32_t batch_multi_idx[8])
{
    uint32_t current_idx = linear_batch_idx;
    for (int32_t i = static_cast<int32_t>(batch_ndim) - 1; i >= 0; --i) {
        uint32_t dim_size = (batch_dims[i] <= 0) ? 1 : static_cast<uint32_t>(batch_dims[i]);
        if (dim_size == 0) { batch_multi_idx[i] = 0; continue; }
        batch_multi_idx[i] = current_idx % dim_size;
        current_idx /= dim_size;
    }
    for(uint32_t i=batch_ndim; i<8; ++i) batch_multi_idx[i]=0;
}

// 多维索引 -> 线性偏移 (32位)
__aicore__ inline uint32_t GetLinearOffsetFromMultiIndex_u32(
    const uint32_t multi_idx[8],
    const uint32_t strides[8],
    uint32_t ndim)
{
    uint32_t offset = 0;
    for (uint32_t i = 0; i < ndim; ++i) {
        uint32_t term_offset = 0;
        // 简单溢出检查
        if (multi_idx[i] > 0 && strides[i] != UINT32_MAX && strides[i] > UINT32_MAX / multi_idx[i]) return UINT32_MAX;
        if (strides[i] == UINT32_MAX && multi_idx[i] != 0) return UINT32_MAX;

        term_offset = multi_idx[i] * strides[i];
        if (offset > UINT32_MAX - term_offset) return UINT32_MAX;
        offset += term_offset;
    }
    return offset;
}

__aicore__ inline bool CheckBroadcastCompatibilityAndGetType_u32(
    const int32_t z_dims[8], uint32_t z_ndim,
    const int32_t bias_dims[8], uint32_t bias_ndim,
    uint32_t M, uint32_t N, // Z's M and N dims (uint32_t), 必须 > 0
    int32_t& bias_type)
{
    bias_type = -1; // 默认为未分类/不兼容
    if (z_ndim == 0) return false; // Z 通常至少是 1D 或 2D

    // --- 1. 标准广播兼容性检查 (必需的前提) ---
    uint32_t max_ndim = (z_ndim > bias_ndim) ? z_ndim : bias_ndim;
    bool compatible = true;
    for (int32_t i = 0; i < static_cast<int32_t>(max_ndim); ++i) {
        // 从右向左获取维度索引
        int32_t z_idx = static_cast<int32_t>(z_ndim) - 1 - i;
        int32_t bias_idx = static_cast<int32_t>(bias_ndim) - 1 - i;

        // 获取维度值，越界或非正则视为 1 (广播兼容性检查中)
        uint32_t z_dim = (z_idx >= 0 && z_dims[z_idx] > 0) ? static_cast<uint32_t>(z_dims[z_idx]) : 1;
        uint32_t bias_dim = (bias_idx >= 0 && bias_dims[bias_idx] > 0) ? static_cast<uint32_t>(bias_dims[bias_idx]) : 1;

        // 广播规则：维度相等，或其中一个为 1
        if (z_dim != bias_dim && z_dim != 1 && bias_dim != 1) {
            compatible = false;
            break;
        }
    }

    if (!compatible) {
        // PRINTF("Error: Bias shape is not broadcast compatible with Z shape.\n");
        return false; // 如果基础广播不兼容，直接返回 false
    }

    // --- 2. 提取 Bias 的有效 M 和 N (根据用户规则) ---
    uint32_t bias_eff_M = 1;
    uint32_t bias_eff_N = 1;

    if (bias_ndim == 1) {
        // 1D Bias: M=1, N=dim[0]
        bias_eff_N = (bias_dims[0] > 0) ? static_cast<uint32_t>(bias_dims[0]) : 1;
    } else if (bias_ndim >= 2) {
        // 2D+ Bias: M=dim[N-2], N=dim[N-1]
        bias_eff_M = (bias_dims[bias_ndim - 2] > 0) ? static_cast<uint32_t>(bias_dims[bias_ndim - 2]) : 1;
        bias_eff_N = (bias_dims[bias_ndim - 1] > 0) ? static_cast<uint32_t>(bias_dims[bias_ndim - 1]) : 1;
    }
    // 如果 bias_ndim == 0 (标量), bias_eff_M 和 bias_eff_N 保持为 1

    // --- 3. 根据有效 M/N 与 Z 的 M/N 进行分类 ---
    // 注意：这里的 M 和 N 是 Z 的 M 和 N，由调用者传入
    if (bias_eff_M == M && bias_eff_N == N) {
        bias_type = 3; // 矩阵维度完全匹配 Z 的 M, N
    } else if (bias_eff_M == 1 && bias_eff_N == N) {
        bias_type = 1; // 行向量广播 (有效 M=1, 有效 N 匹配 Z 的 N)
    } else if (bias_eff_M == M && bias_eff_N == 1) {
        bias_type = 2; // 列向量广播 (有效 M 匹配 Z 的 M, 有效 N=1)
    } else if (bias_eff_M == 1 && bias_eff_N == 1) {
        bias_type = 0; // 标量广播 (有效 M=1, 有效 N=1)
    } else {
        // 虽然整体形状可广播，但最后两维的模式不符合 Type 0/1/2/3
        // 这意味着批处理维度起了作用，或者 M/N 本身就需要广播 (例如 Z=[1,N], Bias=[M,N])
        // 根据用户定义的简化规则，这种情况不归类。
        bias_type = -1;
        // PRINTF("Warning: Bias shape broadcast compatible, but last two dims don't match Type 0/1/2/3 pattern.\n");
        return false; // 返回 false 表示未成功分类到 0, 1, 2, 3
    }

    // 如果成功分类为 0, 1, 2, 或 3
    return true;
}

__aicore__ inline uint32_t DecodeShapeInfo(const uint32_t shape_info[8], int32_t dims[8], uint32_t& ndim) {
    ndim = shape_info[0];
    if (ndim > 7) ndim = 7; 

    uint32_t total_elements = 1;
    // PRINTF("shape: ");
    for (uint32_t i = 0; i < ndim; ++i) {
        dims[i] = static_cast<int32_t>(shape_info[i + 1]);
        // PRINTF("%d ", dims[i]);
        if (dims[i] <= 0) return 0; 
        if (total_elements > UINT32_MAX / static_cast<uint32_t>(dims[i])) {
             return 0; // Overflow
        }
        total_elements *= static_cast<uint32_t>(dims[i]);
    }
    // PRINTF("\n");
    for (uint32_t i = ndim; i < 8; ++i) {
        dims[i] = 1;
    }
    return total_elements;
}

template<typename A_T, typename B_T, typename C_T>
class MatmulCustom
{
    AscendC::GlobalTensor<A_T> a_global_tensor;
    AscendC::GlobalTensor<B_T> b_global_tensor;
    AscendC::GlobalTensor<C_T> c_global_tensor;
    AscendC::TQue<AscendC::QuePosition::A1, 1> a1_t_que;
    AscendC::TQue<AscendC::QuePosition::A2, 1> a2_t_que;
    AscendC::TQue<AscendC::QuePosition::B1, 1> b1_t_que;
    AscendC::TQue<AscendC::QuePosition::B2, 1> b2_t_que;
    AscendC::TQue<AscendC::QuePosition::CO1, 1> co1_t_que;
    static constexpr int L0_M = 128;
    static constexpr int L0_N = 128;
    static constexpr int L0_K = 64;
    static constexpr int L1_M_BLOCKS = 2;
    static constexpr int L1_N_BLOCKS = 2;
    static constexpr int L1_K_BLOCKS = 2;
    static constexpr int L1_M = L0_M * L1_M_BLOCKS;
    static constexpr int L1_N = L0_N * L1_N_BLOCKS;
    static constexpr int L1_K = L0_K * L1_K_BLOCKS;
    int offsetA, offsetB, offsetC;
    int singleCoreM, singleCoreN, singleCoreK;
    int N, K;
public:
    __aicore__ inline MatmulCustom() {}
    __aicore__ inline void Init(AscendC::TPipe *t_pipe) {
        t_pipe->InitBuffer(a1_t_que, 2, L1_M * L1_K * sizeof(A_T));
        t_pipe->InitBuffer(a2_t_que, 2, L0_M * L0_K * sizeof(A_T));
        t_pipe->InitBuffer(b1_t_que, 2, L1_K * L1_N * sizeof(B_T));
        t_pipe->InitBuffer(b2_t_que, 2, L0_K * L0_N * sizeof(B_T));
        t_pipe->InitBuffer(co1_t_que, 2, L0_M * L0_N * sizeof(C_T));
    }
    __aicore__ inline void SetMatrix(AscendC::GlobalTensor<A_T> aGlobal, AscendC::GlobalTensor<B_T> bGlobal, AscendC::GlobalTensor<C_T> cGlobal, int N, int K) {
        this->N = N;
        this->K = K;
        this->a_global_tensor = aGlobal;
        this->b_global_tensor = bGlobal;
        this->c_global_tensor = cGlobal;
    }
    __aicore__ inline void Process(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, bool if_atomicadd) {
        this->offsetA = offsetA;
        this->offsetB = offsetB;
        this->offsetC = offsetC;
        this->singleCoreM = blockM;
        this->singleCoreN = blockN;
        this->singleCoreK = blockK;

        static constexpr uint8_t pad_list[4]{};

        int l2_m_blocks = ceil_div(singleCoreM, L1_M);
        int l2_n_blocks = ceil_div(singleCoreN, L1_N);
        int l2_k_blocks = ceil_div(singleCoreK, L1_K);

        for(int i = 0; i < l2_k_blocks; i++) 
        {
            int l1_k = min(singleCoreK - i * L1_K, L1_K);
            int l1_k_aligned = ceil_round(l1_k, 8);
            AscendC::SetFmatrix(1, l1_k_aligned, pad_list, AscendC::FmatrixMode::FMATRIX_RIGHT);
            for(int j = 0; j < l2_n_blocks; j++) 
            {
                int l1_n = min(singleCoreN - j * L1_N, L1_N);
                int l1_n_aligned = ceil_round(l1_n, 16);
                // copy in b
                {
                    AscendC::LocalTensor<B_T> b1_local_tensor = b1_t_que.AllocTensor<B_T>();
                    AscendC::Nd2NzParams nd2_nz_params;
                    nd2_nz_params.ndNum = 1;
                    nd2_nz_params.nValue = l1_k;
                    nd2_nz_params.dValue = l1_n;
                    nd2_nz_params.srcDValue = N;
                    nd2_nz_params.dstNzC0Stride = l1_k_aligned;
                    nd2_nz_params.dstNzNStride = 1;
                    AscendC::DataCopy(b1_local_tensor, b_global_tensor[offsetB + i * L1_K * N + j * L1_N], nd2_nz_params);
                    b1_t_que.EnQue(b1_local_tensor);
                }
                AscendC::LocalTensor<B_T> b1_local_tensor = b1_t_que.DeQue<B_T>();
                for(int k = 0; k < l2_m_blocks; k++)
                {
                    int l1_m = min(singleCoreM - k * L1_M, L1_M);
                    int l1_m_aligned = ceil_round(l1_m, 16);
                    AscendC::SetFmatrix(1, l1_m_aligned, pad_list, AscendC::FmatrixMode::FMATRIX_LEFT);
                    // copy in a
                    {
                        AscendC::LocalTensor<A_T> a1_local_tensor = a1_t_que.AllocTensor<A_T>();
                        AscendC::Nd2NzParams nd2_nz_params;
                        nd2_nz_params.ndNum = 1;
                        nd2_nz_params.nValue = l1_m;
                        nd2_nz_params.dValue = l1_k;
                        nd2_nz_params.srcDValue = K;
                        nd2_nz_params.dstNzC0Stride = l1_m_aligned;
                        nd2_nz_params.dstNzNStride = 1;
                        AscendC::DataCopy(a1_local_tensor, a_global_tensor[offsetA + k * L1_M * K + i * L1_K], nd2_nz_params);
                        a1_t_que.EnQue(a1_local_tensor);
                    }
                    AscendC::LocalTensor<A_T> a1_local_tensor = a1_t_que.DeQue<A_T>();

                    int l1_m_blocks = ceil_div(l1_m, L0_M);
                    for (int m = 0; m < l1_m_blocks; m++)
                    {
                        int l0_m = min(l1_m - m * L0_M, L0_M);
                        int l0_m_aligned = ceil_round(l0_m, 16);
                        int l1_n_blocks = ceil_div(l1_n, L0_N);
                        for (int n = 0; n < l1_n_blocks; n++)
                        {
                            int l0_n = min(l1_n - n * L0_N, L0_N);
                            int l0_n_aligned = ceil_round(l0_n, 16);
                            AscendC::LocalTensor<C_T> co1_local_tensor = co1_t_que.AllocTensor<C_T>();
                            int l1_k_blocks = ceil_div(l1_k, L0_K);
                            for (int o = 0; o < l1_k_blocks; o++) 
                            {
                                int l0_k = min(l1_k - o * L0_K, L0_K);
                                int l0_k_aligned = ceil_round(l0_k, 8);
                                // split a
                                {
                                    AscendC::LocalTensor<A_T> a2_local_tensor = a2_t_que.AllocTensor<A_T>();
                                    AscendC::LoadData3DParamsV2Pro load_data_3d_params_v2_pro;
                                    load_data_3d_params_v2_pro.channelSize = l1_k_aligned;
                                    short (&extConfig)[4] = *reinterpret_cast<short (*)[4]>(&load_data_3d_params_v2_pro.extConfig);
                                    extConfig[0] = l0_k_aligned;
                                    extConfig[1] = l0_m_aligned;
                                    extConfig[2] = o * L0_K;
                                    extConfig[3] = m * L0_M;
                                    AscendC::LoadData(a2_local_tensor, a1_local_tensor, load_data_3d_params_v2_pro);
                                    a2_t_que.EnQue(a2_local_tensor);
                                }
                                // split b
                                {
                                    AscendC::LocalTensor<B_T> b2_local_tensor = b2_t_que.AllocTensor<B_T>();
                                    AscendC::LoadData3DParamsV2Pro load_data_3d_params_v2_pro;
                                    load_data_3d_params_v2_pro.channelSize = l1_n_aligned;
                                    load_data_3d_params_v2_pro.fMatrixCtrl = true;
                                    short (&extConfig)[4] = *reinterpret_cast<short (*)[4]>(&load_data_3d_params_v2_pro.extConfig);
                                    extConfig[0] = l0_n_aligned;
                                    extConfig[1] = l0_k_aligned;
                                    extConfig[2] = n * L0_N;
                                    extConfig[3] = o * L0_K;
                                    AscendC::LoadData(b2_local_tensor, b1_local_tensor, load_data_3d_params_v2_pro);
                                    b2_t_que.EnQue(b2_local_tensor);
                                }
                                // compute
                                {
                                    AscendC::LocalTensor<A_T> a2_local_tensor = a2_t_que.DeQue<A_T>();
                                    AscendC::LocalTensor<B_T> b2_local_tensor = b2_t_que.DeQue<B_T>();
                                    AscendC::MmadParams mmad_params;
                                    mmad_params.m = max(l0_m, 2);
                                    mmad_params.n = l0_n;
                                    mmad_params.k = l0_k;
                                    mmad_params.cmatrixInitVal = o == 0;
                                    AscendC::Mmad(co1_local_tensor, a2_local_tensor, b2_local_tensor, mmad_params);
                                    if (l0_m * l0_n < 2560)
                                        AscendC::PipeBarrier<PIPE_M>();
                                    a2_t_que.FreeTensor(a2_local_tensor);
                                    b2_t_que.FreeTensor(b2_local_tensor);
                                }
                            }
                            co1_t_que.EnQue(co1_local_tensor);
                            // copy out
                            {
                                co1_local_tensor = co1_t_que.DeQue<C_T>();
                                AscendC::FixpipeParamsV220 fixpipe_params_v220;
                                fixpipe_params_v220.nSize = l0_n;
                                fixpipe_params_v220.mSize = l0_m;
                                fixpipe_params_v220.srcStride = l0_m_aligned;
                                fixpipe_params_v220.dstStride = N;
                                if(if_atomicadd || i) AscendC::SetAtomicAdd<float>();
                                AscendC::Fixpipe(c_global_tensor[offsetC + (k * L1_M + m * L0_M )* N + n * L0_N + j * L1_N], co1_local_tensor, fixpipe_params_v220);
                                if(if_atomicadd || i) AscendC::SetAtomicNone();
                                co1_t_que.FreeTensor(co1_local_tensor);
                            }
                        }
                    }
                    a1_t_que.FreeTensor(a1_local_tensor);
                }
                b1_t_que.FreeTensor(b1_local_tensor);
            }
        }
    }
};

template<typename T>
class SplitRealImag
{
    static constexpr int TILE_LENGTH = 8192;
    GlobalTensor<T> srcGlobal, dstGlobal, dstNegGlobal;
    int srcN;
    int dstM;
    int dstN;
    int padN;
    int alignedN;
    int linesPerIter;
    int elementsPerBlock;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueueReal, outQueueImag;
    LocalTensor<T> realLocal, imagLocal;
    bool enNeg;

public:
    __aicore__ inline SplitRealImag() {}
    __aicore__ inline void Init(TQue<QuePosition::VECIN, 1> inQueue, TQue<QuePosition::VECOUT, 1> outQueueReal, TQue<QuePosition::VECOUT, 1> outQueueImag)
    {
        this->inQueue = inQueue;
        this->outQueueReal = outQueueReal;
        this->outQueueImag = outQueueImag;
        elementsPerBlock = 32 / sizeof(T);
    }

    __aicore__ inline void SetMatrix(GlobalTensor<T> srcGlobal, GlobalTensor<T> dstGlobal, int srcN, int dstM, int dstN)
    {
        this->srcGlobal = srcGlobal;
        this->dstGlobal = dstGlobal;
        enNeg = false;
        this->srcN = srcN;
        this->dstM = dstM;
        this->dstN = dstN;
        // assert(srcN <= 8192);
        alignedN = (srcN + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
        linesPerIter = TILE_LENGTH / alignedN;
        linesPerIter = max(1, linesPerIter & ~1);
        padN = (elementsPerBlock - (srcN * 2 % elementsPerBlock)) % elementsPerBlock;
    }

    __aicore__ inline void SetMatrix(GlobalTensor<T> srcGlobal, GlobalTensor<T> dstGlobal, GlobalTensor<T> dstNegGlobal, int srcN, int dstM, int dstN)
    {
        this->srcGlobal = srcGlobal;
        this->dstGlobal = dstGlobal;
        this->dstNegGlobal = dstNegGlobal;
        enNeg = true;
        this->srcN = srcN;
        this->dstM = dstM;
        this->dstN = dstN;
        alignedN = (srcN + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
        linesPerIter = TILE_LENGTH / alignedN;
        linesPerIter = max(1, linesPerIter & ~1);
        padN = (elementsPerBlock - (srcN * 2 % elementsPerBlock)) % elementsPerBlock;
    }

    __aicore__ inline void SplitRealImagWork(int M, int p, int id)
    {
        int k = M / p, r = M % p;
        Process((k * id + min(id, r)) * srcN * 2, (k * id + min(id, r)) * dstN, k + (id < r));
    }

    __aicore__ inline void Process(int srcOffset, int dstOffset, int lines)
    {
        int loopCount = lines / linesPerIter;
        srcOffset += loopCount * linesPerIter * srcN * 2;
        dstOffset += loopCount * linesPerIter * dstN;
        int tailLines = lines - linesPerIter * loopCount;
        if (tailLines)
        {
            if (srcN << 1 <= TILE_LENGTH)
            {
                CopyIn(srcOffset, tailLines / 2);
                Compute(0, 0, tailLines / 2 * alignedN * 2);
                CopyIn(srcOffset + tailLines / 2 * srcN * 2, (tailLines + 1) / 2);
                Compute(1, tailLines / 2 * alignedN, (tailLines + 1) / 2 * alignedN * 2);
            }
            else
            {
                CopyIn(srcOffset, TILE_LENGTH);
                Compute(0, 0, TILE_LENGTH);
                CopyIn(srcOffset + TILE_LENGTH, alignedN * 2 - TILE_LENGTH);
                Compute(1, TILE_LENGTH / 2, alignedN * 2 - TILE_LENGTH);
            }
            CopyOut(dstOffset, tailLines);
        }
        srcOffset -= linesPerIter * srcN * 2;
        dstOffset -= linesPerIter * dstN;
        for (int i = 0; i < loopCount; ++i)
        {
            if (srcN << 1 <= TILE_LENGTH)
            {
                CopyIn(srcOffset, linesPerIter / 2);
                Compute(0, 0, linesPerIter / 2 * alignedN * 2);
                CopyIn(srcOffset + linesPerIter / 2 * srcN * 2, (linesPerIter + 1) / 2);
                Compute(1, linesPerIter / 2 * alignedN, (linesPerIter + 1) / 2 * alignedN * 2);
            }
            else
            {
                CopyIn(srcOffset, TILE_LENGTH);
                Compute(0, 0, TILE_LENGTH);
                CopyIn(srcOffset + TILE_LENGTH, alignedN * 2 - TILE_LENGTH);
                Compute(1, TILE_LENGTH / 2, alignedN * 2 - TILE_LENGTH);
            }
            CopyOut(dstOffset, linesPerIter);
            srcOffset -= linesPerIter * srcN * 2;
            dstOffset -= linesPerIter * dstN;
        }
    }

    __aicore__ inline void CopyIn(int offset, int lines)
    {
        LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();
        if (srcN << 1 <= TILE_LENGTH)
        {
            DataCopyExtParams intriParams{
                static_cast<uint16_t>(lines),
                static_cast<uint32_t>(srcN * 2 * sizeof(T)),
                static_cast<uint32_t>((srcN * 2 - srcN * 2) * sizeof(T)),
                srcN % elementsPerBlock && srcN % elementsPerBlock <= elementsPerBlock / 2,
                0
            };
            DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(padN), 0};
            DataCopyPad(srcLocal, srcGlobal[offset], intriParams, padParams);
        }
        else
        {
            DataCopy(srcLocal, srcGlobal[offset], lines);
        }
        inQueue.EnQue(srcLocal);
    }

    __aicore__ inline void Compute(int progress, int calcedLength, int length)
    {
        if (!progress)
        {
            realLocal = outQueueReal.AllocTensor<T>();
            imagLocal = outQueueImag.AllocTensor<T>();
        }
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        uint64_t rsvdCnt;
        GatherMask(realLocal[calcedLength], srcLocal, 1, false, 0, {1, static_cast<uint16_t>((length * sizeof(T) + 255) / 256), 8, 0}, rsvdCnt);
        GatherMask(imagLocal[calcedLength], srcLocal, 2, false, 0, {1, static_cast<uint16_t>((length * sizeof(T) + 255) / 256), 8, 0}, rsvdCnt);
        inQueue.FreeTensor(srcLocal);
        if (progress)
        {
            outQueueReal.EnQue(realLocal);
            outQueueImag.EnQue(imagLocal);
        }
    }

    __aicore__ inline void CopyOut(int offset, int lines)
    {
        LocalTensor<T> realLocal = outQueueReal.DeQue<T>();
        LocalTensor<T> imagLocal = outQueueImag.DeQue<T>();
        DataCopyExtParams intriParams{
            static_cast<uint16_t>(lines),
            static_cast<uint32_t>(srcN * sizeof(T)),
            0,
            static_cast<uint32_t>((dstN - srcN) * sizeof(T)),
            0
        };
        DataCopyPad(dstGlobal[offset], realLocal, intriParams);
        DataCopyPad(dstGlobal[offset + dstM * dstN], imagLocal, intriParams);
        outQueueReal.FreeTensor(realLocal);
        if (enNeg)
        {
            LocalTensor<T> imagNegLocal = outQueueReal.AllocTensor<T>();
            Muls(imagNegLocal, imagLocal, static_cast<T>(-1), lines * alignedN);
            outQueueReal.EnQue(imagNegLocal);
            imagNegLocal = outQueueReal.DeQue<T>();
            DataCopyPad(dstNegGlobal[offset], imagNegLocal, intriParams);
            outQueueReal.FreeTensor(imagNegLocal);
        }
        outQueueImag.FreeTensor(imagLocal);
    }
};

class BiasAdd {
    public:
        __aicore__ inline BiasAdd() {}
        __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe)
        {
            pipe->InitBuffer(inQueueX, 2, this->tileLength * sizeof(float) * 2);
            pipe->InitBuffer(outQueueY, 2, this->tileLength * sizeof(float) * 2);
        }
        __aicore__ inline void SetMatrix(GlobalTensor<float> &xGlobal, GlobalTensor<float> &yGlobal, int M, int N, int type)
        {
            this->xGlobal = xGlobal;
            this->yGlobal = yGlobal;
            this->M = M;
            this->N = N;
            this->type = type;
        }
        __aicore__ inline void Process()
        {
            int blockIdx = GetBlockIdx();
            int blockNum = GetBlockNum() * 2;
            // PRINTF("%d\n", type);
            if(type == 0 || type == 3) {
                int total = M * N;
                int chunks = (M * N + 15) / 16;
                int base_chunks = chunks / blockNum;
                int tile_chunks = chunks % blockNum;
                int start = (base_chunks * blockIdx + min(blockIdx, tile_chunks)) * 16;
                int cur_length = min((base_chunks + (blockIdx < tile_chunks))* 16, total - start);
                // PRINTF("here %d %d %d %d\n",base_chunks, tile_chunks, start, cur_length);
                int loopCount = (cur_length + tileLength - 1) / tileLength;
                if(type == 0) {
                    LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
                    DataCopy(xLocal, xGlobal, 8);
                    inQueueX.EnQue(xLocal);
                    xLocal = inQueueX.DeQue<float>();
                    uint8_t repeatTimes = ceil_div((2 * tileLength), 64);
                    uint64_t mask[2] = { 0x5555555555555555, 0 };
                    float real = xLocal.GetValue(0);
                    float imag = xLocal.GetValue(1);
                    Duplicate(yLocal, real, mask, repeatTimes, 1, 8);
                    mask[0] = ~mask[0];
                    Duplicate(yLocal, imag, mask, repeatTimes, 1, 8);
                    outQueueY.EnQue(yLocal);
                    inQueueX.FreeTensor(xLocal);
                }
                for (int32_t i = 0; i < loopCount; i++) {
                    int cur = min(cur_length - i * tileLength, tileLength);
                    if(type == 0) {
                        DataCopyExtParams copyParams{1, static_cast<uint32_t>(2 * cur * sizeof(float)), 0, 0, 0};
                        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
                        SetAtomicAdd<float>();
                        DataCopyPad(yGlobal[(start + i * tileLength) * 2], yLocal, copyParams);
                        SetAtomicNone();
                        outQueueY.EnQue(yLocal);
                    } else {
                        CopyIn((start + i * tileLength) * 2, cur, type);
                        Compute(type, cur);
                        CopyOut((start + i * tileLength) * 2, cur);
                    }
                }
                if(type == 0) {
                    LocalTensor<float> yLocal = outQueueY.DeQue<float>();
                    outQueueY.FreeTensor(yLocal);
                }
            } else {
                int base_M = M / blockNum;
                int tile_M = M % blockNum;
                int start = (base_M * blockIdx + min(tile_M, blockIdx)) * N;
                int cur_M = base_M + (blockIdx < tile_M);
                LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                if(type == 1) {
                    DataCopy(xLocal, xGlobal, 2 * ceil_round(N, 8));
                    inQueueX.EnQue(xLocal);
                    xLocal = inQueueX.DeQue<float>();
                    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
                    uint8_t repeatTimes = ceil_div((2 * N), 64);
                    uint64_t mask = 64;
                    Copy(yLocal, xLocal, mask, repeatTimes, { 1, 1, 8, 8 });
                    outQueueY.EnQue(yLocal);
                    inQueueX.EnQue(xLocal);
                } else {
                    DataCopy(xLocal, xGlobal, 2 * ceil_round(M, 8));
                    inQueueX.EnQue(xLocal);
                }
                for(int i = 0; i < cur_M; i++) {
                    if(type == 2) {
                        Compute(type, N, (start / N) + i);
                        CopyOut((start + i * N) * 2, N);
                    } else {
                        DataCopyExtParams copyParams{1, static_cast<uint32_t>(2 * N * sizeof(float)), 0, 0, 0};
                        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
                        SetAtomicAdd<float>();
                        DataCopyPad(yGlobal[(start + i * N) * 2], yLocal, copyParams);
                        SetAtomicNone();
                        outQueueY.EnQue(yLocal);
                    }
                }
                if(type == 1) {
                    LocalTensor<float> yLocal = outQueueY.DeQue<float>();
                    outQueueY.FreeTensor(yLocal);
                }
                xLocal = inQueueX.DeQue<float>();
                inQueueX.FreeTensor(xLocal);
            }
        }
    
    private:
        __aicore__ inline void CopyIn(int start, int len, int type)
        {
            LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            if(type == 0) {
                DataCopy(xLocal, xGlobal, 8);
            } else if(type == 3) {
                int align_len = ceil_round(len, 8);
                // PRINTF("%d %d\n",start,align_len);
                DataCopy(xLocal, xGlobal[start], 2 * align_len);
            }
            inQueueX.EnQue(xLocal);
        }
        __aicore__ inline void Compute(int type, int len, uint32_t idx = 0)
        {
            if(type == 2) {
                LocalTensor<float> xLocal = inQueueX.DeQue<float>();
                LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
                uint8_t repeatTimes = ceil_div((2 * len), 64);
                uint64_t mask[2] = { 0x5555555555555555, 0 };
                float real = xLocal.GetValue(idx * 2);
                float imag = xLocal.GetValue(idx * 2 + 1);
                Duplicate(yLocal, real, mask, repeatTimes, 1, 8);
                mask[0] = ~mask[0];
                Duplicate(yLocal, imag, mask, repeatTimes, 1, 8);
                outQueueY.EnQue(yLocal);
                inQueueX.EnQue(xLocal);
            } else {
                LocalTensor<float> xLocal = inQueueX.DeQue<float>();
                LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
                uint8_t repeatTimes = ceil_div((2 * len), 64);
                uint64_t mask = 64;
                Copy(yLocal, xLocal, mask, repeatTimes, { 1, 1, 8, 8 });
                outQueueY.EnQue(yLocal);
                inQueueX.FreeTensor(xLocal);
            }
        }
        __aicore__ inline void CopyOut(int start, int len)
        {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(2 * len * sizeof(float)), 0, 0, 0};
            LocalTensor<float> yLocal = outQueueY.DeQue<float>();
            SetAtomicAdd<float>();
            DataCopyPad(yGlobal[start], yLocal, copyParams);
            SetAtomicNone();
            outQueueY.FreeTensor(yLocal);
        }
    
    private:
        AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueueX;
        AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueueY;
        AscendC::GlobalTensor<float> xGlobal;
        AscendC::GlobalTensor<float> yGlobal;
        int M, N, type;
        uint32_t tileLength = 4096;
};

template<typename T>
class MergeRealImag
{
    static const int TILE_LENGTH = 4096;
    GlobalTensor<T> cGlobal, cOrgGlobal;
    int srcM;
    int srcN;
    int dstN;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TQue<QuePosition::VECIN, 1> gatherQueue;
    LocalTensor<T> realLocal, imagLocal;
    LocalTensor<uint32_t> gatherLocal;

public:
    __aicore__ inline MergeRealImag() {}
    __aicore__ inline void Init(TQue<QuePosition::VECIN, 1> inQueue, TQue<QuePosition::VECOUT, 1> outQueue, TQue<QuePosition::VECIN, 1> gatherQueue)
    {
        this->inQueue = inQueue;
        this->outQueue = outQueue;
        this->gatherQueue = gatherQueue;
    }

    __aicore__ inline void SetMatrix(const GlobalTensor<T> &cGlobal, GlobalTensor<T> &cOrgGlobal, int srcM, int srcN, int dstN)
    {
        this->cGlobal = cGlobal;
        this->cOrgGlobal = cOrgGlobal;
        this->srcM = srcM;
        this->srcN = srcN;
        this->dstN = dstN;
    }

    __aicore__ inline void MergeRealImagWork(int M, int p, int id)
    {
        int k = M / p, r = M % p;
        if((k + (id < r)) == 0) return;
        Process((k * id + min(id, r)) * srcN, (k * id + min(id, r)) * dstN * 2, k + (id < r));
    }

    __aicore__ inline void Process(int srcOffset, int dstOffset, int lines)
    {
        int cOffset = srcN * srcM;
        LoadGather();
        if(srcN <= TILE_LENGTH)
        {
            for (int i = 0; i < lines; ++i)
            {
                CopyIn(srcOffset, srcOffset + cOffset, srcN);
                Compute(dstN * 2);
                CopyOut(dstOffset, dstN * 2);
                srcOffset += srcN;
                dstOffset += dstN * 2;
            }
            gatherQueue.FreeTensor(gatherLocal);
        }
        else
        {
            for (int i = 0; i < lines; ++i)
            {
                int remaining = dstN;
                int lineSrcOffset = srcOffset + i * srcN;
                int lineDstOffset = dstOffset + i * dstN * 2;

                while (remaining > 0)
                {
                    int currentTileSize = (remaining <= TILE_LENGTH) ? remaining : TILE_LENGTH;
                    CopyIn(lineSrcOffset, lineSrcOffset + cOffset, (currentTileSize + 7) / 8 * 8);
                    Compute(currentTileSize * 2);
                    CopyOut(lineDstOffset, currentTileSize * 2);

                    lineSrcOffset += currentTileSize;
                    lineDstOffset += currentTileSize * 2;
                    remaining -= currentTileSize;
                }
            }
            gatherQueue.FreeTensor(gatherLocal);
        }
    }

    __aicore__ inline void LoadGather()
    {
        LocalTensor gatherLocal1 = gatherQueue.AllocTensor<int32_t>();
        CreateVecIndex(gatherLocal1, (int32_t)0, TILE_LENGTH * 2);
        uint64_t mask[2] = {0xAAAAAAAAAAAAAAAA, 0};
        int32_t scalar = -1;
        Adds(gatherLocal1, gatherLocal1, scalar, mask, uint8_t(TILE_LENGTH / 32), {1, 1, 8, 8});
        Muls(gatherLocal1, gatherLocal1, (int32_t)2, TILE_LENGTH * 2);
        scalar = 4096 * 4;
        Adds(gatherLocal1, gatherLocal1, scalar, mask, uint8_t(TILE_LENGTH / 32), {1, 1, 8, 8});
        gatherQueue.EnQue(gatherLocal1);
        gatherLocal = gatherQueue.DeQue<uint32_t>();
    }

    __aicore__ inline void CopyIn(int realOffset, int mergeOffset, int length)
    {
        // debug(length);
        LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();
        DataCopy(srcLocal, cGlobal[realOffset], length);
        DataCopy(srcLocal[TILE_LENGTH], cGlobal[mergeOffset], length);
        inQueue.EnQue(srcLocal);
    }

    __aicore__ inline void Compute(int length)
    {
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        Gather(dstLocal, srcLocal, gatherLocal, 0, length);
        inQueue.FreeTensor(srcLocal);
        outQueue.EnQue(dstLocal);
    }

    __aicore__ inline void CopyOut(int offset, int length)
    {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        DataCopyExtParams intriParams{
            1,
            static_cast<uint32_t>(length * sizeof(T)),
            0,
            0,
            0
        };
        DataCopyPad(cOrgGlobal[offset], dstLocal, intriParams);
        outQueue.FreeTensor(dstLocal);
    }
};

__aicore__ inline void cgemm_cube(
    GlobalTensor<float>& aSplitGlobal,
    GlobalTensor<float>& bSplitGlobal,
    GlobalTensor<float>& bSplitNegGlobal,
    GlobalTensor<float>& cSplitGlobal,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t m, uint32_t n, uint32_t k,
    uint32_t total_batch_num,
    const TilingData& tiling_data,
    MatmulCustom<float, float, float>& opmm)
{
    int flag = 0;
    int aoffset = 2 * m * k;
    int boffset = 2 * n * k;
    int bnegoffset = n * k;
    int coffset = 2 * n * m;
    for (uint32_t batch_idx = 0; batch_idx < total_batch_num; ++batch_idx)
    {
        int m_cubes = m / 128;
        int n_cubes = n / 128;
        int mn_cubes = m_cubes * n_cubes;
        int blockDim = GetBlockNum() / 2;
        int blockIdx = GetBlockIdx() / 2;
        int base_cubes = mn_cubes / blockDim;
        int tile_cubes = mn_cubes % blockDim;
        int start = base_cubes * blockIdx + min(blockIdx, tile_cubes);
        int cur_cubes = base_cubes + (blockIdx < tile_cubes);
        int mm, nn, offsetA, offsetB, offsetC;
        uint32_t split_a_stride_k = k;
        uint32_t split_b_stride_n = n;
        uint32_t split_c_stride_n = n;
        if(flag == 0)
            CrossCoreWaitFlag(1);
        else
            CrossCoreWaitFlag(4);
        while(cur_cubes > 0) {
            int compute_cubes = min(n_cubes - start % n_cubes, cur_cubes);
            mm = min(128,int(M - (start / n_cubes) * 128));
            // mm = 32;
            nn = min(compute_cubes * 128, int(N - (start % n_cubes)*128));
            // nn = compute_cubes * 32;
            offsetA = (start / n_cubes) * 128 * split_a_stride_k;
            offsetB = (start % n_cubes) * 128;
            offsetC = (start / n_cubes) * 128 * split_c_stride_n + (start % n_cubes) * 128;
            // PRINTF("%d %d %d %d %d\n",mm,nn,offsetA,offsetB,offsetC);
            if(GetBlockIdx() % 2) {
                opmm.SetMatrix(aSplitGlobal[flag*aoffset], bSplitGlobal[flag*boffset], cSplitGlobal[flag*coffset], n, k);
                opmm.Process(offsetA, offsetB, offsetC, mm, nn, K, false);
                opmm.SetMatrix(aSplitGlobal[flag*aoffset + m * k], bSplitNegGlobal[flag*bnegoffset], cSplitGlobal[flag*coffset], n, k);
                opmm.Process(offsetA, offsetB, offsetC, mm, nn, K, true);
                // opmm.SetMatrix(aSplitGlobal[m * k], bSplitGlobal, cSplitGlobal[m * n], n, k);
                // opmm.Process(offsetA, offsetB, offsetC, mm, nn, K, false);
                // opmm.SetMatrix(aSplitGlobal, bSplitGlobal[k * n], cSplitGlobal[m * n], n, k);
                // opmm.Process(offsetA, offsetB, offsetC, mm, nn, K, true);
            } else {
                opmm.SetMatrix(aSplitGlobal[m * k + flag*aoffset], bSplitGlobal[flag*boffset], cSplitGlobal[flag*coffset+m * n], n, k);
                opmm.Process(offsetA, offsetB, offsetC, mm, nn, K, false);
                opmm.SetMatrix(aSplitGlobal[flag*aoffset], bSplitGlobal[k * n + flag*boffset], cSplitGlobal[flag*coffset+m * n], n, k);
                opmm.Process(offsetA, offsetB, offsetC, mm, nn, K, true);
                // opmm.SetMatrix(aSplitGlobal, bSplitGlobal, cSplitGlobal, n, k);
                // opmm.Process(offsetA, offsetB, offsetC, mm, nn, K, false);
                // opmm.SetMatrix(aSplitGlobal[m * k], bSplitNegGlobal, cSplitGlobal, n, k);
                // opmm.Process(offsetA, offsetB, offsetC, mm, nn, K, true);
            }
            start += compute_cubes;
            cur_cubes -= compute_cubes;
        }
        CrossCoreSetFlag<0, PIPE_FIX>(2);
        CrossCoreWaitFlag(2);
        if(flag == 0)
            CrossCoreSetFlag<2, PIPE_FIX>(3);
        else
            CrossCoreSetFlag<2, PIPE_FIX>(6);
        flag = flag ^ 1;
    }
}

__aicore__ inline void cgemm_vec(
    TBufPool<TPosition::VECCALC, 16> *pipe,
    GM_ADDR xGm, GM_ADDR yGm, GM_ADDR zGm,
    GlobalTensor<float>& aSplitGlobal,
    GlobalTensor<float>& bSplitGlobal,
    GlobalTensor<float>& bSplitNegGlobal,
    GlobalTensor<float>& cSplitGlobal,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t m, uint32_t n, uint32_t k,
    uint32_t total_batch_num,
    const int32_t x_batch_dims[8], uint32_t x_b_ndim, const uint32_t x_strides[8],
    const int32_t y_batch_dims[8], uint32_t y_b_ndim, const uint32_t y_strides[8],
    const int32_t z_batch_dims[8], uint32_t z_b_ndim, const uint32_t z_strides[8],
    SplitRealImag<float>& opsp,
    MergeRealImag<float>& opmg)
{
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueueReal, outQueueImag;
    TQue<QuePosition::VECIN, 1> gatherQueue;
    pipe->InitBuffer(inQueue, 1, 8192 * 4);
    pipe->InitBuffer(outQueueReal, 1, 8192 * 4);
    pipe->InitBuffer(outQueueImag, 1, 8192 * 4);
    pipe->InitBuffer(gatherQueue, 1, 8192 * 4);
    opsp.Init(inQueue, outQueueReal, outQueueImag);
    opmg.Init(inQueue, outQueueReal, gatherQueue);
    int flag = 0;
    int aoffset = 2 * m * k;
    int boffset = 2 * n * k;
    int bnegoffset = n * k;
    int coffset = 2 * n * m;
    uint32_t x_elements_per_batch_slice = M * K;
    uint32_t y_elements_per_batch_slice = K * N;
    uint32_t z_elements_per_batch_slice = M * N; 
    uint32_t last_z_offset_bytes = 0;
    for (uint32_t batch_idx = 0; batch_idx < total_batch_num; ++batch_idx)
    {
        uint32_t z_batch_multi_idx[8] = {0}; // Z 的多维批次索引
        uint32_t x_batch_multi_idx[8] = {0}; // X 的多维批次索引
        uint32_t y_batch_multi_idx[8] = {0}; // Y 的多维批次索引

        // 1. 获取 Z 的多维批次索引
        GetBatchMultiDimIndex_u32(batch_idx, z_batch_dims, z_b_ndim, z_batch_multi_idx);

        // 2. 推导 X 和 Y 的多维批次索引 (假设 X/Y/Z 批次维度兼容或需要广播)
        for (uint32_t i = 0; i < x_b_ndim; ++i) {
            uint32_t x_dim_size = (x_batch_dims[i] <= 0) ? 1 : static_cast<uint32_t>(x_batch_dims[i]);
             uint32_t z_corresponding_idx = (i < z_b_ndim) ? z_batch_multi_idx[i] : 0;
             x_batch_multi_idx[i] = (x_dim_size == 1) ? 0 : z_corresponding_idx;
        }
         for (uint32_t i = 0; i < y_b_ndim; ++i) {
            uint32_t y_dim_size = (y_batch_dims[i] <= 0) ? 1 : static_cast<uint32_t>(y_batch_dims[i]);
             uint32_t z_corresponding_idx = (i < z_b_ndim) ? z_batch_multi_idx[i] : 0;
             y_batch_multi_idx[i] = (y_dim_size == 1) ? 0 : z_corresponding_idx;
        }

        // 3. 计算线性元素偏移量
        uint32_t x_offset_elements = GetLinearOffsetFromMultiIndex_u32(x_batch_multi_idx, x_strides, x_b_ndim);
        uint32_t y_offset_elements = GetLinearOffsetFromMultiIndex_u32(y_batch_multi_idx, y_strides, y_b_ndim);
        uint32_t z_offset_elements = GetLinearOffsetFromMultiIndex_u32(z_batch_multi_idx, z_strides, z_b_ndim);

        // 4. 计算字节偏移量
        uint32_t x_offset_bytes = 0, y_offset_bytes = 0, z_offset_bytes = 0;
        x_offset_bytes = x_offset_elements * COMPLEX_FLOAT_SIZE_U32;
        y_offset_bytes = y_offset_elements * COMPLEX_FLOAT_SIZE_U32;
        z_offset_bytes = z_offset_elements * COMPLEX_FLOAT_SIZE_U32;

        // PRINTF("%d %d %d \n", x_offset_bytes/4,y_offset_bytes/4,z_offset_bytes/4);
        GlobalTensor<float> x_batch_tensor, y_batch_tensor, z_batch_tensor;
        x_batch_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(xGm) + x_offset_bytes / sizeof(float), x_elements_per_batch_slice * 2);
        y_batch_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(yGm) + y_offset_bytes / sizeof(float), y_elements_per_batch_slice * 2);
        z_batch_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(zGm) + last_z_offset_bytes / sizeof(float), z_elements_per_batch_slice * 2);
        // if(GetBlockIdx()==0) {DumpTensor(aSplitGlobal,0,8);DumpTensor(bSplitGlobal,0,8);}
        opsp.SetMatrix(x_batch_tensor, aSplitGlobal[flag*aoffset], K, m, k);
        opsp.SplitRealImagWork(M, GetBlockNum() * 2, GetBlockIdx());
        opsp.SetMatrix(y_batch_tensor, bSplitGlobal[flag*boffset], bSplitNegGlobal[flag*bnegoffset], N, k, n);
        opsp.SplitRealImagWork(K, GetBlockNum() * 2, GetBlockIdx());
        CrossCoreSetFlag<0, PIPE_MTE3>(0);
        CrossCoreWaitFlag(0);
        if(flag == 0)
        {
            CrossCoreSetFlag<2, PIPE_MTE3>(1);
        }
        else 
        {
            CrossCoreSetFlag<2, PIPE_MTE3>(4);
        }
        if(batch_idx) {
            if(flag == 0) {
                CrossCoreWaitFlag(6);
            } else {
                CrossCoreWaitFlag(3);
            }
            opmg.SetMatrix(cSplitGlobal[(flag^1)*coffset], z_batch_tensor, m, n, N);
            opmg.MergeRealImagWork(M, GetBlockNum() * 2, GetBlockIdx());
            CrossCoreSetFlag<0, PIPE_MTE3>(0);
            CrossCoreWaitFlag(0);
            last_z_offset_bytes = z_offset_bytes;
        }
        flag = flag ^ 1;
    }
    GlobalTensor<float> z_batch_tensor;
    z_batch_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(zGm) + last_z_offset_bytes / sizeof(float), z_elements_per_batch_slice * 2);
    opmg.SetMatrix(cSplitGlobal[(flag^1)*coffset], z_batch_tensor, m, n, N);
    if(flag == 0) {
        CrossCoreWaitFlag(6);
    } else {
        CrossCoreWaitFlag(3);
    }
    opmg.MergeRealImagWork(M, GetBlockNum() * 2, GetBlockIdx());

}

extern "C" __global__ __aicore__ void mat_mul(
    GM_ADDR xGm, GM_ADDR yGm, GM_ADDR biasGm, GM_ADDR zGm,
    GM_ADDR workspace, GM_ADDR tiling)
{
    // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // PRINTF("here\n");
    GET_TILING_DATA(tiling_data, tiling);
    int32_t  x_kernel_dims[8], y_kernel_dims[8], z_kernel_dims[8];
    uint32_t x_kernel_ndim, y_kernel_ndim, z_kernel_ndim;
    DecodeShapeInfo(tiling_data.x_shape_info, x_kernel_dims, x_kernel_ndim);
    DecodeShapeInfo(tiling_data.y_shape_info, y_kernel_dims, y_kernel_ndim);
    DecodeShapeInfo(tiling_data.z_shape_info, z_kernel_dims, z_kernel_ndim);
    uint32_t M = static_cast<uint32_t>(x_kernel_dims[x_kernel_ndim - 2]);
    uint32_t K = static_cast<uint32_t>(x_kernel_dims[x_kernel_ndim - 1]);
    uint32_t N = static_cast<uint32_t>(y_kernel_dims[y_kernel_ndim - 1]);
    // PRINTF("mnk %d %d %d\n",M,N,K);
    uint32_t x_strides[8];
    uint32_t y_strides[8];
    uint32_t z_strides[8];
    GetStrides_u32(x_kernel_dims, x_kernel_ndim, x_strides);
    GetStrides_u32(y_kernel_dims, y_kernel_ndim, y_strides);
    GetStrides_u32(z_kernel_dims, z_kernel_ndim, z_strides);
    int32_t x_batch_dims[8] = {0};
    int32_t y_batch_dims[8] = {0};
    int32_t z_batch_dims[8] = {0};
    uint32_t x_b_ndim = (x_kernel_ndim >= 2) ? (x_kernel_ndim - 2) : 0;
    uint32_t y_b_ndim = (y_kernel_ndim >= 2) ? (y_kernel_ndim - 2) : 0;
    uint32_t z_b_ndim = (z_kernel_ndim >= 2) ? (z_kernel_ndim - 2) : 0; 
    for(uint32_t i=0; i < x_b_ndim; ++i) x_batch_dims[i] = x_kernel_dims[i];
    for(uint32_t i=0; i < y_b_ndim; ++i) y_batch_dims[i] = y_kernel_dims[i];
    for(uint32_t i=0; i < z_b_ndim; ++i) z_batch_dims[i] = z_kernel_dims[i];

    uint32_t total_batch_num = 1;
    for (uint32_t i = 0; i < z_b_ndim; ++i) {
         uint32_t dim_val = static_cast<uint32_t>(z_batch_dims[i]);
         total_batch_num *= dim_val;
    }
    uint32_t z_elements_per_matrix = M * N;
    const uint32_t alignment = 128;
    uint32_t m_padded = ceil_round(M, alignment);
    uint32_t n_padded = ceil_round(N, alignment);
    uint32_t k_padded = ceil_round(K, alignment);
    uint32_t ws_offset_a_split = 0;
    uint32_t size_a_split = m_padded * k_padded * 4;
    uint32_t ws_offset_b_split = ws_offset_a_split + size_a_split;
    uint32_t size_b_split = k_padded * n_padded * 4;
    uint32_t ws_offset_b_neg = ws_offset_b_split + size_b_split;
    uint32_t size_b_neg = k_padded * n_padded * 2;
    uint32_t ws_offset_c_split = ws_offset_b_neg + size_b_neg;
    uint32_t size_c_split = m_padded * n_padded * 4;
    GlobalTensor<float> aSplitGlobal;
    GlobalTensor<float> bSplitGlobal;
    GlobalTensor<float> bSplitNegGlobal;
    GlobalTensor<float> cSplitGlobal;
    aSplitGlobal.SetGlobalBuffer((reinterpret_cast<__gm__ float *>(workspace)) + ws_offset_a_split, size_a_split);
    bSplitGlobal.SetGlobalBuffer((reinterpret_cast<__gm__ float *>(workspace)) + ws_offset_b_split, size_b_split);
    bSplitNegGlobal.SetGlobalBuffer((reinterpret_cast<__gm__ float *>(workspace)) + ws_offset_b_neg, size_b_neg);
    cSplitGlobal.SetGlobalBuffer((reinterpret_cast<__gm__ float *>(workspace)) + ws_offset_c_split, size_c_split);
#ifdef __DAV_C220_VEC__
    TPipe pipe_vec;
    TBufPool<TPosition::VECCALC, 16> tbufPool;
    pipe_vec.InitBufPool(tbufPool, 192 * 1024);
    SplitRealImag<float> opsp;
    MergeRealImag<float> opmg;
    cgemm_vec(&tbufPool, xGm, yGm, zGm,
        aSplitGlobal, bSplitGlobal, bSplitNegGlobal, cSplitGlobal,
        M, N, K, m_padded, n_padded, k_padded,
        total_batch_num, 
        x_batch_dims, x_b_ndim, x_strides,
        y_batch_dims, y_b_ndim, y_strides,
        z_batch_dims, z_b_ndim, z_strides,
        opsp, opmg);
#endif
#ifdef __DAV_C220_CUBE__
    TPipe pipe_cube;
    MatmulCustom<float, float, float> opmm;
    opmm.Init(&pipe_cube);
    cgemm_cube(aSplitGlobal, bSplitGlobal, bSplitNegGlobal, cSplitGlobal,
        M, N, K, m_padded, n_padded, k_padded,
        total_batch_num, tiling_data,
        opmm);
#endif
#ifdef __DAV_C220_VEC__
    if(tiling_data.if_bias == 0) return;
    CrossCoreSetFlag<0, PIPE_MTE3>(0);
    CrossCoreWaitFlag(0);
    tbufPool.Reset();
    if (biasGm != nullptr) {
        int32_t bias_kernel_dims[8];
        uint32_t bias_kernel_ndim;
        uint32_t bias_total_elements = DecodeShapeInfo(tiling_data.bias_shape_info, bias_kernel_dims, bias_kernel_ndim);

        if (bias_total_elements > 0) {
            int32_t bias_type = -1;
            if (CheckBroadcastCompatibilityAndGetType_u32(z_kernel_dims, z_kernel_ndim, bias_kernel_dims, bias_kernel_ndim, M, N, bias_type))
            {
                uint32_t z_strides[8];
                uint32_t bias_strides[8];
                GetStrides_u32(z_kernel_dims, z_kernel_ndim, z_strides);
                GetStrides_u32(bias_kernel_dims, bias_kernel_ndim, bias_strides);

                int32_t z_batch_dims[8] = {0};
                int32_t bias_batch_dims[8] = {0};
                uint32_t z_b_ndim = (z_kernel_ndim >= 2) ? (z_kernel_ndim - 2) : 0;
                uint32_t bias_b_ndim = (bias_kernel_ndim >= 2) ? (bias_kernel_ndim - 2) : 0;
                 if (bias_kernel_ndim == 1) bias_b_ndim = (bias_kernel_dims[0] == 1) ? 0 : 1;
                 if (bias_kernel_ndim == 0) bias_b_ndim = 0;

                for(uint32_t i=0; i < z_b_ndim; ++i) z_batch_dims[i] = z_kernel_dims[i];
                for(uint32_t i=0; i < bias_b_ndim; ++i) bias_batch_dims[i] = bias_kernel_dims[i];

                BiasAdd biasAddOp;
                // 确保 tbufPool 有效
                biasAddOp.Init(&tbufPool);

                uint32_t z_elements_per_matrix = M * N; // 32位计算
                // 简单溢出检查
                // PRINTF("type %d\n", bias_type);
                for (uint32_t batch_idx = 0; batch_idx < total_batch_num; ++batch_idx)
                {
                    // Z 偏移量 (字节)
                    uint32_t z_offset_bytes = batch_idx * z_elements_per_matrix * COMPLEX_FLOAT_SIZE_U32;

                    // Bias 偏移量 (字节)
                    uint32_t z_batch_multi_idx[8] = {0};
                    uint32_t bias_batch_multi_idx[8] = {0};
                    uint32_t bias_offset_elements = 0;

                    GetBatchMultiDimIndex_u32(batch_idx, z_batch_dims, z_b_ndim, z_batch_multi_idx);

                    for (uint32_t i = 0; i < bias_b_ndim; ++i) {
                        uint32_t bias_dim_size = (bias_batch_dims[i] <= 0) ? 1 : static_cast<uint32_t>(bias_batch_dims[i]);
                        bias_batch_multi_idx[i] = (bias_dim_size == 1) ? 0 : z_batch_multi_idx[i];
                    }

                    bias_offset_elements = GetLinearOffsetFromMultiIndex_u32(bias_batch_multi_idx, bias_strides, bias_b_ndim);

                    if (bias_offset_elements == UINT32_MAX) { /* Handle offset error */ continue; }
                    // 检查字节偏移量溢出
                    uint32_t bias_offset_bytes = 0;
                    if (bias_offset_elements > UINT32_MAX / COMPLEX_FLOAT_SIZE_U32) { /* Handle overflow */ continue; }
                     bias_offset_bytes = bias_offset_elements * COMPLEX_FLOAT_SIZE_U32;

                    // 设置 GlobalTensor
                    GlobalTensor<float> z_batch_tensor;
                    GlobalTensor<float> bias_batch_tensor;

                    z_batch_tensor.SetGlobalBuffer(
                        reinterpret_cast<__gm__ float *>(zGm) + z_offset_bytes / sizeof(float),
                        z_elements_per_matrix * 2
                    );

                    uint32_t bias_M_local = (bias_kernel_ndim >= 2) ? static_cast<uint32_t>(bias_kernel_dims[bias_kernel_ndim - 2]) : 1;
                    uint32_t bias_N_local = (bias_kernel_ndim >= 1) ? static_cast<uint32_t>(bias_kernel_dims[bias_kernel_ndim - 1]) : 1;
                    if (bias_kernel_ndim <= 1) bias_M_local = 1;
                    uint32_t bias_elements_per_slice = bias_M_local * bias_N_local;
                     if (bias_M_local > 0 && bias_N_local > 0 && bias_M_local > UINT32_MAX / bias_N_local) { /* Handle overflow */ }


                    bias_batch_tensor.SetGlobalBuffer(
                        reinterpret_cast<__gm__ float *>(biasGm) + bias_offset_bytes / sizeof(float),
                        bias_elements_per_slice * 2
                    );
                    // PRINTF("offset %d %d \n",z_offset_bytes/4, bias_offset_bytes/4);
                    // 调用 BiasAdd
                    // DumpTensor(bias_batch_tensor,1,8);
                    biasAddOp.SetMatrix(bias_batch_tensor, z_batch_tensor, M, N, bias_type);
                    biasAddOp.Process();
                }
            } 
        }
    }
#endif
}

