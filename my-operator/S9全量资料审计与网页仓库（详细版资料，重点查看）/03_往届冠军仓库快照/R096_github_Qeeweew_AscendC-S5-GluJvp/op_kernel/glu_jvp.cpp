#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM_OUT = 2;

__aicore__ inline uint32_t AlignUP(uint32_t x, uint32_t p)
{
    return (x + (p - 1)) & ~(p - 1);
}

template<typename T>
struct GluJvpConfig;

template<>
struct GluJvpConfig<float> {
    static constexpr uint32_t MAX_TILE_LENGTH = 4096;
    static constexpr uint32_t OFFSET_LENGTH = MAX_TILE_LENGTH + 8;
    static constexpr bool NEEDS_FP32_COMPUTE = false;
};

template<>
struct GluJvpConfig<half> {
    static constexpr uint32_t MAX_TILE_LENGTH = 2048;
    static constexpr uint32_t OFFSET_LENGTH = MAX_TILE_LENGTH + 8;
    static constexpr bool NEEDS_FP32_COMPUTE = true;
};

template<>
struct GluJvpConfig<bfloat16_t> {
    static constexpr uint32_t MAX_TILE_LENGTH = 2048;
    static constexpr uint32_t OFFSET_LENGTH = MAX_TILE_LENGTH + 8;
    static constexpr bool NEEDS_FP32_COMPUTE = true;
};

// --- 辅助计算结构体，用于处理不同数据类型的计算逻辑 ---

// 模板声明
template <typename T, bool UseFp32Compute>
struct GluJvpComputer;

// 特化版本 1: 用于 float 类型，直接在原生类型上计算
template <typename T>
struct GluJvpComputer<T, false> {
    __aicore__ static inline void Compute(
        AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM>& inQueueInput,
        AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM>& inQueueV,
        AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM_OUT>& outQueueJvpOut,
        AscendC::TBuf<AscendC::TPosition::VECIN>& tmpWorkspace,
        uint32_t currentTileLength)
    {

        constexpr uint32_t offset_input = GluJvpConfig<T>::MAX_TILE_LENGTH;
        constexpr uint32_t offset_length = GluJvpConfig<T>::OFFSET_LENGTH;

        // --- Stage 1: Dequeue Input and compute S and GluOut ---
        // 仅依赖 a, b 的计算可以提前进行

        // Dequeue合并后的输入张量 (A, B)
        AscendC::LocalTensor<T> inputLocal = inQueueInput.DeQue<T>();
        AscendC::LocalTensor<T> aLocal = inputLocal;
        AscendC::LocalTensor<T> bLocal = inputLocal[offset_input];

        // 分配临时和输出张量
        AscendC::LocalTensor<T> tmpBuffer = tmpWorkspace.Get<T>();
        AscendC::LocalTensor<T> sLocal      = tmpBuffer;
        AscendC::LocalTensor<T> gluOutLocal = tmpBuffer[offset_length];
        AscendC::LocalTensor<T> term2Local  = tmpBuffer[offset_length * 2];
        AscendC::LocalTensor<T> tmpReusable = tmpBuffer[offset_length * 3];
        AscendC::LocalTensor<T> jvpOutLocal = outQueueJvpOut.AllocTensor<T>();

        // 计算 s = sigmoid(-b)
        AscendC::Muls(tmpReusable, bLocal, (T)-1.0f, currentTileLength);
        AscendC::Exp(tmpReusable, tmpReusable, currentTileLength);
        AscendC::Adds(tmpReusable, tmpReusable, (T)1.0f, currentTileLength);
        AscendC::Duplicate(sLocal, (T)1.0f, currentTileLength);
        AscendC::Div(sLocal, sLocal, tmpReusable, currentTileLength);

        // 计算 glu_out = a * s
        AscendC::Mul(gluOutLocal, aLocal, sLocal, currentTileLength);

        // --- Stage 2: Dequeue V and compute the final JVP ---
        // 此处会等待 V (VA, VB) 的数据到达，但此时 Stage 1 的计算已经与 V 的搬运并行执行

        // Dequeue合并后的 V 张量 (VA, VB)
        AscendC::LocalTensor<T> vLocal = inQueueV.DeQue<T>();
        AscendC::LocalTensor<T> vaLocal = vLocal;
        AscendC::LocalTensor<T> vbLocal = vLocal[offset_input];

        // 计算 jvp_out = va * s
        AscendC::Mul(jvpOutLocal, vaLocal, sLocal, currentTileLength);

        // 计算 term2 = (vb - s * vb) * glu_out
        AscendC::Mul(tmpReusable, sLocal, vbLocal, currentTileLength);
        AscendC::Sub(term2Local, vbLocal, tmpReusable, currentTileLength);
        AscendC::Mul(term2Local, term2Local, gluOutLocal, currentTileLength);

        // 计算 jvp_out = jvp_out + term2
        AscendC::Add(jvpOutLocal, jvpOutLocal, term2Local, currentTileLength);

        // Enqueue输出并释放输入
        outQueueJvpOut.EnQue(jvpOutLocal);
        inQueueInput.FreeTensor(inputLocal);
        inQueueV.FreeTensor(vLocal);
    }
};

// 特化版本 2: 用于 half 和 bfloat16_t 类型，上转型到 float32 进行计算 (已优化流水)
template <typename T>
struct GluJvpComputer<T, true> {
    __aicore__ static inline void Compute(
        AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM>& inQueueInput,
        AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM>& inQueueV,
        AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM_OUT>& outQueueJvpOut,
        AscendC::TBuf<AscendC::TPosition::VECIN>& tmpWorkspace,
        uint32_t currentTileLength)
    {

        constexpr uint32_t offset_input = GluJvpConfig<T>::MAX_TILE_LENGTH;
        constexpr uint32_t offset_length = GluJvpConfig<T>::OFFSET_LENGTH;

        // --- Stage 1: Dequeue Input, cast to FP32, and compute S and GluOut ---

        // Dequeue T 类型的合并输入 (A, B)
        AscendC::LocalTensor<T> inputLocalIn = inQueueInput.DeQue<T>();
        AscendC::LocalTensor<T> aLocalIn = inputLocalIn;
        AscendC::LocalTensor<T> bLocalIn = inputLocalIn[offset_input];

        // 准备 float32 版本的张量
        AscendC::LocalTensor<float> tmpBufferFp32 = tmpWorkspace.Get<float>();
        AscendC::LocalTensor<float> aLocalFp32      = tmpBufferFp32;
        AscendC::LocalTensor<float> bLocalFp32      = tmpBufferFp32[offset_length];
        // 临时计算和输出也使用FP32
        AscendC::LocalTensor<float> sLocalFp32      = tmpBufferFp32[offset_length * 4]; // 预留va,vb的空间
        AscendC::LocalTensor<float> gluOutLocalFp32 = tmpBufferFp32[offset_length * 5];
        AscendC::LocalTensor<float> term2LocalFp32  = tmpBufferFp32[offset_length * 6];
        AscendC::LocalTensor<float> tmpReusableFp32 = tmpBufferFp32[offset_length * 7];
        AscendC::LocalTensor<float> jvpOutLocalFp32 = tmpBufferFp32[offset_length * 8];

        // 执行 Cast: T -> float for A, B
        AscendC::Cast(aLocalFp32, aLocalIn, AscendC::RoundMode::CAST_NONE, currentTileLength);
        AscendC::Cast(bLocalFp32, bLocalIn, AscendC::RoundMode::CAST_NONE, currentTileLength);

        // 计算 s = sigmoid(-b) in FP32
        AscendC::Muls(tmpReusableFp32, bLocalFp32, -1.0f, currentTileLength);
        AscendC::Exp(tmpReusableFp32, tmpReusableFp32, currentTileLength);
        AscendC::Adds(tmpReusableFp32, tmpReusableFp32, 1.0f, currentTileLength);
        AscendC::Duplicate(sLocalFp32, 1.0f, currentTileLength);
        AscendC::Div(sLocalFp32, sLocalFp32, tmpReusableFp32, currentTileLength);

        // 计算 glu_out = a * s in FP32
        AscendC::Mul(gluOutLocalFp32, aLocalFp32, sLocalFp32, currentTileLength);

        // --- Stage 2: Dequeue V, cast to FP32, and compute the final JVP ---

        // Dequeue T 类型的 V 张量 (VA, VB)
        AscendC::LocalTensor<T> vLocalIn = inQueueV.DeQue<T>();
        AscendC::LocalTensor<T> vaLocalIn = vLocalIn;
        AscendC::LocalTensor<T> vbLocalIn = vLocalIn[offset_input];

        // 准备 float32 版本的 VA, VB 张量
        AscendC::LocalTensor<float> vaLocalFp32 = tmpBufferFp32[offset_length * 2];
        AscendC::LocalTensor<float> vbLocalFp32 = tmpBufferFp32[offset_length * 3];

        // 执行 Cast: T -> float for VA, VB
        AscendC::Cast(vaLocalFp32, vaLocalIn, AscendC::RoundMode::CAST_NONE, currentTileLength);
        AscendC::Cast(vbLocalFp32, vbLocalIn, AscendC::RoundMode::CAST_NONE, currentTileLength);

        // 计算 jvp_out = va * s
        AscendC::Mul(jvpOutLocalFp32, vaLocalFp32, sLocalFp32, currentTileLength);

        // 计算 term2 = (vb - s * vb) * glu_out
        AscendC::Mul(tmpReusableFp32, sLocalFp32, vbLocalFp32, currentTileLength);
        AscendC::Sub(term2LocalFp32, vbLocalFp32, tmpReusableFp32, currentTileLength);
        AscendC::Mul(term2LocalFp32, term2LocalFp32, gluOutLocalFp32, currentTileLength);

        // 计算 jvp_out = jvp_out + term2
        AscendC::Add(jvpOutLocalFp32, jvpOutLocalFp32, term2LocalFp32, currentTileLength);

        // --- 将 float 结果转换回 T ---
        AscendC::LocalTensor<T> jvpOutLocalOut = outQueueJvpOut.AllocTensor<T>();
        AscendC::Cast(jvpOutLocalOut, jvpOutLocalFp32, AscendC::RoundMode::CAST_RINT, currentTileLength);

        // Enqueue输出并释放输入
        outQueueJvpOut.EnQue(jvpOutLocalOut);
        inQueueInput.FreeTensor(inputLocalIn);
        inQueueV.FreeTensor(vLocalIn);
    }
};

// 主模板类
template<typename T, bool LargeBatch>
class KernelGluJvp {
public:
    static constexpr uint32_t ALIGNMENT_BYTES = 256;
    static constexpr uint32_t ALIGNMENT_IN_ELEMENTS = ALIGNMENT_BYTES / sizeof(T);
    static constexpr uint32_t MAX_TILE_LENGTH = GluJvpConfig<T>::MAX_TILE_LENGTH;
    static constexpr uint32_t OFFSET_LENGTH = GluJvpConfig<T>::OFFSET_LENGTH;
    static constexpr bool NEEDS_FP32_COMPUTE = GluJvpConfig<T>::NEEDS_FP32_COMPUTE;

    __aicore__ inline KernelGluJvp() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR v, GM_ADDR jvp_out,
                                AscendC::TPipe* pipe, uint32_t batch_size, uint32_t length) {
        this->pipe = pipe;
        this->batch_size = batch_size;
        this->length = length;
        this->halfLength = length / 2;

        // Tiling策略 (保持不变)
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (LargeBatch) {
            this->coreOffset = 0;
            this->coreLength = this->halfLength;
        } else {
            uint32_t totalAlignedBlocks = this->halfLength / ALIGNMENT_IN_ELEMENTS;
            uint32_t blocksPerCore = totalAlignedBlocks / blockNum;
            uint32_t blocksRemainder = totalAlignedBlocks % blockNum;
            uint32_t startBlock = blockIdx * blocksPerCore + (blockIdx < blocksRemainder ? blockIdx : blocksRemainder);
            uint32_t numBlocks = blocksPerCore + (blockIdx < blocksRemainder ? 1 : 0);
            this->coreOffset = startBlock * ALIGNMENT_IN_ELEMENTS;
            this->coreLength = numBlocks * ALIGNMENT_IN_ELEMENTS;
            if (blockIdx == blockNum - 1) {
                this->coreLength += this->halfLength % ALIGNMENT_IN_ELEMENTS;
            }
        }
        this->blockNum = blockNum;

        jvpOutGm.SetGlobalBuffer((__gm__ T*)jvp_out, batch_size * this->halfLength);
        inputGm.SetGlobalBuffer((__gm__ T*)input, batch_size * this->length);
        vGm.SetGlobalBuffer((__gm__ T*)v, batch_size * this->length);

        // 初始化队列和工作空间 (保持不变)
        pipe->InitBuffer(inQueueInput, BUFFER_NUM, 2 * MAX_TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(inQueueV, BUFFER_NUM, 2 * MAX_TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(outQueueJvpOut, BUFFER_NUM_OUT, MAX_TILE_LENGTH * sizeof(T));

        if constexpr (NEEDS_FP32_COMPUTE) {
            pipe->InitBuffer(tmpWorkspace, 9 * OFFSET_LENGTH * sizeof(float));
        } else {
            pipe->InitBuffer(tmpWorkspace, 4 * OFFSET_LENGTH * sizeof(T));
        }
    }

    __aicore__ inline void ProcessLoop(uint32_t b) {
        uint32_t baseInputOffset = b * this->length + this->coreOffset;
        uint32_t baseJvpOutOffset = b * this->halfLength + this->coreOffset;
        for (uint32_t offsetInCore = 0; offsetInCore < this->coreLength; offsetInCore += MAX_TILE_LENGTH) {
            uint32_t currentTileLength = MAX_TILE_LENGTH < this->coreLength - offsetInCore ? MAX_TILE_LENGTH : this->coreLength - offsetInCore;
            CopyIn(baseInputOffset, offsetInCore, currentTileLength);
            Compute(currentTileLength);
            CopyOut(baseJvpOutOffset, offsetInCore, currentTileLength);
        }
    }

    __aicore__ inline void Process() {
        if (LargeBatch) {
            for (uint32_t b = AscendC::GetBlockIdx(); b < this->batch_size; b += this->blockNum) {
                ProcessLoop(b);
            }
        } else {
            for (uint32_t b = 0; b < this->batch_size; ++b) {
                ProcessLoop(b);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t baseInputOffset, uint32_t offsetInCore, uint32_t currentTileLength) {
        AscendC::LocalTensor<T> inputLocal = inQueueInput.AllocTensor<T>();
        AscendC::LocalTensor<T> vLocal = inQueueV.AllocTensor<T>();

        uint32_t gmOffset = baseInputOffset + offsetInCore;

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 2;
        copyParams.blockLen = currentTileLength * sizeof(T);
        copyParams.srcStride = (this->halfLength - currentTileLength) * sizeof(T);
        copyParams.dstStride = (MAX_TILE_LENGTH  * sizeof(T) - AlignUP(currentTileLength * sizeof(T), 32)) / 32;

        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        AscendC::DataCopyPad(inputLocal, inputGm[gmOffset], copyParams, padParams);
        AscendC::DataCopyPad(vLocal, vGm[gmOffset], copyParams, padParams);

        inQueueInput.EnQue(inputLocal);
        inQueueV.EnQue(vLocal);
    }

    __aicore__ inline void Compute(uint32_t currentTileLength) {
        GluJvpComputer<T, NEEDS_FP32_COMPUTE>::Compute(
            inQueueInput, inQueueV,
            outQueueJvpOut, tmpWorkspace, currentTileLength
        );
    }

    __aicore__ inline void CopyOut(uint32_t baseJvpOutOffset, uint32_t offsetInCore, uint32_t currentTileLength) {
        AscendC::LocalTensor<T> jvpOutLocal = outQueueJvpOut.DeQue<T>();
        uint32_t gmOffset = baseJvpOutOffset + offsetInCore;
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = currentTileLength * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(jvpOutGm[gmOffset], jvpOutLocal, copyParams);
        outQueueJvpOut.FreeTensor(jvpOutLocal);
    }

private:
    AscendC::TPipe* pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueInput, inQueueV;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM_OUT> outQueueJvpOut;
    AscendC::TBuf<AscendC::TPosition::VECIN> tmpWorkspace;
    AscendC::GlobalTensor<T> inputGm, vGm, jvpOutGm;
    uint32_t batch_size, length, halfLength;
    uint32_t coreOffset, coreLength;
    uint32_t blockNum;
};

extern "C" __global__ __aicore__ void glu_jvp(GM_ADDR glu_out, GM_ADDR input, GM_ADDR v, GM_ADDR jvp_out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(1)) {
        KernelGluJvp<DTYPE_INPUT, true> op;
        op.Init(
            input, v, jvp_out, &pipe,
            tiling_data.batch_size, tiling_data.length
        );
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelGluJvp<DTYPE_INPUT, false> op;
        op.Init(
            input, v, jvp_out, &pipe,
            tiling_data.batch_size, tiling_data.length
        );
        op.Process();
    }
}
