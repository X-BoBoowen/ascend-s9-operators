#include "kernel_operator.h"
using namespace AscendC;

constexpr uint16_t BufferNum_16 = 1;
template <typename T>
class KernelBitwiseLeftShift_16 {
public:
    __aicore__ inline KernelBitwiseLeftShift_16() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out, uint32_t smallSize, uint32_t incSize, uint16_t formerNum, TPipe *pipeIn) {
        this->pipe = pipeIn;

        uint32_t beginIndex = 0;
        if (GetBlockIdx() < formerNum) {
            this->size = smallSize + incSize;
            beginIndex = this->size * GetBlockIdx();
        } else {
            this->size = smallSize;
            beginIndex = this->size * GetBlockIdx() + formerNum * incSize;
        }

        inputGm.SetGlobalBuffer((__gm__ T *)input + beginIndex, size);
        otherGm.SetGlobalBuffer((__gm__ T *)other + beginIndex, size);
        outGm.SetGlobalBuffer((__gm__ T *)out + beginIndex, size);

        uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(180 / 5) * 1024 / BufferNum_16);
        this->n_elements_per_iter = spaceSize / sizeof(T);
        this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;

        pipe->InitBuffer(inputBuf, BufferNum_16, spaceSize + 32);
        pipe->InitBuffer(otherBuf, BufferNum_16, spaceSize + 32);
        pipe->InitBuffer(outBuf, BufferNum_16, spaceSize + 32);

        pipe->InitBuffer(indexBuf, spaceSize * 2 + 32);

        pipe->InitBuffer(powerBuf, 512);
        this->powerLocal = powerBuf.Get<T>();
    }
    __aicore__ inline void Process_iter(uint32_t beginIndex, uint32_t iterSize) {
        LocalTensor<T> inputLocal = inputBuf.AllocTensor<T>();
        LocalTensor<T> otherLocal = otherBuf.AllocTensor<T>();

        // 从全局内存读取数据到本地内存
        DataCopy(inputLocal, inputGm[beginIndex], iterSize);
        DataCopy(otherLocal, otherGm[beginIndex], iterSize);

        inputBuf.EnQue<T>(inputLocal);
        otherBuf.EnQue<T>(otherLocal);
        inputLocal = inputBuf.DeQue<T>();
        otherLocal = otherBuf.DeQue<T>();

        LocalTensor<T> outLocal = outBuf.AllocTensor<T>();

        Mins(otherLocal, otherLocal, (int16_t)(sizeof(T) * 8), iterSize);
        // for (int i = 0; i < 8; i++) {
        //     printf("otherLocal[%d]: %d\n", i, otherLocal.GetValue(i));
        // }
        if constexpr (std::is_same<T, int32_t>::value) {
            ShiftLeft(otherLocal, otherLocal, (int16_t)2, iterSize);
        } else if constexpr (std::is_same<T, int16_t>::value) {
            ShiftLeft(otherLocal, otherLocal, (int16_t)1, iterSize);
        }
        // for (int i = 0; i < 8; i++) {
        //     printf("otherLocal[%d]: %d\n", i, otherLocal.GetValue(i));
        // }

        LocalTensor<float> indexLocal_fp32 = indexBuf.Get<float>();
        Cast(indexLocal_fp32, otherLocal, RoundMode::CAST_NONE, iterSize);
        LocalTensor<int32_t> indexLocal_int32 = indexBuf.Get<int32_t>();
        Cast(indexLocal_int32, indexLocal_fp32, RoundMode::CAST_RINT, iterSize);

        Gather(outLocal, powerLocal, indexLocal_int32.template ReinterpretCast<uint32_t>(), 0, iterSize);
        Mul(outLocal, inputLocal, outLocal, iterSize);

        inputBuf.FreeTensor<T>(inputLocal);
        otherBuf.FreeTensor<T>(otherLocal);

        outBuf.EnQue<T>(outLocal);
        outLocal = outBuf.DeQue<T>();
        // 将结果写回全局内存
        DataCopy(outGm[beginIndex], outLocal, iterSize);
        outBuf.FreeTensor<T>(outLocal);
    }
    __aicore__ inline void Process() {
        int n = sizeof(T) * 8;
        for (int i = 0; i < n; i++) {
            powerLocal.SetValue(i, 1 << i);
        }
        powerLocal.SetValue(n, 0);
        // for (int i = 0; i < n + 1; i++) {
        //     printf("powerLocal[%d]: %d\n", i, powerLocal.GetValue(i));
        // }
        uint32_t blockBeginIndex = 0;
        // printf("Block %d, Size: %d, Small Loop Times: %d, Elements per Iter: %d\n", GetBlockIdx(), size, smallLoopTimes, n_elements_per_iter);
        for (uint32_t i = 0; i < smallLoopTimes; ++i) {
            // printf("Block %d, Iteration %d, Begin Index: %d\n", GetBlockIdx(), i, blockBeginIndex);

            uint32_t iterSize = min(n_elements_per_iter, size - blockBeginIndex);
            Process_iter(blockBeginIndex, iterSize);
            blockBeginIndex += iterSize;
        }
    }

private:
    TPipe *pipe;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> otherGm;
    GlobalTensor<T> outGm;

    TQue<QuePosition::VECIN, BufferNum_16> inputBuf;
    TQue<QuePosition::VECIN, BufferNum_16> otherBuf;
    TQue<QuePosition::VECOUT, BufferNum_16> outBuf;

    TBuf<QuePosition::VECCALC> powerBuf;
    TBuf<QuePosition::VECCALC> indexBuf;
    LocalTensor<T> powerLocal;
    uint32_t size;
    uint32_t smallLoopTimes;
    uint32_t n_elements_per_iter;
};
