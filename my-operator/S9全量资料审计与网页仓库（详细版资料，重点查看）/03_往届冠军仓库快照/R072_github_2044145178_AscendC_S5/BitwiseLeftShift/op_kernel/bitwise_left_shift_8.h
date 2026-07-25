#include "kernel_operator.h"
using namespace AscendC;

constexpr uint16_t BufferNum_8 = 1;
template <typename T>
class KernelBitwiseLeftShift_8 {
public:
    __aicore__ inline KernelBitwiseLeftShift_8() {}
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

        uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(190 / 5) * 1024 / BufferNum_8);
        this->n_elements_per_iter = spaceSize / sizeof(T);
        this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;

        pipe->InitBuffer(inputBuf, BufferNum_8, spaceSize + 32);
        pipe->InitBuffer(otherBuf, BufferNum_8, spaceSize + 32);
        pipe->InitBuffer(outBuf, BufferNum_8, spaceSize + 32);

        pipe->InitBuffer(input_16Buf, spaceSize * 2 + 32);
        pipe->InitBuffer(other_16Buf, spaceSize * 2 + 32);
        pipe->InitBuffer(indexBuf, spaceSize * 4 + 32);

        pipe->InitBuffer(powerBuf, 64);
        this->powerLocal = powerBuf.Get<int16_t>();
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

        LocalTensor<half> inputLocal_fp16 = input_16Buf.Get<half>();
        LocalTensor<half> otherLocal_fp16 = other_16Buf.Get<half>();
        LocalTensor<int16_t> inputLocal_16 = input_16Buf.Get<int16_t>();
        LocalTensor<int16_t> otherLocal_16 = other_16Buf.Get<int16_t>();

        Cast(inputLocal_fp16, inputLocal, RoundMode::CAST_NONE, iterSize);
        Cast(inputLocal_16, inputLocal_fp16, RoundMode::CAST_RINT, iterSize);

        Cast(otherLocal_fp16, otherLocal, RoundMode::CAST_NONE, iterSize);
        Cast(otherLocal_16, otherLocal_fp16, RoundMode::CAST_RINT, iterSize);

        Mins(otherLocal_16, otherLocal_16, (int16_t)(sizeof(T) * 8), iterSize);

        LocalTensor<float> indexLocal_fp32 = indexBuf.Get<float>();
        Cast(indexLocal_fp32, otherLocal_16, RoundMode::CAST_NONE, iterSize);
        LocalTensor<int32_t> indexLocal_int32 = indexBuf.Get<int32_t>();
        Cast(indexLocal_int32, indexLocal_fp32, RoundMode::CAST_RINT, iterSize);

        LocalTensor<int16_t> tmpLocal_16 = other_16Buf.Get<int16_t>();
        ShiftLeft(indexLocal_int32, indexLocal_int32, 1, iterSize);

        Gather(tmpLocal_16, powerLocal, indexLocal_int32.template ReinterpretCast<uint32_t>(), 0, iterSize);
        Mul(tmpLocal_16, inputLocal_16, tmpLocal_16, iterSize);

        // 直接截断
        LocalTensor<int16_t> tmp_16 = indexBuf.Get<int16_t>();
        Duplicate(tmp_16, (int16_t)0xFF, iterSize);
        And(tmpLocal_16, tmpLocal_16, tmp_16, iterSize);
        ShiftLeft(tmpLocal_16, tmpLocal_16, (int16_t)8, iterSize);
        ShiftRight(tmpLocal_16, tmpLocal_16, (int16_t)8, iterSize);

        LocalTensor<half> outLocal_fp16 = input_16Buf.Get<half>();
        Cast(outLocal_fp16, tmpLocal_16, RoundMode::CAST_TRUNC, iterSize);
        Cast(outLocal, outLocal_fp16, RoundMode::CAST_TRUNC, iterSize);
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
        uint32_t blockBeginIndex = 0;
        // printf("Block %d, Size: %d, Small Loop Times: %d, Elements per Iter: %d\n", GetBlockIdx(), size, smallLoopTimes, n_elements_per_iter);
        for (uint32_t i = 0; i < smallLoopTimes - 1; ++i) {
            // printf("Block %d, Iteration %d, Begin Index: %d\n", GetBlockIdx(), i, blockBeginIndex);
            Process_iter(blockBeginIndex, n_elements_per_iter);
            blockBeginIndex += n_elements_per_iter;
        }
        uint32_t iterSize = min(n_elements_per_iter, size - blockBeginIndex);
        Process_iter(blockBeginIndex, iterSize);
    }

private:
    TPipe *pipe;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> otherGm;
    GlobalTensor<T> outGm;

    TQue<QuePosition::VECIN, BufferNum_8> inputBuf;
    TQue<QuePosition::VECIN, BufferNum_8> otherBuf;
    TQue<QuePosition::VECOUT, BufferNum_8> outBuf;

    TBuf<QuePosition::VECCALC> input_16Buf;
    TBuf<QuePosition::VECCALC> other_16Buf;
    TBuf<QuePosition::VECCALC> powerBuf;
    TBuf<QuePosition::VECCALC> indexBuf;
    LocalTensor<int16_t> powerLocal;
    uint32_t size;
    uint32_t smallLoopTimes;
    uint32_t n_elements_per_iter;
};
