#include "kernel_operator.h"
using namespace AscendC;

constexpr uint16_t BufferNum_32_Big = 1;
template <typename T>
class KernelBitwiseLeftShift_32_big {
public:
    __aicore__ inline KernelBitwiseLeftShift_32_big() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out, uint32_t smallSize, uint32_t incSize, uint16_t formerNum, uint32_t totalSize, TPipe *pipeIn) {
        this->pipe = pipeIn;

        // uint32_t beginIndex = 0;
        // if (GetBlockIdx() < formerNum) {
        //     this->size = smallSize + incSize;
        //     beginIndex = this->size * GetBlockIdx();
        // } else {
        //     this->size = smallSize;
        //     beginIndex = this->size * GetBlockIdx() + formerNum * incSize;
        // }

        inputGm.SetGlobalBuffer((__gm__ T *)input, totalSize);
        otherGm.SetGlobalBuffer((__gm__ T *)other, totalSize);
        outGm.SetGlobalBuffer((__gm__ T *)out, totalSize);

        uint32_t spaceSize = min((uint32_t)(totalSize * sizeof(T)), (uint32_t)(60) * 1024 / BufferNum_32_Big);

        this->n_elements_per_iter = spaceSize / sizeof(T);
        uint32_t total_block_size = (totalSize + n_elements_per_iter - 1) / n_elements_per_iter;
        // 看L2了，有可能不会报异常
        this->smallLoopTimes = (total_block_size + GetBlockNum() - 1) / GetBlockNum();

        pipe->InitBuffer(inputBuf, BufferNum_32_Big, spaceSize);
        pipe->InitBuffer(otherBuf, BufferNum_32_Big, spaceSize + 32);
        pipe->InitBuffer(outBuf, BufferNum_32_Big, spaceSize + 32);
        pipe->InitBuffer(powerBuf, 256);
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
        constexpr T n = sizeof(T) * 8;

        Mins(otherLocal, otherLocal, n, iterSize);
        ShiftLeft(otherLocal, otherLocal, 2, iterSize);
        Gather(outLocal, powerLocal, otherLocal.template ReinterpretCast<uint32_t>(), 0, iterSize);
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
        constexpr int n = sizeof(T) * 8;
        for (int i = 0; i < n; i += 4) {
            powerLocal.SetValue(i, 1 << i);
            powerLocal.SetValue(i + 1, 1 << (i + 1));
            powerLocal.SetValue(i + 2, 1 << (i + 2));
            powerLocal.SetValue(i + 3, 1 << (i + 3));
        }
        powerLocal.SetValue(n, 0);
        uint32_t blockBeginIndex = GetBlockIdx() * n_elements_per_iter;
        for (uint32_t i = 0; i < smallLoopTimes; ++i) {
            Process_iter(blockBeginIndex, n_elements_per_iter);
            blockBeginIndex += n_elements_per_iter * GetBlockNum();
        }
        // uint32_t iterSize = min(n_elements_per_iter, size - blockBeginIndex);
        // Process_iter(blockBeginIndex, iterSize);
    }

private:
    TPipe *pipe;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> otherGm;
    GlobalTensor<T> outGm;

    TQue<QuePosition::VECIN, BufferNum_32_Big> inputBuf;
    TQue<QuePosition::VECIN, BufferNum_32_Big> otherBuf;
    TQue<QuePosition::VECOUT, BufferNum_32_Big> outBuf;

    TBuf<QuePosition::VECCALC> powerBuf;
    LocalTensor<T> powerLocal;
    uint32_t size;
    uint32_t smallLoopTimes;
    uint32_t n_elements_per_iter;
};
