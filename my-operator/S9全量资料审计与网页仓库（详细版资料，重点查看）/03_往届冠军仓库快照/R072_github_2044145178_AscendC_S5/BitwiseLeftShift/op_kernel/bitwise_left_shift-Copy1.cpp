#include "kernel_operator.h"
#include "bitwise_left_shift_8.h"
#include "bitwise_left_shift_16.h"
#include "bitwise_left_shift_64_spec.h"
#include "bitwise_left_shift_sca.h"

using namespace AscendC;

// int16、 int32_t类型的左移操作
constexpr uint16_t BufferNum_k = 1;
template <typename T>
class KernelBitwiseLeftShift_32 {
public:
    __aicore__ inline KernelBitwiseLeftShift_32() {}
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

        uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(190 / 3) * 1024 / BufferNum_k);
        this->n_elements_per_iter = spaceSize / sizeof(T);
        this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;

        pipe->InitBuffer(inputBuf, BufferNum_k, spaceSize + 32);
        pipe->InitBuffer(otherBuf, BufferNum_k, spaceSize + 32);
        pipe->InitBuffer(outBuf, BufferNum_k, spaceSize + 32);

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

        Mins(otherLocal, otherLocal, (T)sizeof(T) * 8, iterSize);
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
        int n = sizeof(T) * 8;
        for (int i = 0; i < n; i += 4) {
            powerLocal.SetValue(i, 1 << i);
            powerLocal.SetValue(i + 1, 1 << (i + 1));
            powerLocal.SetValue(i + 2, 1 << (i + 2));
            powerLocal.SetValue(i + 3, 1 << (i + 3));
        }
        powerLocal.SetValue(n, 0);
        uint32_t blockBeginIndex = 0;
        for (uint32_t i = 0; i < smallLoopTimes; ++i) {
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

    TQue<QuePosition::VECIN, BufferNum_k> inputBuf;
    TQue<QuePosition::VECIN, BufferNum_k> otherBuf;
    TQue<QuePosition::VECOUT, BufferNum_k> outBuf;

    TBuf<QuePosition::VECCALC> powerBuf;
    LocalTensor<T> powerLocal;
    uint32_t size;
    uint32_t smallLoopTimes;
    uint32_t n_elements_per_iter;
};

extern "C" __global__ __aicore__ void bitwise_left_shift(GM_ADDR input, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    // TODO: user kernel impl
    if (TILING_KEY_IS(1)) {
        if constexpr (std::is_same<DTYPE_INPUT, int64_t>::value) {
            KernelBitwiseLeftShift_sca<int64_t> kernel;
            kernel.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.totalSize, tiling_data.mmInputDims, tiling_data.mmOtherDims,
                        tiling_data.mmOutputDims, tiling_data.nOutputDims, &pipe);
            kernel.Process();
        } else if constexpr (std::is_same<DTYPE_INPUT, int32_t>::value) {
            KernelBitwiseLeftShift_32<int32_t> kernel;
            kernel.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, &pipe);
            kernel.Process();
        } else if constexpr (std::is_same<DTYPE_INPUT, int16_t>::value) {
            KernelBitwiseLeftShift_16<int16_t> kernel;
            kernel.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, &pipe);
            kernel.Process();
        } else if constexpr (std::is_same<DTYPE_INPUT, int8_t>::value) {
            KernelBitwiseLeftShift_8<int8_t> kernel;
            kernel.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, &pipe);
            kernel.Process();
        }
    } else if (TILING_KEY_IS(2)) {
        KernelBitwiseLeftShift_sca<DTYPE_INPUT> kernel;
        kernel.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.totalSize, tiling_data.mmInputDims, tiling_data.mmOtherDims,
                    tiling_data.mmOutputDims, tiling_data.nOutputDims, &pipe);
        kernel.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelBitwiseLeftShift_64_spec<int64_t> kernel;
        kernel.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.totalSize, tiling_data.mmInputDims, tiling_data.mmOtherDims,
                    tiling_data.mmOutputDims, tiling_data.nOutputDims, &pipe);
        kernel.Process();
    }
}