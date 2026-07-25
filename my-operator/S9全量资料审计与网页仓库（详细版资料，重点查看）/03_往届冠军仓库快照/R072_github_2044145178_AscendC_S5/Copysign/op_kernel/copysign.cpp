#include "kernel_operator.h"
#include "copysign_broadcast_sca.h"
#include "copysign_broadcast_aligned.h"
#include "copysign_broadcast_spec.h"
using namespace AscendC;
constexpr uint16_t BufferNum_k = 1;
template <typename T>
class KernelCopysign {
public:
    __aicore__ inline KernelCopysign() {}
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

        uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(180 / 3) * 1024 / BufferNum_k);
        this->n_elements_per_iter = spaceSize / sizeof(T);
        this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;

        pipe->InitBuffer(inputBuf, BufferNum_k, spaceSize + 32);
        pipe->InitBuffer(otherBuf, BufferNum_k, spaceSize + 32);
        pipe->InitBuffer(outBuf, BufferNum_k, spaceSize + 32);
        pipe->InitBuffer(selMaskQueue, BufferNum_k, spaceSize >> 4);
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
        LocalTensor<uint8_t> selMaskLocal = selMaskQueue.AllocTensor<uint8_t>();
        // 计算结果
        Abs(outLocal, inputLocal, iterSize);
        // 乘以-1应该没影响
        Muls(inputLocal, outLocal, static_cast<T>(-1.0), iterSize);
        // 大于等于0，从outLocal取，否则从inputLocal取
        CompareScalar(selMaskLocal, otherLocal, static_cast<T>(+0.0), CMPMODE::GE, iterSize);
        Select(outLocal, selMaskLocal, outLocal, inputLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);
        inputBuf.FreeTensor<T>(inputLocal);
        otherBuf.FreeTensor<T>(otherLocal);
        selMaskQueue.FreeTensor<uint8_t>(selMaskLocal);

        outBuf.EnQue<T>(outLocal);
        outLocal = outBuf.DeQue<T>();
        // 将结果写回全局内存
        DataCopy(outGm[beginIndex], outLocal, iterSize);
        outBuf.FreeTensor<T>(outLocal);
    }
    __aicore__ inline void Process() {
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

    TQue<QuePosition::VECIN, BufferNum_k> inputBuf;
    TQue<QuePosition::VECIN, BufferNum_k> otherBuf;
    TQue<QuePosition::VECOUT, BufferNum_k> outBuf;
    TQue<QuePosition::VECCALC, BufferNum_k> selMaskQueue;

    uint32_t size;
    uint32_t smallLoopTimes;
    uint32_t n_elements_per_iter;
};

extern "C" __global__ __aicore__ void copysign(GM_ADDR input, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    // TODO: user kernel impl
    if (TILING_KEY_IS(1)) {
        if constexpr (std::is_same<DTYPE_INPUT, float>::value) {
            KernelCopysign<float> kernel;
            kernel.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, &pipe);
            kernel.Process();
        } else {
            KernelCopysign<half> kernel;
            kernel.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, &pipe);
            kernel.Process();
        }
    } else if (TILING_KEY_IS(2)) {
        if constexpr (std::is_same<DTYPE_INPUT, float>::value) {
            KernelCopysignBroadcast<float> kernel;
            kernel.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.totalSize, tiling_data.mmInputDims, tiling_data.mmOtherDims,
                        tiling_data.mmOutputDims, tiling_data.nOutputDims, &pipe);
            kernel.Process();
        } else {
            KernelCopysignBroadcast<half> kernel;
            kernel.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.totalSize, tiling_data.mmInputDims, tiling_data.mmOtherDims,
                        tiling_data.mmOutputDims, tiling_data.nOutputDims, &pipe);
            kernel.Process();
        }
    } else if (TILING_KEY_IS(3)) {
        if constexpr (std::is_same<DTYPE_INPUT, float>::value) {
            KernelCopysignBroadcastSpec<float> kernel;
            kernel.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.totalSize, tiling_data.mmInputDims, tiling_data.mmOtherDims,
                        tiling_data.mmOutputDims, tiling_data.nOutputDims, &pipe);
            kernel.Process();
        } else {
            KernelCopysignBroadcastSpec<half> kernel;
            kernel.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.totalSize, tiling_data.mmInputDims, tiling_data.mmOtherDims,
                        tiling_data.mmOutputDims, tiling_data.nOutputDims, &pipe);
            kernel.Process();
        }
    }
}