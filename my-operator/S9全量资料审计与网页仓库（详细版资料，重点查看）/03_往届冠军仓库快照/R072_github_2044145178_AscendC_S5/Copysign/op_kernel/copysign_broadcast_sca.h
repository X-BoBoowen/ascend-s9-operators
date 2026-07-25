#include "kernel_operator.h"
#include <type_traits>

using namespace AscendC;
#define BufferNum 1

// 用位运算实现符号转化
template <typename T>
__aicore__ inline T replace_msb(T dst, T src) {
    constexpr T mask = static_cast<T>(1) << (sizeof(T) * 8 - 1);
    return (dst & ~mask) | (src & mask);
}
template <typename T>
class KernelCopysignBroadcastSca {
public:
    __aicore__ inline KernelCopysignBroadcastSca() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out, uint32_t smallSize, uint32_t incSize, uint16_t formerNum, uint32_t totalSize, uint16_t *mmInputDims, uint16_t *mmOtherDims,
                                uint16_t *mmOutputDims, uint8_t nOutputDims, TPipe *pipeIn) {
        this->pipe = pipeIn;
        this->mmInputDims = mmInputDims;
        this->mmOtherDims = mmOtherDims;
        this->mmOutputDims = mmOutputDims;
        this->nOutputDims = nOutputDims;

        this->totalSize = totalSize;

        this->beginIndex = 0;

        if (GetBlockIdx() < formerNum) {
            this->size = smallSize + incSize;
            beginIndex = this->size * GetBlockIdx();
        } else {
            this->size = smallSize;
            beginIndex = this->size * GetBlockIdx() + formerNum * incSize;
        }

        inputGm.SetGlobalBuffer((__gm__ T *)input, totalSize + 255);
        otherGm.SetGlobalBuffer((__gm__ T *)other, totalSize + 255);
        outGm.SetGlobalBuffer((__gm__ T *)out, totalSize + 255);

        uint32_t spaceSize = min((uint32_t)(size * sizeof(int32_t)), (uint32_t)(160 / 8) * 1024 / BufferNum);
        this->n_elements_per_iter = spaceSize / sizeof(int32_t);
        this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;

        pipe->InitBuffer(tmpBuf, spaceSize * 5 + 32);
        pipe->InitBuffer(outIndexBuf, spaceSize + 32);
        pipe->InitBuffer(inputIndexBuf, spaceSize + 32);
        pipe->InitBuffer(otherIndexBuf, spaceSize + 32);
    }
    // 会修改outputIdx的值
    // 最多表示16M个数
    __aicore__ inline void mapIndex(const LocalTensor<int> &inputIdx, const LocalTensor<int> &outputIdx, const int calCount, uint16_t *inputShape, uint16_t *outputShape, const int nDims) {
        LocalTensor<float> outputDim_float = tmpBuf.GetWithOffset<float>(n_elements_per_iter, 0);    // 输出张量的当前维度
        LocalTensor<int> coord = tmpBuf.GetWithOffset<int>(n_elements_per_iter, n_elements_per_iter * 4);      // 当前维度坐标
        LocalTensor<float> remainingIdx_float = tmpBuf.GetWithOffset<float>(n_elements_per_iter, n_elements_per_iter * 4 * 2);  // 剩余待分解的输出下标
        LocalTensor<float> tmpRemainingIdx_float = tmpBuf.GetWithOffset<float>(n_elements_per_iter, n_elements_per_iter * 4 * 3);      // 暂存一下剩余待分解的输出下标

        Cast(remainingIdx_float, outputIdx, RoundMode::CAST_NONE, calCount);

        Duplicate(inputIdx, 0, calCount);

        int stride = 1;        // 输入张量的步长累积

        for (int dim = nDims - 1; dim >= 0; --dim) {// 从尾向头处理
            int intermediate = outputShape[dim]; // 将 uint16_t 转换为 int32_t
            float result = intermediate;   // 再将 int32_t 转换为 float
            Duplicate(outputDim_float, result, calCount);
            // 潜在优化项
            Adds(tmpRemainingIdx_float, remainingIdx_float, (float)0.0, calCount);

            Div(remainingIdx_float, remainingIdx_float, outputDim_float, calCount);
            // 截断remainingIdx_float
            Cast(outputIdx, remainingIdx_float, RoundMode::CAST_FLOOR, calCount);
            Cast(remainingIdx_float, outputIdx, RoundMode::CAST_NONE, calCount);

            LocalTensor<int> outputDim_int = tmpBuf.GetWithOffset<int>(n_elements_per_iter, n_elements_per_iter * 4 * 4);
            Duplicate(outputDim_int, (int)outputShape[dim], calCount);

            Mul(outputIdx, outputIdx, outputDim_int, calCount);

            LocalTensor<int> tmpRemainingIdx_int = tmpBuf.GetWithOffset<int>(n_elements_per_iter, n_elements_per_iter * 4 * 4);
            Cast(tmpRemainingIdx_int, tmpRemainingIdx_float, RoundMode::CAST_RINT, calCount);

            Sub(coord, tmpRemainingIdx_int, outputIdx, calCount);

                // 仅当输入维度不为1时累加坐标贡献
                // 本质上是0*stride，维度为1只能取0
                // 维度不为1，则表示在该维度上，input和output相等，所以就是coord * stride
                // “stride *= inputShape[dim];”也可向外提
            if (inputShape[dim] != 1) {
                Muls(coord, coord, stride, calCount);
                Add(inputIdx, inputIdx, coord, calCount);
                stride *= inputShape[dim];
            }
        }
    }
    __aicore__ inline void Process_iter(uint32_t offset, uint32_t iterSize) {// out绝对偏移量,以元素为单位
        LocalTensor<int> outputIdxTensor = outIndexBuf.Get<int>();
        LocalTensor<int> inputIdxTensor = inputIndexBuf.Get<int>();
        LocalTensor<int> otherIdxTensor = otherIndexBuf.Get<int>();
        CreateVecIndex(outputIdxTensor, (int)offset, iterSize);
        mapIndex(inputIdxTensor, outputIdxTensor, iterSize, mmInputDims, mmOutputDims, nOutputDims);
        CreateVecIndex(outputIdxTensor, (int)offset, iterSize);
        mapIndex(otherIdxTensor, outputIdxTensor, iterSize, mmOtherDims, mmOutputDims, nOutputDims);
        for (uint32_t i = 0; i < iterSize; i++) {
            uint32_t inputIdx = inputIdxTensor.GetValue(i);
            uint32_t otherIdx = otherIdxTensor.GetValue(i);
            // printf("Block %d, i: %d, inputIdx: %d, otherIdx: %d\n", GetBlockIdx(), i, inputIdx, otherIdx);
            T input_val = inputGm.GetValue(inputIdx);
            T other_val = otherGm.GetValue(otherIdx);
            T out_val = replace_msb(input_val, other_val);
            outGm.SetValue(i + offset, out_val);
        }
    }
    __aicore__ inline void Process() {
        uint32_t blockBeginIndex = 0;
        // printf("Block %d, Size: %d, Small Loop Times: %d, Elements per Iter: %d\n", GetBlockIdx(), size, smallLoopTimes, n_elements_per_iter);
        for (uint32_t i = 0; i < smallLoopTimes; ++i) {
            uint32_t iterSize = n_elements_per_iter;
            uint32_t offset = blockBeginIndex + beginIndex;
            if (offset + iterSize > totalSize) {
                iterSize = totalSize - offset;
            }
            Process_iter(offset, iterSize);// 绝对偏移量
            blockBeginIndex += iterSize;
        }
    }

private:
    TPipe *pipe;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> otherGm;
    GlobalTensor<T> outGm;

    TBuf<QuePosition::VECCALC> tmpBuf; // 用于存储计算索引的临时空间
    TBuf<QuePosition::VECCALC> outIndexBuf; // 用于存储计算索引的临时空间
    TBuf<QuePosition::VECCALC> inputIndexBuf; // 用于存储计算索引的临时空间
    TBuf<QuePosition::VECCALC> otherIndexBuf; // 用于存储计算索引的临时空间

    uint32_t size;
    uint32_t smallLoopTimes;
    uint32_t n_elements_per_iter;
    uint32_t totalSize;

    uint16_t *mmInputDims;
    uint16_t *mmOtherDims;
    uint16_t *mmOutputDims;
    int nOutputDims;
    uint32_t beginIndex;
};