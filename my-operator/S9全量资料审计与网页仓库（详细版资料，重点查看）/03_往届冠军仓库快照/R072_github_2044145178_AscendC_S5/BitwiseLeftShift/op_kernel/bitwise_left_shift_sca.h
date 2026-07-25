#include "kernel_operator.h"
using namespace AscendC;
// Debug control macro
#define DEBUG_PRINT 1  // Set to 1 to enable debug prints, 0 to disable
constexpr uint16_t BufferNum_sca = 1;
template <typename T>
class KernelBitwiseLeftShift_sca {
public:
    __aicore__ inline KernelBitwiseLeftShift_sca() {}
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

        uint32_t spaceSize = min((uint32_t)(size * sizeof(int32_t)), (uint32_t)(180 / 5) * 1024);
        this->n_elements_per_iter = spaceSize / sizeof(int32_t);
        this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;

        pipe->InitBuffer(tmpBuf, spaceSize * 2 + 32);
        pipe->InitBuffer(outIndexBuf, spaceSize + 32);
        pipe->InitBuffer(inputIndexBuf, spaceSize + 32);
        pipe->InitBuffer(otherIndexBuf, spaceSize + 32);
    }
    // 会修改outputIdx的值
    __aicore__ inline void mapIndex(const LocalTensor<int> &inputIdx, const LocalTensor<int> &outputIdx, const int calCount, uint16_t *inputShape, uint16_t *outputShape, const int nDims) {
        LocalTensor<float> temp_float = tmpBuf.GetWithOffset<float>(calCount, 0);
        LocalTensor<float> tmpRemainingIdx_float = tmpBuf.GetWithOffset<float>(calCount, calCount * 4);

        LocalTensor<float> remainingIdx_float = outputIdx.template ReinterpretCast<float>();
        LocalTensor<float> inputIdx_float = inputIdx.template ReinterpretCast<float>();
        Cast(remainingIdx_float, outputIdx, RoundMode::CAST_NONE, calCount);

        Duplicate(inputIdx_float, (float)0.0, calCount);

        int stride = 1;// 输入张量的步长累积

        for (int dim = nDims - 1; dim >= 0; --dim) {// 从尾向头处理
            int intermediate = outputShape[dim]; // 将 uint16_t 转换为 int32_t
            float result = intermediate;   // 再将 int32_t 转换为 float
            Duplicate(temp_float, result, calCount);
            // 暂存一下remainingIdx_float
            Adds(tmpRemainingIdx_float, remainingIdx_float, (float)0.0, calCount);

            Div(remainingIdx_float, remainingIdx_float, temp_float, calCount);
            // 截断remainingIdx_float
            Cast(outputIdx, remainingIdx_float, RoundMode::CAST_FLOOR, calCount);
            Cast(remainingIdx_float, outputIdx, RoundMode::CAST_NONE, calCount);

            Mul(temp_float, remainingIdx_float, temp_float, calCount);

            Sub(temp_float, tmpRemainingIdx_float, temp_float, calCount);

            if (inputShape[dim] != 1) {
                Muls(temp_float, temp_float, (float)stride, calCount);
                Add(inputIdx_float, inputIdx_float, temp_float, calCount);
                stride *= inputShape[dim];
            }
        }
        Cast(inputIdx, inputIdx_float, RoundMode::CAST_RINT, calCount);
    }
    __aicore__ inline void Process_iter(uint32_t offset, uint32_t iterSize) {// out绝对偏移量,以元素为单位
        LocalTensor<int> outputIdxTensor = outIndexBuf.Get<int>();
        LocalTensor<int> inputIdxTensor = inputIndexBuf.Get<int>();
        LocalTensor<int> otherIdxTensor = otherIndexBuf.Get<int>();
        CreateVecIndex(outputIdxTensor, (int)offset, iterSize);
        mapIndex(inputIdxTensor, outputIdxTensor, iterSize, mmInputDims, mmOutputDims, nOutputDims);
        CreateVecIndex(outputIdxTensor, (int)offset, iterSize);
        mapIndex(otherIdxTensor, outputIdxTensor, iterSize, mmOtherDims, mmOutputDims, nOutputDims);
        constexpr T n_bits = sizeof(T) * 8;
        for (uint32_t i = 0; i < iterSize; ++i) {
            uint32_t inputIdx = inputIdxTensor.GetValue(i);
            uint32_t otherIdx = otherIdxTensor.GetValue(i);
            T a = inputGm.GetValue(inputIdx);
            T b = otherGm.GetValue(otherIdx);
            if (b >= n_bits) {
                // 如果b大于类型的位数，或者小于0，则左移无效
                outGm.SetValue(i + offset, 0);
            } else {
                outGm.SetValue(i + offset, a << b);
            }
        }
    }
    __aicore__ inline void Process() {
        uint32_t blockBeginIndex = 0;
        // printf("Block %d, Size: %d, Small Loop Times: %d, Elements per Iter: %d\n", GetBlockIdx(), size, smallLoopTimes, n_elements_per_iter);
        for (uint32_t i = 0; i < smallLoopTimes; ++i) {
            uint32_t iterSize = n_elements_per_iter;
            uint32_t offset = blockBeginIndex + beginIndex;
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