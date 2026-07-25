#include "kernel_operator.h"
using namespace AscendC;
// Debug control macro
constexpr uint16_t BufferNum_spec = 1;

template <typename T>
class KernelCopysignBroadcastSpec {
public:
    __aicore__ inline KernelCopysignBroadcastSpec() {}
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
        this->inputSize = 1;
        this->otherSize = 1;
        for (int i = 0; i < nOutputDims; i++) {
            this->inputSize *= mmInputDims[i];
            this->otherSize *= mmOtherDims[i];
        }
        uint32_t max_size_for_broadcast = 1;
        if (this->inputSize != this->totalSize) {
            max_size_for_broadcast = max(this->inputSize, max_size_for_broadcast);
        }
        if (this->otherSize != this->totalSize) {
            max_size_for_broadcast = max(this->otherSize, max_size_for_broadcast);
        }
        inputGm.SetGlobalBuffer((__gm__ T *)input, totalSize + 255);
        otherGm.SetGlobalBuffer((__gm__ T *)other, totalSize + 255);
        outGm.SetGlobalBuffer((__gm__ T *)out, totalSize + 255);
        if constexpr (std::is_same<DTYPE_INPUT, float>::value) {
            uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(180 / 8) * 1024 / BufferNum_spec);
            this->n_elements_per_iter = spaceSize / sizeof(T);
            this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;

            // 广播张量最大个数：24K。
            //  tmpBuf最小 spaceSize*4
            pipe->InitBuffer(tmpBuf, max((uint32_t)(spaceSize * 5), (uint32_t)(max_size_for_broadcast * sizeof(T) + spaceSize)));
            pipe->InitBuffer(inputBuf, BufferNum_spec, spaceSize);
            pipe->InitBuffer(otherBuf, BufferNum_spec, spaceSize);
            pipe->InitBuffer(outBuf, BufferNum_spec, spaceSize);
        } else {
            uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(180 / 12) * 1024 / BufferNum_spec);
            this->n_elements_per_iter = spaceSize / sizeof(T);
            this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;
            // 广播张量最大个数：24K。
            //  tmpBuf最小 spaceSize*8
            pipe->InitBuffer(tmpBuf, max((uint32_t)(spaceSize * 9), (uint32_t)(max_size_for_broadcast * sizeof(T) + spaceSize * 2)));
            pipe->InitBuffer(inputBuf, BufferNum_spec, spaceSize);
            pipe->InitBuffer(otherBuf, BufferNum_spec, spaceSize);
            pipe->InitBuffer(outBuf, BufferNum_spec, spaceSize);
        }
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
    __aicore__ inline void CopyInBroadcast(const GlobalTensor<T> &srcGm, const LocalTensor<T> &inputLocal, uint32_t offset, uint32_t iterSize, uint16_t *inputShape, uint16_t *outputShape,
                                           const int nDims, const uint32_t srcSize) {
        LocalTensor<int> outputIdxTensor = tmpBuf.GetWithOffset<int>(iterSize, iterSize * 8);
        // 不能被修改
        LocalTensor<int> inputIdxTensor = tmpBuf.GetWithOffset<int>(iterSize, iterSize * 12);
        CreateVecIndex(outputIdxTensor, (int)offset, iterSize);

        mapIndex(inputIdxTensor, outputIdxTensor, iterSize, inputShape, outputShape, nDims);
        // srcLocal最多分配 iterSize *16 B
        LocalTensor<T> srcLocal = tmpBuf.Get<T>();

        uint16_t blockCount = 1;
        uint32_t blockLen = srcSize * sizeof(T);
        DataCopyExtParams copyParams{blockCount, blockLen, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        int32_t eventIDS_V_MTE2_0 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        SetFlag<AscendC::HardEvent::V_MTE2>(eventIDS_V_MTE2_0);
        WaitFlag<AscendC::HardEvent::V_MTE2>(eventIDS_V_MTE2_0);

        DataCopyPad(srcLocal, srcGm, copyParams, padParams);

        int32_t eventIDS_MTE2_V_0 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        SetFlag<AscendC::HardEvent::MTE2_V>(eventIDS_MTE2_V_0);
        WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDS_MTE2_V_0);

        if constexpr (std::is_same<DTYPE_INPUT, float>::value) {
            ShiftLeft(inputIdxTensor, inputIdxTensor, 2, iterSize);
        } else {
            ShiftLeft(inputIdxTensor, inputIdxTensor, 1, iterSize);
        }
        Gather(inputLocal, srcLocal, inputIdxTensor.template ReinterpretCast<uint32_t>(), 0, iterSize);
    }
    __aicore__ inline void Process_iter(uint32_t offset, uint32_t iterSize) {// out绝对偏移量,以元素为单位

        LocalTensor<T> inputLocal = inputBuf.AllocTensor<T>();
        LocalTensor<T> otherLocal = otherBuf.AllocTensor<T>();

        if (inputSize != totalSize) {
            CopyInBroadcast(inputGm, inputLocal, offset, iterSize, mmInputDims, mmOutputDims, nOutputDims, inputSize);
            inputBuf.EnQue<T>(inputLocal);
            inputLocal = inputBuf.DeQue<T>();
        } else {
            DataCopy(inputLocal, inputGm[offset], iterSize);
            inputBuf.EnQue<T>(inputLocal);
            inputLocal = inputBuf.DeQue<T>();
        }
        if (otherSize != totalSize) {
            CopyInBroadcast(otherGm, otherLocal, offset, iterSize, mmOtherDims, mmOutputDims, nOutputDims, otherSize);
            otherBuf.EnQue<T>(otherLocal);
            otherLocal = otherBuf.DeQue<T>();
        } else {
            DataCopy(otherLocal, otherGm[offset], iterSize);
            otherBuf.EnQue<T>(otherLocal);
            otherLocal = otherBuf.DeQue<T>();
        }
        LocalTensor<T> outLocal = outBuf.AllocTensor<T>();
        LocalTensor<uint8_t> selMaskLocal = tmpBuf.Get<uint8_t>();

        Abs(outLocal, inputLocal, iterSize);

        Muls(inputLocal, outLocal, static_cast<T>(-1.0), iterSize);

        CompareScalar(selMaskLocal, otherLocal, static_cast<T>(+0.0), CMPMODE::GE, iterSize);
        Select(outLocal, selMaskLocal, outLocal, inputLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);

        inputBuf.FreeTensor<T>(inputLocal);
        otherBuf.FreeTensor<T>(otherLocal);

        outBuf.EnQue<T>(outLocal);
        outLocal = outBuf.DeQue<T>();
        // 将结果写回全局内存
        DataCopyExtParams storeParams{1, (uint32_t)(iterSize * sizeof(T)), 0, 0, 0};
        DataCopyPad(outGm[offset], outLocal, storeParams);
        outBuf.FreeTensor<T>(outLocal);
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
    TQue<QuePosition::VECIN, BufferNum_spec> inputBuf;
    TQue<QuePosition::VECIN, BufferNum_spec> otherBuf;
    TQue<QuePosition::VECOUT, BufferNum_spec> outBuf;
    TBuf<QuePosition::VECCALC> tmpBuf;

    uint32_t size;
    uint32_t smallLoopTimes;
    uint32_t n_elements_per_iter;
    uint32_t totalSize;
    uint32_t inputSize;
    uint32_t otherSize;

    uint16_t *mmInputDims;
    uint16_t *mmOtherDims;
    uint16_t *mmOutputDims;
    int nOutputDims;
    uint32_t beginIndex;
};
