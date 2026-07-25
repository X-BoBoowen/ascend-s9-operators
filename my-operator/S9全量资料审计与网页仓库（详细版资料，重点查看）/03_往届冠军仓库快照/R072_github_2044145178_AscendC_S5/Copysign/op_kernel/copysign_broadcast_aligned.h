#include "kernel_operator.h"
#include <type_traits>

using namespace AscendC;
#define BufferNum   1

// Debug control macro
#define DEBUG_PRINT 0  // Set to 1 to enable debug prints, 0 to disable

template <typename T>
class KernelCopysignBroadcast {
public:
    __aicore__ inline KernelCopysignBroadcast() {}
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

        if constexpr (std::is_same<DTYPE_INPUT, float>::value) {
            uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(160 / 10) * 1024 / BufferNum);
            this->n_elements_per_iter = spaceSize / sizeof(T);
            this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;
            pipe->InitBuffer(inputBuf, BufferNum, spaceSize + 64);
            pipe->InitBuffer(otherBuf, BufferNum, spaceSize + 64);
            pipe->InitBuffer(outBuf, BufferNum, spaceSize + 64);
            pipe->InitBuffer(selMaskQueue, BufferNum, spaceSize / 16);

            pipe->InitBuffer(tmpBuf, spaceSize * 5 + 64);
            pipe->InitBuffer(outIndexBuf, spaceSize + 64);
            pipe->InitBuffer(inIndexBuf, spaceSize + 64);
        } else {
            uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(160 / 20) * 1024 / BufferNum);
            this->n_elements_per_iter = spaceSize / sizeof(T);
            this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;
            pipe->InitBuffer(inputBuf, BufferNum, spaceSize + 32);
            pipe->InitBuffer(otherBuf, BufferNum, spaceSize * 2 + 32);
            pipe->InitBuffer(outBuf, BufferNum, spaceSize * 2 + 32);
            pipe->InitBuffer(selMaskQueue, BufferNum, spaceSize / 16);

            pipe->InitBuffer(tmpBuf, spaceSize * 2 * 5 + 32);
            pipe->InitBuffer(outIndexBuf, spaceSize * 2 + 32);
            pipe->InitBuffer(inIndexBuf, spaceSize * 2 + 32);
        }
    }

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

            if (inputShape[dim] != 1) {
                Muls(coord, coord, stride, calCount);
                Add(inputIdx, inputIdx, coord, calCount);
                stride *= inputShape[dim];
            }
        }
    }

    __aicore__ inline void CopyInBroadcast(const GlobalTensor<T> &srcGm, const LocalTensor<T> &inputLocal, uint32_t offset, uint32_t iterSize, uint16_t *inputShape, uint16_t *outputShape,
                                           const int nDims) {
        LocalTensor<int> inputIdxTensor = inIndexBuf.Get<int>();
        LocalTensor<int> outputIdxTensor = outIndexBuf.Get<int>();
        CreateVecIndex(outputIdxTensor, (int)offset, iterSize); // 创建输出张量的线性下标
        mapIndex(inputIdxTensor, outputIdxTensor, iterSize, inputShape, outputShape, nDims);
        LocalTensor<float> inputIdxTensor_fp32 = inputIdxTensor.template ReinterpretCast<float>();
        Cast(inputIdxTensor_fp32, inputIdxTensor, RoundMode::CAST_RINT, iterSize);
        LocalTensor<float> workLocal = tmpBuf.GetWithOffset<float>(iterSize, 0);
        ReduceMax(workLocal, inputIdxTensor_fp32, workLocal, iterSize, false);
        int GMmaxIndex = workLocal.GetValue(0);
        ReduceMin(workLocal, inputIdxTensor_fp32, workLocal, iterSize, false);
        int GMminIndex = workLocal.GetValue(0);

#if DEBUG_PRINT
        printf("offset:%d, iterSize:%d\n", offset, iterSize);
        printf("GMmaxIndex:%d, GMminIndex:%d\n", GMmaxIndex, GMminIndex);
#endif

        int totalGMSize = GMmaxIndex - GMminIndex + 1;
        int numTiles = (totalGMSize + iterSize - 1) / iterSize;

#if DEBUG_PRINT
        printf("numTiles:%d\n", numTiles);
#endif

        LocalTensor<T> inputOriginTile = tmpBuf.GetWithOffset<T>(iterSize, 0);
        for (int i = 0; i < numTiles; ++i) {
            int tileStart = i * iterSize + GMminIndex;
            int tileEnd = min(tileStart + (int)iterSize, GMminIndex + totalGMSize);
            int tileSize = tileEnd - tileStart;

            uint16_t blockCount = 1;
            uint32_t blockLen = tileSize * sizeof(T);

#if DEBUG_PRINT
            printf("tileStart:%d, tileEnd:%d, tileSize:%d\n", tileStart, tileEnd, tileSize);
            printf("blockLen:%d\n", blockLen);
#endif

            DataCopyExtParams copyParams{blockCount, blockLen, 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

            int32_t eventIDS_V_MTE2_0 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
            SetFlag<AscendC::HardEvent::V_MTE2>(eventIDS_V_MTE2_0);
            WaitFlag<AscendC::HardEvent::V_MTE2>(eventIDS_V_MTE2_0);

            DataCopyPad(inputOriginTile, srcGm[tileStart], copyParams, padParams);

#if DEBUG_PRINT
            for (int j = 0; j < 8; ++j) {
                printf("inputOriginTile[%d]:%f\n", j, inputOriginTile.GetValue(j));
            }
#endif

            int32_t eventIDS_MTE2_V_0 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            SetFlag<AscendC::HardEvent::MTE2_V>(eventIDS_MTE2_V_0);
            WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDS_MTE2_V_0);

            LocalTensor<uint8_t> maskTensorGE = tmpBuf.GetWithOffset<uint8_t>(iterSize, iterSize * sizeof(int32_t));
            CompareScalar(maskTensorGE, inputIdxTensor_fp32, (float)tileEnd, CMPMODE::GE, iterSize);
            LocalTensor<uint8_t> maskTensorLT = tmpBuf.GetWithOffset<uint8_t>(iterSize, iterSize * sizeof(int32_t) * 2);
            CompareScalar(maskTensorLT, inputIdxTensor_fp32, (float)tileStart, CMPMODE::LT, iterSize);

#if DEBUG_PRINT
            for (int j = 0; j < 8; ++j) {
                printf("maskTensorGE[%d]:%d\n", j, maskTensorGE.GetValue(j));
                printf("maskTensorLT[%d]:%d\n", j, maskTensorLT.GetValue(j));
            }
#endif

            LocalTensor<float> relIdxTensor_fp32 = tmpBuf.GetWithOffset<float>(iterSize, iterSize * sizeof(int32_t) * 3);

#if DEBUG_PRINT
            for (int j = 0; j < 8; ++j) {
                printf("inputIdxTensor_fp32[%d]:%f\n", j, inputIdxTensor_fp32.GetValue(j));
            }
#endif

            Mins(relIdxTensor_fp32, inputIdxTensor_fp32, (float)tileEnd - 1, iterSize);

#if DEBUG_PRINT
            for (int j = 0; j < 8; ++j) {
                printf("Min inputIdxTensor_fp32[%d]:%f\n", j, relIdxTensor_fp32.GetValue(j));
            }
#endif

            Maxs(relIdxTensor_fp32, relIdxTensor_fp32, (float)tileStart, iterSize);

#if DEBUG_PRINT
            for (int j = 0; j < 8; ++j) {
                printf("Max inputIdxTensor_fp32[%d]:%f\n", j, relIdxTensor_fp32.GetValue(j));
            }
#endif

            Adds(relIdxTensor_fp32, relIdxTensor_fp32, (float)-tileStart, iterSize);

#if DEBUG_PRINT
            for (int j = 0; j < 8; ++j) {
                printf("relIdxTensor_fp32[%d]:%f\n", j, relIdxTensor_fp32.GetValue(j));
            }
#endif

            LocalTensor<int32_t> relIdxTensor_int = relIdxTensor_fp32.template ReinterpretCast<int32_t>();
            Cast(relIdxTensor_int, relIdxTensor_fp32, RoundMode::CAST_RINT, iterSize);

            if constexpr (std::is_same<DTYPE_INPUT, float>::value) {
                ShiftLeft(relIdxTensor_int, relIdxTensor_int, 2, iterSize);
            } else {
                ShiftLeft(relIdxTensor_int, relIdxTensor_int, 1, iterSize);
            }

#if DEBUG_PRINT
            for (int j = 0; j < 8; ++j) {
                printf("relIdxTensor_int[%d]:%d\n", j, relIdxTensor_int.GetValue(j));
            }
#endif

            LocalTensor<T> tmpInputLocal = tmpBuf.GetWithOffset<T>(iterSize, iterSize * sizeof(int32_t) * 4);

            Gather(tmpInputLocal, inputOriginTile, relIdxTensor_int.template ReinterpretCast<uint32_t>(), 0, iterSize);

#if DEBUG_PRINT
            for (int j = 0; j < 8; ++j) {
                printf("tmpInputLocal[%d]:%f\n", j, tmpInputLocal.GetValue(j));
            }
#endif

            Select(inputLocal, maskTensorLT, inputLocal, tmpInputLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);
            Select(inputLocal, maskTensorGE, inputLocal, tmpInputLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);
        }
    }

    __aicore__ inline void Process_iter(uint32_t offset, uint32_t iterSize) {
        LocalTensor<T> inputLocal = inputBuf.AllocTensor<T>();
        LocalTensor<T> otherLocal = otherBuf.AllocTensor<T>();

#if DEBUG_PRINT
        printf("input index:\n");
#endif
        CopyInBroadcast(inputGm, inputLocal, offset, iterSize, mmInputDims, mmOutputDims, nOutputDims);

#if DEBUG_PRINT
        printf("other index:\n");
#endif
        CopyInBroadcast(otherGm, otherLocal, offset, iterSize, mmOtherDims, mmOutputDims, nOutputDims);

        inputBuf.EnQue<T>(inputLocal);
        otherBuf.EnQue<T>(otherLocal);
        inputLocal = inputBuf.DeQue<T>();
        otherLocal = otherBuf.DeQue<T>();

        LocalTensor<uint8_t> selMaskLocal = selMaskQueue.AllocTensor<uint8_t>();
        LocalTensor<T> outLocal = outBuf.AllocTensor<T>();

        Abs(outLocal, inputLocal, iterSize);

#if DEBUG_PRINT
        printf("Block %d, Offset: %d, Iter Size: %d\n", GetBlockIdx(), offset, iterSize);
#endif

        Muls(inputLocal, outLocal, static_cast<T>(-1.0), iterSize);

        CompareScalar(selMaskLocal, otherLocal, static_cast<T>(+0.0), CMPMODE::GE, n_elements_per_iter);
        Select(outLocal, selMaskLocal, outLocal, inputLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);

        inputBuf.FreeTensor<T>(inputLocal);
        otherBuf.FreeTensor<T>(otherLocal);
        selMaskQueue.FreeTensor<uint8_t>(selMaskLocal);

        outBuf.EnQue<T>(outLocal);
        outLocal = outBuf.DeQue<T>();

        DataCopyExtParams storeParams{1, (uint32_t)(iterSize * sizeof(T)), 0, 0, 0};
        DataCopyPad(outGm[offset], outLocal, storeParams);

        outBuf.FreeTensor<T>(outLocal);
    }

    __aicore__ inline void Process() {
        uint32_t blockBeginIndex = 0;

#if DEBUG_PRINT
        printf("Block %d, Size: %d, Small Loop Times: %d, Elements per Iter: %d\n", GetBlockIdx(), size, smallLoopTimes, n_elements_per_iter);
#endif

        for (uint32_t i = 0; i < smallLoopTimes; ++i) {
            uint32_t iterSize = n_elements_per_iter;
            uint32_t offset = blockBeginIndex + beginIndex;

#if DEBUG_PRINT
            printf("Iter %d, Offset: %d, Iter Size: %d\n", i, offset, iterSize);
#endif

            Process_iter(offset, iterSize);
            blockBeginIndex += iterSize;
        }
    }

private:
    TPipe *pipe;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> otherGm;
    GlobalTensor<T> outGm;

    TQue<QuePosition::VECIN, BufferNum> inputBuf;
    TQue<QuePosition::VECIN, BufferNum> otherBuf;
    TQue<QuePosition::VECOUT, BufferNum> outBuf;
    TQue<QuePosition::VECCALC, BufferNum> selMaskQueue;

    TBuf<QuePosition::VECCALC> tmpBuf;
    TBuf<QuePosition::VECCALC> outIndexBuf;
    TBuf<QuePosition::VECCALC> inIndexBuf;

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