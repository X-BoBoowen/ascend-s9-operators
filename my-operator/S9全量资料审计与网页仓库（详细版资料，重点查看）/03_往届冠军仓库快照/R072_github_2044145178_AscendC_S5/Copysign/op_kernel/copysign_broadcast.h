#include "kernel_operator.h"
#include <type_traits>

using namespace AscendC;
#define BufferNum 1
#include <cstdint>
#include <type_traits>


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
            pipe->InitBuffer(selMaskQueue, BufferNum, spaceSize / 8);

            pipe->InitBuffer(tmpBuf, spaceSize * 5 + 64);
            pipe->InitBuffer(outIndexBuf, spaceSize + 64);
        } else {
            uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(160 / 20) * 1024 / BufferNum);
            this->n_elements_per_iter = spaceSize / sizeof(T);
            this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;
            pipe->InitBuffer(inputBuf, BufferNum, spaceSize + 32);
            pipe->InitBuffer(otherBuf, BufferNum, spaceSize * 2 + 32);
            pipe->InitBuffer(outBuf, BufferNum, spaceSize * 2 + 32);
            pipe->InitBuffer(selMaskQueue, BufferNum, spaceSize / 8);

            pipe->InitBuffer(tmpBuf, spaceSize * 2 * 5 + 32);
            pipe->InitBuffer(outIndexBuf, spaceSize * 2 + 32);
        }
    }
    // 会修改outputIdx的值
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

        LocalTensor<T> inputLocal = inputBuf.AllocTensor<T>();
        LocalTensor<T> otherLocal = otherBuf.AllocTensor<T>();

        // 复用一下otherLocal
        LocalTensor<int> inputIdxTensor = otherLocal.template ReinterpretCast<int>();
        LocalTensor<int> outputIdxTensor = outIndexBuf.Get<int>();

        CreateVecIndex(outputIdxTensor, (int)offset, iterSize); // 创建输出张量的线性下标
        // printf("offset: %d\n", offset);
        // printf("iterSize: %d\n", iterSize);
        // for (uint32_t i = 0; i < 8; ++i) {
        //     printf("outputIdxTensor[%d]: %d\n", i, outputIdxTensor.GetValue(i));
        // }

        mapIndex(inputIdxTensor, outputIdxTensor, iterSize, mmInputDims, mmOutputDims, nOutputDims);

        // for (uint32_t i = 0; i < 8; ++i) {
        //     printf("inputIdxTensor[%d]: %d\n", i, inputIdxTensor.GetValue(i));
        // }

        int GMbeginIndex = inputIdxTensor.GetValue(0); // 获取第一个元素的线性下标
        int GMSize = inputIdxTensor.GetValue(iterSize - 1) - GMbeginIndex + 1; // 获取最后一个元素的线性下标

        LocalTensor<T> inputOriginLocal = tmpBuf.GetWithOffset<T>(iterSize, 0);

        // printf("GMbeginIndex:%d GMSize:%d\n ", GMbeginIndex, GMSize);

        int32_t eventIDS_V_MTE2_0 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        SetFlag<AscendC::HardEvent::V_MTE2>(eventIDS_V_MTE2_0);
        WaitFlag<AscendC::HardEvent::V_MTE2>(eventIDS_V_MTE2_0);

        // printf("DataCopy:%d DataCopy:%d\n ", GMbeginIndex, GMSize);

        DataCopy(inputOriginLocal, inputGm[GMbeginIndex], GMSize + 15);
        // for (uint32_t i = 0; i < 8; ++i) {
        //     printf("inputOriginLocal[%d]: %f\n", i, inputOriginLocal.GetValue(i));
        // }
        int32_t eventIDS_MTE2_V_0 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));

        SetFlag<AscendC::HardEvent::MTE2_V>(eventIDS_MTE2_V_0);
        WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDS_MTE2_V_0);

        Adds(inputIdxTensor, inputIdxTensor, -GMbeginIndex, iterSize); // 将线性下标转换为相对偏移量
        if constexpr (std::is_same<DTYPE_INPUT, float>::value) {
            ShiftLeft(inputIdxTensor, inputIdxTensor, 2, iterSize);
        } else {
            ShiftLeft(inputIdxTensor, inputIdxTensor, 1, iterSize);
        }

        LocalTensor<uint32_t> inputIdxTensor_uint32 = inputIdxTensor.template ReinterpretCast<uint32_t>();

        Gather(inputLocal, inputOriginLocal, inputIdxTensor_uint32, (uint32_t)0, iterSize);

        // for (uint32_t i = 0; i < 8; ++i) {
        //     printf("inputLocal[%d]: %f\n", i, inputLocal.GetValue(i));
        // }

        // 复用一下outLocal
        LocalTensor<T> outLocal = outBuf.AllocTensor<T>();
        LocalTensor<int> otherIdxTensor = outLocal.template ReinterpretCast<int>();

        CreateVecIndex(outputIdxTensor, (int)offset, iterSize); // 创建输出张量的线性下标
        mapIndex(otherIdxTensor, outputIdxTensor, iterSize, mmOtherDims, mmOutputDims, nOutputDims);

        int otherGMbeginIndex = otherIdxTensor.GetValue(0); // 获取第一个元素的线性下标
        int otherGMSize = otherIdxTensor.GetValue(iterSize - 1) - otherGMbeginIndex + 1; // 获取最后一个元素的线性下标

        // printf("otherGMbeginIndex:%d otherGMSize:%d\n ", otherGMbeginIndex, otherGMSize);
        // // Debug print for otherIdxTensor
        // for (uint32_t i = 0; i < 10; ++i) {
        //     printf("otherIdxTensor[%d]: %d\n", i, otherIdxTensor.GetValue(i));
        // }

        LocalTensor<T> otherOriginLocal = tmpBuf.GetWithOffset<T>(iterSize, 0);

        int32_t eventIDS_V_MTE2_1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        SetFlag<AscendC::HardEvent::V_MTE2>(eventIDS_V_MTE2_1);
        WaitFlag<AscendC::HardEvent::V_MTE2>(eventIDS_V_MTE2_1);

        DataCopy(otherOriginLocal, otherGm[otherGMbeginIndex], otherGMSize + 15);
        // for (uint32_t i = 0; i < 10; ++i) {
        //     printf("otherOriginLocal[%d]: %f\n", i, otherOriginLocal.GetValue(i));
        // }
        int32_t eventIDS_MTE2_V_1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        SetFlag<AscendC::HardEvent::MTE2_V>(eventIDS_MTE2_V_1);
        WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDS_MTE2_V_1);

        Adds(otherIdxTensor, otherIdxTensor, -otherGMbeginIndex, iterSize); // 将线性下标转换为相对偏移量
        if constexpr (std::is_same<DTYPE_INPUT, float>::value) {
            ShiftLeft(otherIdxTensor, otherIdxTensor, 2, iterSize);
        } else {
            ShiftLeft(otherIdxTensor, otherIdxTensor, 1, iterSize);
        }
        // for (uint32_t i = 0; i < 10; ++i) {
        //     printf("otherIdxTensor[%d]: %d\n", i, otherIdxTensor.GetValue(i));
        // }
        LocalTensor<uint32_t> otherIdxTensor_uint32 = otherIdxTensor.template ReinterpretCast<uint32_t>();

        Gather(otherLocal, otherOriginLocal, otherIdxTensor_uint32, (uint32_t)0, iterSize);

        // for (uint32_t i = 0; i < 10; ++i) {
        //     printf("otherLocal[%d]: %f\n", i, otherLocal.GetValue(i));
        // }

        inputBuf.EnQue<T>(inputLocal);
        otherBuf.EnQue<T>(otherLocal);
        inputLocal = inputBuf.DeQue<T>();
        otherLocal = otherBuf.DeQue<T>();

        LocalTensor<uint8_t> selMaskLocal = selMaskQueue.AllocTensor<uint8_t>();
        // 计算结果
        Abs(outLocal, inputLocal, iterSize);
        // for (uint32_t i = 0; i < 10; ++i) {
        //     printf("outLocal[%d]: %f\n", i, outLocal.GetValue(i));
        // }
        // 乘以-1应该没影响
        Muls(inputLocal, outLocal, static_cast<T>(-1.0), iterSize);
        // for (uint32_t i = 0; i < 10; ++i) {
        //     printf("inputLocal[%d]: %f\n", i, inputLocal.GetValue(i));
        // }
        // 大于等于0，从outLocal取，否则从inputLocal取
        CompareScalar(selMaskLocal, otherLocal, static_cast<T>(+0.0), CMPMODE::GE, n_elements_per_iter);// 需要256B对齐
        Select(outLocal, selMaskLocal, outLocal, inputLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);
        // for (uint32_t i = 0; i < 10; ++i) {
        //     printf("end outLocal[%d]: %f\n", i, outLocal.GetValue(i));
        // }
        inputBuf.FreeTensor<T>(inputLocal);
        otherBuf.FreeTensor<T>(otherLocal);
        selMaskQueue.FreeTensor<uint8_t>(selMaskLocal);

        outBuf.EnQue<T>(outLocal);
        outLocal = outBuf.DeQue<T>();

        // for (uint32_t i = 0; i < 10; ++i) {
        //     printf("end end outLocal[%d]: %f\n", i, outLocal.GetValue(i));
        // }

        // 将结果写回全局内存
        DataCopyExtParams storeParams{1, (uint32_t)(iterSize * sizeof(T)), 0, 0, 0};
        DataCopyPad(outGm[offset], outLocal, storeParams);

        // for (uint32_t i = 0; i < 10; ++i) {
        //     printf("end end end outLocal[%d]: %f\n", i, outLocal.GetValue(i));
        // }

        outBuf.FreeTensor<T>(outLocal);
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

    TQue<QuePosition::VECIN, BufferNum> inputBuf;
    TQue<QuePosition::VECIN, BufferNum> otherBuf;
    TQue<QuePosition::VECOUT, BufferNum> outBuf;
    TQue<QuePosition::VECCALC, BufferNum> selMaskQueue;

    TBuf<QuePosition::VECCALC> tmpBuf; // 用于存储计算索引的临时空间
    TBuf<QuePosition::VECCALC> outIndexBuf; // 用于存储计算索引的临时空间

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