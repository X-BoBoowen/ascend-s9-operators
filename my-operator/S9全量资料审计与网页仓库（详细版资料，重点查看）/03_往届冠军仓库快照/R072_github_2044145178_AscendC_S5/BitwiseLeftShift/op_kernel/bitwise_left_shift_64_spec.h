#include "kernel_operator.h"
using namespace AscendC;
// Debug control macro
#define DEBUG_PRINT 1  // Set to 1 to enable debug prints, 0 to disable
constexpr uint16_t BufferNum_64 = 2;

// tmpLocal 内存空间大小为 2 * iterSize * 4B
__aicore__ inline void ShiftLeft_int64_1_bit(const LocalTensor<int32_t> &hi, const LocalTensor<int32_t> &lo, const LocalTensor<int32_t> &tmpLocal, uint32_t iterSize) {
    LocalTensor<uint32_t> tmpLocal_uint32_t = tmpLocal.template ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> tmp_hi = tmpLocal_uint32_t[0];  // 临时存储高32位
    LocalTensor<uint32_t> tmp_lo = tmpLocal_uint32_t[iterSize];  // 临时存储低32位

    LocalTensor<uint32_t> hi_uint32_t = hi.template ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> lo_uint32_t = lo.template ReinterpretCast<uint32_t>();

    // hi = (hi << 1) | (lo >> (32 - 1))
    ShiftLeft(tmp_hi, hi_uint32_t, (uint32_t)1, iterSize);         // hi << 1
    ShiftRight(tmp_lo, lo_uint32_t, (uint32_t)31, iterSize);   // lo >> (32 - 1)

    LocalTensor<int16_t> tmp_hi_16 = tmp_hi.template ReinterpretCast<int16_t>();
    LocalTensor<int16_t> tmp_lo_16 = tmp_lo.template ReinterpretCast<int16_t>();

    LocalTensor<int16_t> hi_16 = hi.template ReinterpretCast<int16_t>();
    Or(hi_16, tmp_hi_16, tmp_lo_16, iterSize * 2);           // hi = tmp_hi | tmp_lo
    // lo = lo << 1
    ShiftLeft(lo_uint32_t, lo_uint32_t, (uint32_t)1, iterSize);
}

template <typename T>
class KernelBitwiseLeftShift_64_spec {
public:
    __aicore__ inline KernelBitwiseLeftShift_64_spec() {}
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

        // 是因为T写成了int32_t？
        uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(192 / 6) * 1024);
        this->n_elements_per_iter = spaceSize / sizeof(T);
        this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;

        // 广播张量最大个数：10K。双缓冲就5K个
        //  tmpBuf最小 spaceSize*2
        pipe->InitBuffer(tmpBuf, BufferNum_64, max((uint32_t)(spaceSize * 3), (uint32_t)(max_size_for_broadcast * sizeof(T) + spaceSize)));
        pipe->InitBuffer(inputBuf, BufferNum_64, spaceSize);
        pipe->InitBuffer(otherBuf, BufferNum_64, spaceSize);
        pipe->InitBuffer(outBuf, BufferNum_64, spaceSize);
    }
    // 会修改outputIdx的值
    // 需要2*calCount个float的空间
    __aicore__ inline void mapIndex(const LocalTensor<int> &inputIdx, const LocalTensor<int> &outputIdx, const LocalTensor<float> &tmpLocal, const int calCount, uint16_t *inputShape,
                                    uint16_t *outputShape, const int nDims) {
        LocalTensor<float> temp_float = tmpLocal;
        LocalTensor<float> tmpRemainingIdx_float = tmpLocal[calCount];

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
    // 需要4*calCount个int的空间
    __aicore__ inline void CopyInBroadcast(const GlobalTensor<T> &srcGm, const LocalTensor<T> &inputLocal, const LocalTensor<int32_t> &tmpLocal, uint32_t offset, uint32_t iterSize,
                                           uint16_t *inputShape, uint16_t *outputShape, const int nDims, const uint32_t srcSize) {
        LocalTensor<int> outputIdxTensor = tmpLocal[iterSize * 3];
        // 不能被修改
        LocalTensor<int> inputIdxTensor = tmpLocal[iterSize * 4];
        CreateVecIndex(outputIdxTensor, (int)offset, iterSize);
        mapIndex(inputIdxTensor, outputIdxTensor, tmpLocal.template ReinterpretCast<float>(), iterSize, inputShape, outputShape, nDims);
        // srcLocal最多分配 iterSize *16 B
        LocalTensor<T> srcLocal = tmpLocal.template ReinterpretCast<T>();

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

        // 先提取低32位
        ShiftLeft(inputIdxTensor, inputIdxTensor, 3, iterSize);
        Gather(inputLocal.template ReinterpretCast<int32_t>(), srcLocal.template ReinterpretCast<int32_t>(), inputIdxTensor.template ReinterpretCast<uint32_t>(), 0, iterSize);

        // 再提取高32位
        Adds(inputIdxTensor, inputIdxTensor, (int)4, iterSize);
        Gather((inputLocal.template ReinterpretCast<int32_t>())[iterSize], srcLocal.template ReinterpretCast<int32_t>(), inputIdxTensor.template ReinterpretCast<uint32_t>(), 0, iterSize);
    }
    __aicore__ inline void Process_iter(uint32_t offset, uint32_t iterSize) {// out绝对偏移量,以元素为单位

        LocalTensor<T> inputLocal = inputBuf.AllocTensor<T>();
        LocalTensor<T> otherLocal = otherBuf.AllocTensor<T>();

        LocalTensor<int32_t> inputLocal_32 = inputLocal.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> otherLocal_32 = otherLocal.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> tmpLocal = tmpBuf.AllocTensor<int32_t>();
        if (inputSize != totalSize) {
            CopyInBroadcast(inputGm, inputLocal, tmpLocal, offset, iterSize, mmInputDims, mmOutputDims, nOutputDims, inputSize);
            inputBuf.EnQue<T>(inputLocal);
            inputLocal = inputBuf.DeQue<T>();
        } else {
            DataCopy(inputLocal, inputGm[offset], iterSize);
            inputBuf.EnQue<T>(inputLocal);
            inputLocal = inputBuf.DeQue<T>();
        }
        if (otherSize != totalSize) {
            // 低32位有用
            CopyInBroadcast(otherGm, otherLocal, tmpLocal, offset, iterSize, mmOtherDims, mmOutputDims, nOutputDims, otherSize);
            otherBuf.EnQue<T>(otherLocal);
            otherLocal = otherBuf.DeQue<T>();
        } else {
            DataCopy(otherLocal, otherGm[offset], iterSize);
            otherBuf.EnQue<T>(otherLocal);
            otherLocal = otherBuf.DeQue<T>();
            // 左移位数：int64_t转int32_t
            Cast(otherLocal_32, otherLocal, RoundMode::CAST_NONE, iterSize);
        }
        LocalTensor<int32_t> inputLocal_hi = tmpLocal;
        LocalTensor<int32_t> inputLocal_lo = tmpLocal[iterSize];
        if (inputSize != totalSize) {
            Adds(inputLocal_lo.template ReinterpretCast<int32_t>(), inputLocal.template ReinterpretCast<int32_t>(), (int32_t)0, iterSize);
            Adds(inputLocal_hi.template ReinterpretCast<int32_t>(), (inputLocal.template ReinterpretCast<int32_t>())[iterSize], (int32_t)0, iterSize);
        } else {
            uint8_t repeatTimes = (uint8_t)((iterSize * sizeof(T) + 255) / 256);
            uint64_t rsvdCnt = 0; // 用于保存筛选后保留下来的元素个数
            GatherMask(inputLocal_hi, inputLocal.template ReinterpretCast<int32_t>(), (uint8_t)2, false, 0, {1, repeatTimes, 8, 0}, rsvdCnt);
            GatherMask(inputLocal_lo, inputLocal.template ReinterpretCast<int32_t>(), (uint8_t)1, false, 0, {1, repeatTimes, 8, 0}, rsvdCnt);

            Adds(inputLocal_32[iterSize], inputLocal_hi, (int32_t)0, iterSize);
            Adds(inputLocal_32, inputLocal_lo, (int32_t)0, iterSize);
        }

        constexpr int n_bits = sizeof(T) * 8;

        Mins(otherLocal_32, otherLocal_32, n_bits, iterSize);

        LocalTensor<T> outLocal = outBuf.AllocTensor<T>();
        LocalTensor<uint8_t> selMaskLocal = tmpLocal[iterSize * 2].template ReinterpretCast<uint8_t>();
        LocalTensor<int32_t> outLocal_32 = outLocal.template ReinterpretCast<int32_t>();
        for (int i = 1; i <= n_bits; i++) {
            // inputLocal_32应该没用了，用于暂存分离结果。outLocal_32暂时用于充当临时变量
            ShiftLeft_int64_1_bit(inputLocal_hi, inputLocal_lo, outLocal_32, iterSize);

            CompareScalar(selMaskLocal, otherLocal_32, (int32_t)i, CMPMODE::EQ, iterSize);

            Select(inputLocal_32.template ReinterpretCast<float>(), selMaskLocal, inputLocal_lo.template ReinterpretCast<float>(), inputLocal_32.template ReinterpretCast<float>(),
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);
            Select(inputLocal_32[iterSize].template ReinterpretCast<float>(), selMaskLocal, inputLocal_hi.template ReinterpretCast<float>(), inputLocal_32[iterSize].template ReinterpretCast<float>(),
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);
        }

        // 因为没有scatter，就只能生成index，然后用gather
        // 需要生成[0,0,1,1,....,N-1,N-1],考虑使用Broadcast实现N,1->N,2
        // 再用Adds 和mask实现生成[0,0+N,1,1+N,....,N-1,N-1+N]
        // 再用Gather实现合并成int64
        LocalTensor<int32_t> indexLocal = tmpLocal;
        LocalTensor<float> indexLocal_fp32 = indexLocal.template ReinterpretCast<float>();
        LocalTensor<int32_t> indexLocal_broadcast = tmpLocal[iterSize];
        LocalTensor<float> indexLocal_broadcast_fp32 = indexLocal_broadcast.template ReinterpretCast<float>();
        LocalTensor<uint8_t> sharedTmpBuffer = otherLocal.template ReinterpretCast<uint8_t>();
        CreateVecIndex(indexLocal, 0, iterSize);
        // 需要临时空间
        uint32_t broadcastShapeIn[2] = {iterSize, 1};
        uint32_t broadcastShapeOut[2] = {iterSize, 2};
        Broadcast<float, 2, 1>(indexLocal_broadcast_fp32, indexLocal_fp32, broadcastShapeOut, broadcastShapeIn, sharedTmpBuffer);
        uint64_t mask[2] = {0xAAAAAAAAAAAAAAAA, 0};
        Adds(indexLocal_broadcast, indexLocal_broadcast, (int32_t)iterSize, mask, iterSize * sizeof(int64_t) / 256, {1, 1, 8, 8});

        // 按字节偏移，不要忘记计算个数乘以4
        ShiftLeft(indexLocal_broadcast, indexLocal_broadcast, 2, iterSize * 2);

        Gather(outLocal_32, inputLocal_32, indexLocal_broadcast.template ReinterpretCast<uint32_t>(), 0, iterSize * 2);

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
    TQue<QuePosition::VECIN, 1> inputBuf;
    TQue<QuePosition::VECIN, 1> otherBuf;
    TQue<QuePosition::VECOUT, 1> outBuf;
    TQue<QuePosition::VECCALC, 1> tmpBuf;

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
