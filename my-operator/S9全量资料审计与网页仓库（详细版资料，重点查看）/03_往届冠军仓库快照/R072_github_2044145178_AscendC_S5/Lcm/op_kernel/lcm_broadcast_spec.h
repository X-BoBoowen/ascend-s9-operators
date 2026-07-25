#include "kernel_operator.h"
using namespace AscendC;
// Debug control macro
constexpr uint16_t BufferNum_spec = 1;
constexpr int MyMAX_64 = 1 << 24; // 2^24
constexpr int MyMAX_64_small = 1 << 15; // 2^24
template <typename T>
class KernelLcm_broadcast_spec {
public:
    __aicore__ inline KernelLcm_broadcast_spec() {}
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
        max_size_for_broadcast = 1;
        if (this->inputSize != this->totalSize) {
            max_size_for_broadcast = max(this->inputSize, max_size_for_broadcast);
        }
        if (this->otherSize != this->totalSize) {
            max_size_for_broadcast = max(this->otherSize, max_size_for_broadcast);
        }
        inputGm.SetGlobalBuffer((__gm__ T *)input, totalSize);
        otherGm.SetGlobalBuffer((__gm__ T *)other, totalSize);
        outGm.SetGlobalBuffer((__gm__ T *)out, totalSize);
        uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(184 / 23) * 1024 / BufferNum_spec);
        this->n_elements_per_iter = 64;
        this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;

        // 广播张量最大个数：19K。
        //  tmpBuf最小 spaceSize*2
        pipe->InitBuffer(tmpBuf, max((uint32_t)(spaceSize * 20), (uint32_t)(max_size_for_broadcast * sizeof(T) + spaceSize)));
        pipe->InitBuffer(inputBuf, BufferNum_spec, spaceSize);
        pipe->InitBuffer(otherBuf, BufferNum_spec, spaceSize);
        pipe->InitBuffer(outBuf, BufferNum_spec, spaceSize);
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
        LocalTensor<int> inputIdxTensor = tmpBuf.GetWithOffset<int>(iterSize * 2, (max_size_for_broadcast * 8 + 31) / 32 * 32);
        CreateVecIndex(outputIdxTensor, (int)offset, iterSize);

        mapIndex(inputIdxTensor, outputIdxTensor, iterSize, inputShape, outputShape, nDims);

        LocalTensor<float> inputIdxTensor_fp32_tmp = tmpBuf.GetWithOffset<float>(iterSize, 0);
        Adds(inputIdxTensor_fp32_tmp, inputIdxTensor.template ReinterpretCast<float>(), (float)0.0, iterSize);

        uint32_t broadcastShapeIn[2] = {iterSize, 1};
        uint32_t broadcastShapeOut[2] = {iterSize, 2};
        LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.GetWithOffset<uint8_t>(iterSize * 8 * 4, iterSize * sizeof(int32_t));
        Broadcast<float, 2, 1>(inputIdxTensor.template ReinterpretCast<float>(), inputIdxTensor_fp32_tmp, broadcastShapeOut, broadcastShapeIn, sharedTmpBuffer);
        Muls(inputIdxTensor, inputIdxTensor, (int32_t)2, iterSize * 2);
        uint64_t mask[2] = {0xAAAAAAAAAAAAAAAA, 0};
        Adds(inputIdxTensor, inputIdxTensor, (int32_t)1, mask, iterSize * sizeof(int64_t) / 256, {1, 1, 8, 8});

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

        ShiftLeft(inputIdxTensor, inputIdxTensor, 2, iterSize * 2);

        Gather(inputLocal.template ReinterpretCast<int32_t>(), srcLocal.template ReinterpretCast<int32_t>(), inputIdxTensor.template ReinterpretCast<uint32_t>(), 0, iterSize * 2);
    }
    // 简化的欧几里得算法（避免双重循环）
    // 只适合小于 2^24 的整数
    __aicore__ inline void EuclideanGCD(const LocalTensor<int32_t> &outLocal, const LocalTensor<int64_t> &ALocal, const LocalTensor<int64_t> &BLocal, int validDigits, uint32_t iterSize) {
        LocalTensor<float> tmpA_fp32 = tmpBuf.GetWithOffset<float>(iterSize, 0);
        LocalTensor<float> tmpB_fp32 = tmpBuf.GetWithOffset<float>(iterSize, iterSize * 4);
        LocalTensor<uint8_t> maskCond = tmpBuf.GetWithOffset<uint8_t>(iterSize, iterSize * 8);
        LocalTensor<float> outLocal_fp32 = outLocal.template ReinterpretCast<float>();

        Cast(tmpA_fp32, ALocal, RoundMode::CAST_RINT, iterSize);
        Cast(tmpB_fp32, BLocal, RoundMode::CAST_RINT, iterSize);
        Abs(tmpA_fp32, tmpA_fp32, iterSize);
        Abs(tmpB_fp32, tmpB_fp32, iterSize);
        for (int i = 0; i < validDigits; ++i) {
            // b == 0 的检查
            CompareScalar(maskCond, tmpB_fp32.template ReinterpretCast<int32_t>(), (int32_t)0, CMPMODE::EQ, iterSize);

            // temp = a % b
            // 计算a/b
            Div(outLocal_fp32, tmpA_fp32, tmpB_fp32, iterSize);
            Cast(outLocal, outLocal_fp32, RoundMode::CAST_FLOOR, iterSize);
            Cast(outLocal_fp32, outLocal, RoundMode::CAST_NONE, iterSize);

            // 计算a/b*b
            Mul(outLocal_fp32, outLocal_fp32, tmpB_fp32, iterSize);
            // 计算a - a/b*b
            Sub(outLocal_fp32, tmpA_fp32, outLocal_fp32, iterSize);

            // a = b, b = temp (交换)
            Select(tmpA_fp32, maskCond, tmpA_fp32, tmpB_fp32, SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);
            Select(tmpB_fp32, maskCond, tmpB_fp32, outLocal_fp32, SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);
        }

        // 最终结果存储在 tmpA_fp32 中
        // 将结果转换为整数
        Cast(outLocal, tmpA_fp32, RoundMode::CAST_RINT, iterSize);
    }
    __aicore__ inline T gcd_binary(T a, T b) {
        if (a == 0) return b;
        if (b == 0) return a;

        int shift = ScalarGetSFFValue<1>((uint64_t)(a | b));  // 计算 a 和 b 的公共因子 2 的幂次
        a >>= ScalarGetSFFValue<1>((uint64_t)(a));  // 移除 a 的所有因子 2

        while (b != 0) {
            b >>= ScalarGetSFFValue<1>((uint64_t)(b));  // 移除 b 的所有因子 2
            if (a > b) {
                T tmp = a;
                a = b;
                b = tmp;
            }
            b -= a;  // 这里 b 一定大于等于 a
        }
        return a << shift;
    }
    __aicore__ inline void Abs_int(const LocalTensor<int32_t> &srcLocal, const LocalTensor<int32_t> &tmpLocal, uint32_t iterSize) {
        ShiftRight(tmpLocal.template ReinterpretCast<uint32_t>(), srcLocal.template ReinterpretCast<uint32_t>(), (uint32_t)31, iterSize);
        Muls(tmpLocal, tmpLocal, (int32_t)(-2), iterSize);
        Adds(tmpLocal, tmpLocal, (int32_t)1, iterSize); // 将负数转换为正数
        Mul(srcLocal, tmpLocal, srcLocal, iterSize);
    }
    // 会修改 aLocal、bLocal
    // tmpLocal用于存储中间结果
    // iterSize是每次处理的元素数量
    // 注意：outLocal的内存空间大小为 2 * iterSize * sizeof(int32_t)
    // tmpLocal的内存空间大小为 4 * iterSize * sizeof(int32_t)
    __aicore__ inline void Mul_int64(const LocalTensor<int64_t> &outLocal, const LocalTensor<int32_t> &aLocal, const LocalTensor<int32_t> &bLocal, const LocalTensor<int32_t> &tmpLocal,
                                     const uint32_t iterSize) {
        // printf("aLocal[0]: %d, bLocal[0]: %d, iterSize: %d\n", aLocal.GetValue(0), bLocal.GetValue(0), iterSize);
        LocalTensor<uint32_t> a0Local_uint32 = outLocal.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> a1Local_uint32 = aLocal.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> b0Local_uint32 = (outLocal.ReinterpretCast<uint32_t>())[iterSize];
        LocalTensor<uint32_t> b1Local_uint32 = bLocal.ReinterpretCast<uint32_t>();

        LocalTensor<uint32_t> p0Local_uint32 = tmpLocal[iterSize].ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> p1Local_uint32 = a0Local_uint32;
        LocalTensor<uint32_t> p2Local_uint32 = b0Local_uint32;
        LocalTensor<uint32_t> p3Local_uint32 = a1Local_uint32;
        LocalTensor<uint32_t> p12Local_uint32 = b1Local_uint32;
        // 要让lowLocal_uint32和highLocal_uint32连续存放,方便后面Gather
        LocalTensor<uint32_t> lowLocal_uint32 = tmpLocal[iterSize * 2].ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> highLocal_uint32 = tmpLocal[iterSize * 3].ReinterpretCast<uint32_t>();

        Duplicate(tmpLocal, (int32_t)0xFFFF, iterSize);
        And(a0Local_uint32.template ReinterpretCast<uint16_t>(), aLocal.template ReinterpretCast<uint16_t>(), tmpLocal.template ReinterpretCast<uint16_t>(), iterSize * 2);
        ShiftRight(a1Local_uint32, aLocal.template ReinterpretCast<uint32_t>(), (uint32_t)16, iterSize);
        And(b0Local_uint32.template ReinterpretCast<uint16_t>(), bLocal.template ReinterpretCast<uint16_t>(), tmpLocal.template ReinterpretCast<uint16_t>(), iterSize * 2);
        ShiftRight(b1Local_uint32, bLocal.template ReinterpretCast<uint32_t>(), (uint32_t)16, iterSize);
        Mul(p0Local_uint32.template ReinterpretCast<int32_t>(), a0Local_uint32.template ReinterpretCast<int32_t>(), b0Local_uint32.template ReinterpretCast<int32_t>(), iterSize);
        Mul(p1Local_uint32.template ReinterpretCast<int32_t>(), a0Local_uint32.template ReinterpretCast<int32_t>(), b1Local_uint32.template ReinterpretCast<int32_t>(), iterSize);
        Mul(p2Local_uint32.template ReinterpretCast<int32_t>(), a1Local_uint32.template ReinterpretCast<int32_t>(), b0Local_uint32.template ReinterpretCast<int32_t>(), iterSize);
        Mul(p3Local_uint32.template ReinterpretCast<int32_t>(), a1Local_uint32.template ReinterpretCast<int32_t>(), b1Local_uint32.template ReinterpretCast<int32_t>(), iterSize);
        Add(p12Local_uint32.template ReinterpretCast<int32_t>(), p1Local_uint32.template ReinterpretCast<int32_t>(), p2Local_uint32.template ReinterpretCast<int32_t>(), iterSize);
        //(p12 >> 16)
        ShiftLeft(lowLocal_uint32, p12Local_uint32, (uint32_t)16, iterSize);
        // low_part
        Add(lowLocal_uint32.template ReinterpretCast<int32_t>(), p0Local_uint32.template ReinterpretCast<int32_t>(), lowLocal_uint32.template ReinterpretCast<int32_t>(), iterSize);
        // printf("lowLocal_uint32[0]: %d\n", lowLocal_uint32.GetValue(0));
        // carry
        ShiftRight(p0Local_uint32, p0Local_uint32, (uint32_t)16, iterSize);
        // highLocal_uint32暂存(p1 & 0xFFFF)
        And(highLocal_uint32.template ReinterpretCast<uint16_t>(), p1Local_uint32.template ReinterpretCast<uint16_t>(), tmpLocal.template ReinterpretCast<uint16_t>(), iterSize * 2);
        // tmpLocal暂存(p2 & 0xFFFF)
        And(tmpLocal.template ReinterpretCast<uint16_t>(), p2Local_uint32.template ReinterpretCast<uint16_t>(), tmpLocal.template ReinterpretCast<uint16_t>(), iterSize * 2);
        ShiftRight(p12Local_uint32, p12Local_uint32, (uint32_t)16, iterSize);
        Add(highLocal_uint32.template ReinterpretCast<int32_t>(), p0Local_uint32.template ReinterpretCast<int32_t>(), highLocal_uint32.template ReinterpretCast<int32_t>(), iterSize);
        Add(highLocal_uint32.template ReinterpretCast<int32_t>(), highLocal_uint32.template ReinterpretCast<int32_t>(), tmpLocal.template ReinterpretCast<int32_t>(), iterSize);
        Add(highLocal_uint32.template ReinterpretCast<int32_t>(), highLocal_uint32.template ReinterpretCast<int32_t>(), p12Local_uint32.template ReinterpretCast<int32_t>(), iterSize);
        // printf("carry[0]: %d\n", highLocal_uint32.GetValue(0));
        // high_part
        ShiftRight(p1Local_uint32, p1Local_uint32, (uint32_t)16, iterSize);
        // printf("p1Local_uint32>>16: %d\n", p1Local_uint32.GetValue(0));
        ShiftRight(p2Local_uint32, p2Local_uint32, (uint32_t)16, iterSize);
        // printf("p2Local_uint32>>16: %d\n", p2Local_uint32.GetValue(0));
        ShiftRight(highLocal_uint32, highLocal_uint32, (uint32_t)16, iterSize);
        // printf("carry[0]>>16: %d\n", highLocal_uint32.GetValue(0));
        // printf("p3Local_uint32[0]: %d\n", p3Local_uint32.GetValue(0));
        Add(highLocal_uint32.template ReinterpretCast<int32_t>(), highLocal_uint32.template ReinterpretCast<int32_t>(), p3Local_uint32.template ReinterpretCast<int32_t>(), iterSize);
        Add(highLocal_uint32.template ReinterpretCast<int32_t>(), highLocal_uint32.template ReinterpretCast<int32_t>(), p2Local_uint32.template ReinterpretCast<int32_t>(), iterSize);
        Add(highLocal_uint32.template ReinterpretCast<int32_t>(), highLocal_uint32.template ReinterpretCast<int32_t>(), p1Local_uint32.template ReinterpretCast<int32_t>(), iterSize);
        // printf("highLocal_uint32[0]: %d\n", highLocal_uint32.GetValue(0));
        // 因为没有scatter，就只能生成index，然后用gather
        // 需要生成[0,0,1,1,....,N-1,N-1],考虑使用Broadcast实现N,1->N,2
        // 再用Adds 和mask实现生成[0,0+N,1,1+N,....,N-1,N-1+N]
        // 再用Gather实现合并成int64
        LocalTensor<int32_t> indexLocal = aLocal;
        LocalTensor<float> indexLocal_fp32 = indexLocal.template ReinterpretCast<float>();
        // 用前2*iterSize个元素
        LocalTensor<int32_t> indexLocal_broadcast = tmpLocal;
        LocalTensor<float> indexLocal_broadcast_fp32 = indexLocal_broadcast.template ReinterpretCast<float>();
        LocalTensor<uint8_t> sharedTmpBuffer = outLocal.template ReinterpretCast<uint8_t>();
        CreateVecIndex(indexLocal, 0, iterSize);
        uint32_t broadcastShapeIn[2] = {iterSize, 1};
        uint32_t broadcastShapeOut[2] = {iterSize, 2};
        Broadcast<float, 2, 1>(indexLocal_broadcast_fp32, indexLocal_fp32, broadcastShapeOut, broadcastShapeIn, sharedTmpBuffer);
        uint64_t mask[2] = {0xAAAAAAAAAAAAAAAA, 0};
        Adds(indexLocal_broadcast, indexLocal_broadcast, (int32_t)iterSize, mask, iterSize * sizeof(int64_t) / 256, {1, 1, 8, 8});

        // 按字节偏移，不要忘记计算个数乘以4
        ShiftLeft(indexLocal_broadcast, indexLocal_broadcast, 2, iterSize * 2);

        Gather(outLocal.template ReinterpretCast<int32_t>(), tmpLocal[iterSize * 2], indexLocal_broadcast.template ReinterpretCast<uint32_t>(), 0, iterSize * 2);
        // printf("outLocal[0]: %d\n", outLocal.GetValue(0));
    }
    __aicore__ inline void Process_iter(uint32_t offset, uint32_t iterSize) {// out绝对偏移量,以元素为单位

        LocalTensor<T> inputLocal = inputBuf.AllocTensor<T>();
        LocalTensor<T> otherLocal = otherBuf.AllocTensor<T>();

        // printf("Process_iter offset %d, iterSize %d\n", offset, iterSize);
        if (inputSize != totalSize) {
            CopyInBroadcast(inputGm, inputLocal, offset, iterSize, mmInputDims, mmOutputDims, nOutputDims, inputSize);
        } else {
            DataCopy(inputLocal, inputGm[offset], iterSize);
        }
        inputBuf.EnQue<T>(inputLocal);
        inputLocal = inputBuf.DeQue<T>();
        if (otherSize != totalSize) {
            CopyInBroadcast(otherGm, otherLocal, offset, iterSize, mmOtherDims, mmOutputDims, nOutputDims, otherSize);
        } else {
            DataCopy(otherLocal, otherGm[offset], iterSize);
        }
        otherBuf.EnQue<T>(otherLocal);
        otherLocal = otherBuf.DeQue<T>();

        LocalTensor<float> inputLocal_fp32 = tmpBuf.GetWithOffset<float>(iterSize, 0);
        LocalTensor<float> otherLocal_fp32 = tmpBuf.GetWithOffset<float>(iterSize, iterSize * 4);
        LocalTensor<int32_t> outLocal_int32 = tmpBuf.GetWithOffset<int32_t>(iterSize, iterSize * 12);
        LocalTensor<int64_t> outLocal = outBuf.AllocTensor<T>();
        Cast(inputLocal_fp32, inputLocal, RoundMode::CAST_RINT, iterSize);
        Abs(inputLocal_fp32, inputLocal_fp32, iterSize);
        ReduceMax<float>(inputLocal_fp32, inputLocal_fp32, outLocal_int32.template ReinterpretCast<float>(), iterSize, false);

        Cast(otherLocal_fp32, otherLocal, RoundMode::CAST_RINT, iterSize);
        Abs(otherLocal_fp32, otherLocal_fp32, iterSize);
        ReduceMax<float>(otherLocal_fp32, otherLocal_fp32, outLocal_int32.template ReinterpretCast<float>(), iterSize, false);

        int64_t input_max = (int64_t)inputLocal_fp32.GetValue(0);
        int64_t other_max = (int64_t)otherLocal_fp32.GetValue(0);
        int64_t current_max = max(input_max, other_max);
        if (current_max <= MyMAX_64) {
            int64_t LeadingZeroCount = ScalarCountLeadingZero((uint64_t)current_max);
            int32_t validDigits = (int32_t)((64.0f - LeadingZeroCount) * 1.44f) + 1;
            EuclideanGCD(outLocal_int32, inputLocal, otherLocal, validDigits, iterSize);

            LocalTensor<int32_t> inputLocal_int32 = inputLocal.template ReinterpretCast<int32_t>();
            LocalTensor<int32_t> otherLocal_int32 = otherLocal.template ReinterpretCast<int32_t>();
            Cast(inputLocal_int32, inputLocal, RoundMode::CAST_NONE, iterSize);
            Cast(otherLocal_int32, otherLocal, RoundMode::CAST_NONE, iterSize);
                // 计算最小公倍数
            LocalTensor<float> tmpA_fp32 = tmpBuf.GetWithOffset<float>(iterSize, 0);
            LocalTensor<float> tmpGcd_fp32 = tmpBuf.GetWithOffset<float>(iterSize, iterSize * 4);
            LocalTensor<uint8_t> maskCond = tmpBuf.GetWithOffset<uint8_t>(iterSize, iterSize * 8);
            CompareScalar(maskCond, outLocal_int32.template ReinterpretCast<int32_t>(), (int32_t)0, CMPMODE::EQ, iterSize);
            Not(maskCond.template ReinterpretCast<int16_t>(), maskCond.template ReinterpretCast<int16_t>(), iterSize / 2);
                // 将输入转换为浮点数
            Cast(tmpA_fp32, inputLocal_int32, RoundMode::CAST_NONE, iterSize);
            Cast(tmpGcd_fp32, outLocal_int32, RoundMode::CAST_NONE, iterSize);

                // 计算 a / gcd
            Div(tmpA_fp32, tmpA_fp32, tmpGcd_fp32, iterSize);
                // 将结果转换回整数
            Cast(inputLocal_int32, tmpA_fp32, RoundMode::CAST_FLOOR, iterSize);
            if (current_max <= MyMAX_64_small) {
                // 计算最小公倍数
                Mul(outLocal_int32, inputLocal_int32, otherLocal_int32, iterSize);

                // 使用选择操作处理 GCD 为 0 的情况，有必要吗？0/0等于？
                Select(outLocal_int32.template ReinterpretCast<float>(), maskCond, outLocal_int32.template ReinterpretCast<float>(), static_cast<float>(+0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                       iterSize);
                Abs_int(outLocal_int32, inputLocal.template ReinterpretCast<int32_t>(), iterSize);
                Cast(outLocal, outLocal_int32, RoundMode::CAST_NONE, iterSize);
            } else {
                Abs_int(inputLocal_int32, outLocal.template ReinterpretCast<int32_t>(), iterSize);
                Abs_int(otherLocal_int32, outLocal.template ReinterpretCast<int32_t>(), iterSize);
                LocalTensor<int64_t> outLocal_int64 = outLocal.template ReinterpretCast<int64_t>();
                Mul_int64(outLocal_int64, inputLocal_int32, otherLocal_int32, tmpBuf.GetWithOffset<int32_t>(iterSize * 4, 0), iterSize);
            }
        } else {
            // 计算GCD
            for (uint32_t i = 0; i < iterSize; ++i) {
                T a = inputLocal.GetValue(i);
                T b = otherLocal.GetValue(i);
                if (a < 0) a = -a;
                if (b < 0) b = -b;
                // 计算GCD
                T gcd = gcd_binary(a, b);
                // 计算最小公倍数
                if (gcd == 0) {
                    outLocal.SetValue(i, 0); // 如果GCD为0，结果为0
                } else {
                    T lcm = (a / gcd) * b; // 使用整数除法避免溢出
                    if (lcm < 0) {
                        lcm = -lcm; // 确保结果为非负数
                    }
                    outLocal.SetValue(i, lcm); // 计算最小公倍数
                }
            }
        }

        inputBuf.FreeTensor<T>(inputLocal);
        otherBuf.FreeTensor<T>(otherLocal);
        // printf("Process_iter offset %d, iterSize %d, outLocal[0] %p\n", offset, iterSize, outLocal(0));
        outBuf.EnQue<T>(outLocal);
        outLocal = outBuf.DeQue<T>();
        DataCopyExtParams storeParams{1, (uint32_t)(iterSize * sizeof(T)), 0, 0, 0};
        DataCopyPad(outGm[offset], outLocal, storeParams);
        DataCopyPad(outGm[offset], outLocal, storeParams);
        outBuf.FreeTensor<T>(outLocal);
    }
    __aicore__ inline void Process() {
        uint32_t blockBeginIndex = 0;
        for (uint32_t i = 0; i < smallLoopTimes; ++i) {
            uint32_t iterSize = n_elements_per_iter;
            uint32_t offset = blockBeginIndex + beginIndex;
            Process_iter(offset, iterSize); // 绝对偏移量
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
    uint32_t max_size_for_broadcast;
};
