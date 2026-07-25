#include "kernel_operator.h"
using namespace AscendC;
constexpr uint16_t BufferNum_32 = 1;
constexpr int MyMAX = 1 << 24; // 2^24
template <typename T>
class KernelLcm_32 {
public:
    __aicore__ inline KernelLcm_32() {}
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

        uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(180 / 5) * 1024 / BufferNum_32);
        this->n_elements_per_iter = spaceSize / sizeof(T);
        this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;

        pipe->InitBuffer(inputBuf, BufferNum_32, spaceSize + 32);
        pipe->InitBuffer(otherBuf, BufferNum_32, spaceSize + 32);
        pipe->InitBuffer(outBuf, BufferNum_32, spaceSize + 32);

        pipe->InitBuffer(tmpBuf, n_elements_per_iter * 9 + 32);
    }
    // 简化的欧几里得算法（避免双重循环）
    // 只适合小于 2^24 的整数
    __aicore__ inline void EuclideanGCD(const LocalTensor<int> &outLocal, const LocalTensor<int> &ALocal, const LocalTensor<int> &BLocal, int validDigits, uint32_t iterSize) {
        LocalTensor<float> tmpA_fp32 = tmpBuf.GetWithOffset<float>(iterSize, 0);
        LocalTensor<float> tmpB_fp32 = tmpBuf.GetWithOffset<float>(iterSize, iterSize * 4);
        LocalTensor<uint8_t> maskCond = tmpBuf.GetWithOffset<uint8_t>(iterSize, iterSize * 8);
        LocalTensor<float> outLocal_fp32 = outLocal.template ReinterpretCast<float>();

        Cast(tmpA_fp32, ALocal, RoundMode::CAST_NONE, iterSize);
        Cast(tmpB_fp32, BLocal, RoundMode::CAST_NONE, iterSize);

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
    __aicore__ inline void Abs_int(const LocalTensor<T> &srcLocal, const LocalTensor<T> &tmpLocal, uint32_t iterSize) {
        if constexpr (std::is_same<T, int32_t>::value) {
            ShiftRight(tmpLocal.template ReinterpretCast<uint32_t>(), srcLocal.template ReinterpretCast<uint32_t>(), (uint32_t)31, iterSize);
            Muls(tmpLocal, tmpLocal, (int32_t)(-2), iterSize);
            Adds(tmpLocal, tmpLocal, (int32_t)1, iterSize); // 将负数转换为正数
            Mul(srcLocal, tmpLocal, srcLocal, iterSize);
        } else if constexpr (std::is_same<T, int16_t>::value) {
            ShiftRight(tmpLocal.template ReinterpretCast<uint16_t>(), srcLocal.template ReinterpretCast<uint16_t>(), (uint16_t)15, iterSize);
            Muls(tmpLocal, tmpLocal, (int16_t)(-2), iterSize);
            Adds(tmpLocal, tmpLocal, (int16_t)1, iterSize); // 将负数转换为正数
            Mul(srcLocal, tmpLocal, srcLocal, iterSize);
        }
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
        Abs_int(inputLocal, outLocal, iterSize);
        Abs_int(otherLocal, outLocal, iterSize);

        LocalTensor<float> inputLocal_fp32 = tmpBuf.GetWithOffset<float>(iterSize, 0);
        LocalTensor<float> otherLocal_fp32 = tmpBuf.GetWithOffset<float>(iterSize, iterSize * 4);

        Cast(inputLocal_fp32, inputLocal, RoundMode::CAST_NONE, iterSize);
        ReduceMax<float>(inputLocal_fp32, inputLocal_fp32, outLocal.template ReinterpretCast<float>(), iterSize, false);

        Cast(otherLocal_fp32, otherLocal, RoundMode::CAST_NONE, iterSize);
        ReduceMax<float>(otherLocal_fp32, otherLocal_fp32, outLocal.template ReinterpretCast<float>(), iterSize, false);

        int input_max = (int)inputLocal_fp32.GetValue(0);
        int other_max = (int)otherLocal_fp32.GetValue(0);
        int current_max = max(input_max, other_max);
        if (current_max <= MyMAX) {
            int64_t LeadingZeroCount = ScalarCountLeadingZero((uint64_t)current_max);
            int32_t validDigits = (int32_t)((64.0f - LeadingZeroCount) * 1.44f) + 1;
            EuclideanGCD(outLocal, inputLocal, otherLocal, validDigits, iterSize);

            // 计算最小公倍数
            LocalTensor<float> tmpA_fp32 = tmpBuf.GetWithOffset<float>(iterSize, 0);
            LocalTensor<float> tmpGcd_fp32 = tmpBuf.GetWithOffset<float>(iterSize, iterSize * 4);
            LocalTensor<uint8_t> maskCond = tmpBuf.GetWithOffset<uint8_t>(iterSize, iterSize * 8);
            CompareScalar(maskCond, outLocal.template ReinterpretCast<int32_t>(), (int32_t)0, CMPMODE::EQ, iterSize);
            Not(maskCond.template ReinterpretCast<int16_t>(), maskCond.template ReinterpretCast<int16_t>(), iterSize / 2);
            // 将输入转换为浮点数
            Cast(tmpA_fp32, inputLocal, RoundMode::CAST_NONE, iterSize);
            Cast(tmpGcd_fp32, outLocal, RoundMode::CAST_NONE, iterSize);

            // 计算 a / gcd
            Div(tmpA_fp32, tmpA_fp32, tmpGcd_fp32, iterSize);
            // 将结果转换回整数
            Cast(inputLocal, tmpA_fp32, RoundMode::CAST_FLOOR, iterSize);

            // 计算最小公倍数
            Mul(outLocal, inputLocal, otherLocal, iterSize);

            // 使用选择操作处理 GCD 为 0 的情况
            Select(outLocal.template ReinterpretCast<float>(), maskCond, outLocal.template ReinterpretCast<float>(), static_cast<float>(+0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, iterSize);
        } else {
            // 计算GCD
            for (uint32_t i = 0; i < iterSize; ++i) {
                T a = inputLocal.GetValue(i);
                T b = otherLocal.GetValue(i);
                // 计算GCD
                T gcd = gcd_binary(a, b);
                // 计算最小公倍数
                if (gcd == 0) {
                    outLocal.SetValue(i, 0); // 如果GCD为0，结果为0
                } else {
                    T lcm = (a / gcd) * b; // 使用整数除法避免溢出
                    outLocal.SetValue(i, lcm);  // 计算最小公倍数
                }
            }
        }

        Abs_int(outLocal, inputLocal, iterSize);
        inputBuf.FreeTensor<T>(inputLocal);
        otherBuf.FreeTensor<T>(otherLocal);

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
            Process_iter(blockBeginIndex, n_elements_per_iter);
            blockBeginIndex += n_elements_per_iter;
        }
    }

private:
    TPipe *pipe;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> otherGm;
    GlobalTensor<T> outGm;

    TQue<QuePosition::VECIN, BufferNum_32> inputBuf;
    TQue<QuePosition::VECIN, BufferNum_32> otherBuf;
    TQue<QuePosition::VECOUT, BufferNum_32> outBuf;

    TBuf<QuePosition::VECCALC> tmpBuf;
    uint32_t size;
    uint32_t smallLoopTimes;
    uint32_t n_elements_per_iter;
};