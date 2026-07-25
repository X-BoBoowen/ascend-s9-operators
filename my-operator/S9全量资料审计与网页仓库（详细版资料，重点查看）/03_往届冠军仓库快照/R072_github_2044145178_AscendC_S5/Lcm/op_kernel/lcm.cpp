#include "kernel_operator.h"
#include "lcm_broadcast_sca.h"
#include "lcm_broadcast_spec.h"
#include "lcm_int32.h"
using namespace AscendC;
constexpr uint16_t BufferNum_k = 1;
template <typename T>
class KernelLcm {
public:
    __aicore__ inline KernelLcm() {}
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

        pipe->InitBuffer(inputBuf, BufferNum_k, spaceSize);
        pipe->InitBuffer(otherBuf, BufferNum_k, spaceSize);
        pipe->InitBuffer(outBuf, BufferNum_k, spaceSize);
    }

    __aicore__ inline T gcd_Euclidean(T a, T b) {
        // 处理 a=0 或 b=0 的情况
        if (a == 0) return b;
        if (b == 0) return a;
        while (b != 0) {
            T tmp = a;
            a = b;
            b = tmp % b;
        }
        return a;
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
        if constexpr (std::is_same<T, int16_t>::value || std::is_same<T, int32_t>::value) {
            Abs_int(inputLocal, outLocal, iterSize);
            Abs_int(otherLocal, outLocal, iterSize);
        }
        // 计算GCD
        for (uint32_t i = 0; i < iterSize; ++i) {
            T a = inputLocal.GetValue(i);
            T b = otherLocal.GetValue(i);
            if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, int64_t>::value) {
                if (a < 0) a = -a;
                if (b < 0) b = -b;
            }
            // 计算GCD
            T gcd = gcd_binary(a, b);
            // 计算最小公倍数
            if (gcd == 0) {
                outLocal.SetValue(i, 0); // 如果GCD为0，结果为0
            } else {
                T lcm = (a / gcd) * b; // 使用整数除法避免溢出
                if constexpr (std::is_same<DTYPE_INPUT, int64_t>::value) {
                    if (lcm < 0) {
                        lcm = -lcm; // 确保结果为非负数
                    }
                }
                outLocal.SetValue(i, lcm);  // 计算最小公倍数
            }
        }
        if constexpr (std::is_same<DTYPE_INPUT, int32_t>::value) {
            // 对结果进行绝对值处理
            Abs_int(outLocal, inputLocal, iterSize);
        }
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

    TQue<QuePosition::VECIN, BufferNum_k> inputBuf;
    TQue<QuePosition::VECIN, BufferNum_k> otherBuf;
    TQue<QuePosition::VECOUT, BufferNum_k> outBuf;

    uint32_t size;
    uint32_t smallLoopTimes;
    uint32_t n_elements_per_iter;
};
extern "C" __global__ __aicore__ void lcm(GM_ADDR input, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        if (std::is_same<DTYPE_INPUT, int32_t>::value) {
            KernelLcm_32<int32_t> kernel;
            kernel.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, &pipe);
            kernel.Process();
        } else {
            KernelLcm<DTYPE_INPUT> kernel;
            kernel.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, &pipe);
            kernel.Process();
        }
    } else if (TILING_KEY_IS(2)) {
        KernelLcm_broadcast<DTYPE_INPUT> kernel;
        kernel.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.totalSize, tiling_data.mmInputDims, tiling_data.mmOtherDims,
                    tiling_data.mmOutputDims, tiling_data.nOutputDims, &pipe);
        kernel.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelLcm_broadcast_spec<int64_t> kernel;
        kernel.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.totalSize, tiling_data.mmInputDims, tiling_data.mmOtherDims,
                    tiling_data.mmOutputDims, tiling_data.nOutputDims, &pipe);
        kernel.Process();
    }
}