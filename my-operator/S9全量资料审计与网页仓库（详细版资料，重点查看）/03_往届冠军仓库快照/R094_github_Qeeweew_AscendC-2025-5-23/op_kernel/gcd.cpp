#include "kernel_operator.h"

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define REP 50

template<typename T>
__aicore__ inline T gcd_1(T a, T b) {
    a = (a < 0) ? -a : a;
    b = (b < 0) ? -b : b;

    if (a == 0) return b;
    if (b == 0) return a;

    while (b != 0) {
        T temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

template<typename T>
__aicore__ inline T gcd_2(T a, T b) {
    while (b != 0) {
        T temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

#define TILE_SIZE 4096

using namespace AscendC;

template <typename T>
class KernelGcd
{
public:
    __aicore__ inline KernelGcd()
    {
    }
    __aicore__ inline void Init(int N0, int N1, int N2, int N3, int N4, int broadcast_mask, TPipe *pipe, GM_ADDR x1, GM_ADDR x2, GM_ADDR y)
    {
        N[0] = N0;
        N[1] = N1;
        N[2] = N2;
        N[3] = N3;
        N[4] = N4;
        // printf("%d %d %d %d %d\n", N[0], N[1], N[2], N[3], N[4]);
        for (int i = 0; i < 5;i++) {
            if (broadcast_mask & (1<<i)) {
                M[i] = 1;
            } else {
                M[i] = N[i];
            }
        }
        sizeX1 = 1;
        sizeX2 = 1;
        for (int i = 0;i < 5;i++) {
            sizeX1 *= N[i];
            sizeX2 *= M[i];
        }
        x1Gm.SetGlobalBuffer((__gm__ T *)x1, sizeX1 * sizeof(T));
        x2Gm.SetGlobalBuffer((__gm__ T *)x2, sizeX2 * sizeof(T));
        yGm.SetGlobalBuffer((__gm__ T *)y, sizeX1 * sizeof(T));

        if constexpr(std::is_same<T, int16_t>::value) {
            pipe->InitBuffer(tBufNext, 2 * TILE_SIZE * sizeof(int16_t));
            pipe->InitBuffer(tBufMask, 2 * TILE_SIZE * sizeof(uint8_t));
        }

        if constexpr(std::is_same<T, int32_t>::value) {
            pipe->InitBuffer(tBufNext, 2 * TILE_SIZE * sizeof(int32_t));
            pipe->InitBuffer(tBufMask, 2 * TILE_SIZE * sizeof(uint8_t));
        }

        pipe->InitBuffer(inX1, 1, TILE_SIZE * sizeof(T));
        pipe->InitBuffer(inX2, 1, TILE_SIZE * sizeof(T));
        pipe->InitBuffer(outY, 1, TILE_SIZE * sizeof(T));
    }

    __aicore__ inline void CopyInX1(int64_t offset, int len) {
        LocalTensor<T> x1 = inX1.AllocTensor<T>();
        DataCopyExtParams copyParamsX;
        copyParamsX.blockCount = 1;
        copyParamsX.blockLen = len * sizeof(T);
        copyParamsX.srcStride = 0;
        copyParamsX.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x1, x1Gm[offset], copyParamsX, padParams);
        inX1.EnQue(x1);
    }

    __aicore__ inline void CopyInX2(int64_t offset, int len) {
        LocalTensor<T> x2 = inX2.AllocTensor<T>();
        DataCopyExtParams copyParamsX;
        copyParamsX.blockCount = 1;
        copyParamsX.blockLen = len * sizeof(T);
        copyParamsX.srcStride = 0;
        copyParamsX.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x2, x2Gm[offset], copyParamsX, padParams);
        inX2.EnQue(x2);
    }

    __aicore__ inline void CopyOut(int64_t offset, int len) {
        LocalTensor<T> y = outY.DeQue<T>();
        DataCopyExtParams copyParamsX;
        copyParamsX.blockCount = 1;
        copyParamsX.blockLen = len * sizeof(T);
        copyParamsX.srcStride = 0;
        copyParamsX.dstStride = 0;
        DataCopyPad(yGm[offset], y, copyParamsX);
        outY.FreeTensor(y);
    }

    __aicore__ inline void GcdLikely16(const LocalTensor<int16_t>& c, const LocalTensor<int16_t>& a, const LocalTensor<int16_t>& b) {
        auto n_a = tBufNext.Get<int16_t>();
        auto n_b = tBufNext.Get<int16_t>()[TILE_SIZE];
        auto mask = tBufMask.Get<uint8_t>();
        auto mask1 = tBufMask.Get<uint8_t>()[TILE_SIZE];
        auto a_h = a.ReinterpretCast<half>();
        auto b_h = b.ReinterpretCast<half>();
        auto n_a_h = n_a.ReinterpretCast<half>();
        auto n_b_h = n_b.ReinterpretCast<half>();

        Not(n_a, a, TILE_SIZE);
        Not(n_b, b, TILE_SIZE);
        Adds(n_a, n_a, (int16_t) 1, TILE_SIZE);
        Adds(n_b, n_b, (int16_t) 1, TILE_SIZE);
        Max(a, a, n_a, TILE_SIZE);
        Max(b, b, n_b, TILE_SIZE);

        for (int i = 0;i < REP;i++) {
            Min(n_a, a, b, TILE_SIZE);
            Max(n_b, a, b, TILE_SIZE);
            Sub(n_b, n_b, n_a, TILE_SIZE);
            CompareScalar(mask, b_h, (half)0.0f, CMPMODE::EQ, TILE_SIZE);
            Select(a_h, mask, a_h, n_a_h, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
            Select(b_h, mask, b_h, n_b_h, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
        }

        Max(n_a, a, b, TILE_SIZE);
        Min(n_b, a, b, TILE_SIZE);
        DataCopy(a, n_a, TILE_SIZE);
        DataCopy(b, n_b, TILE_SIZE);

        int16_t tmp = 1;
        half* tmp_h = (half*) &tmp;
        CompareScalar(mask, b_h, *tmp_h, CMPMODE::NE, TILE_SIZE);
        Select(a_h, mask, a_h, *tmp_h, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE_SIZE);
        Select(b_h, mask, b_h, (half)0.0f, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE_SIZE);

        DataCopy(c, a, TILE_SIZE);

        CompareScalar(mask, b_h, (half)0.0f, CMPMODE::EQ, TILE_SIZE);

        for (int i = 0;i < TILE_SIZE;i+=8) {
            uint8_t bit = mask.GetValue(i / 8);
            if (bit == 0) continue;
            for (int j = 0;j < 8;j++) {
                if ((bit & (1<<j)) == 0) {
                    c.SetValue(i + j, gcd_2(a.GetValue(i + j), b.GetValue(i + j)));
                }
            }
        }
    }

    __aicore__ inline void GcdLikely32(const LocalTensor<int32_t>& c, const LocalTensor<int32_t>& a, const LocalTensor<int32_t>& b) {
        auto n_a = tBufNext.Get<int32_t>();
        auto n_b = tBufNext.Get<int32_t>()[TILE_SIZE];
        auto mask = tBufMask.Get<uint8_t>();
        auto mask1 = tBufMask.Get<uint8_t>()[TILE_SIZE];
        auto a_h = a.ReinterpretCast<float>();
        auto b_h = b.ReinterpretCast<float>();
        auto n_a_h = n_a.ReinterpretCast<float>();
        auto n_b_h = n_b.ReinterpretCast<float>();

        Not(n_a.ReinterpretCast<uint16_t>(), a.ReinterpretCast<uint16_t>(), 2 * TILE_SIZE);
        Not(n_b.ReinterpretCast<uint16_t>(), b.ReinterpretCast<uint16_t>(), 2 * TILE_SIZE);
        Adds(n_a, n_a, (int32_t) 1, TILE_SIZE);
        Adds(n_b, n_b, (int32_t) 1, TILE_SIZE);
        Max(a, a, n_a, TILE_SIZE);
        Max(b, b, n_b, TILE_SIZE);

        for (int i = 0;i < REP;i++) {
            Min(n_a, a, b, TILE_SIZE);
            Max(n_b, a, b, TILE_SIZE);
            Sub(n_b, n_b, n_a, TILE_SIZE);
            CompareScalar(mask, b_h, (float)0.0f, CMPMODE::EQ, TILE_SIZE);
            Select(a_h, mask, a_h, n_a_h, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
            Select(b_h, mask, b_h, n_b_h, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
        }

        Max(n_a, a, b, TILE_SIZE);
        Min(n_b, a, b, TILE_SIZE);
        DataCopy(a, n_a, TILE_SIZE);
        DataCopy(b, n_b, TILE_SIZE);

        int32_t tmp = 1;
        float* tmp_h = (float*) &tmp;
        CompareScalar(mask, b_h, *tmp_h, CMPMODE::NE, TILE_SIZE);
        Select(a_h, mask, a_h, *tmp_h, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE_SIZE);
        Select(b_h, mask, b_h, (float)0.0f, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE_SIZE);

        DataCopy(c, a, TILE_SIZE);

        CompareScalar(mask, b_h, (float)0.0f, CMPMODE::EQ, TILE_SIZE);

        for (int i = 0;i < TILE_SIZE;i+=8) {
            uint8_t bit = mask.GetValue(i / 8);
            if (bit == 0) continue;
            for (int j = 0;j < 8;j++) {
                if ((bit & (1<<j)) == 0) {
                    c.SetValue(i + j, gcd_2(a.GetValue(i + j), b.GetValue(i + j)));
                }
            }
        }
    }

    __aicore__ inline void Compute() {
        auto x1 = inX1.DeQue<T>();
        auto x2 = inX2.DeQue<T>();
        auto y = outY.AllocTensor<T>();
        if constexpr(std::is_same<T, int16_t>::value) {
            GcdLikely16(y, x1, x2);
        } else if constexpr(std::is_same<T, int32_t>::value) {
            GcdLikely32(y, x1, x2);
        } else {
            for (int i = 0;i < TILE_SIZE;i++) {
                y.SetValue(i, gcd_1(x1.GetValue(i), x2.GetValue(i)));
            }
        }

        outY.EnQue(y);
        inX1.FreeTensor(x1);
        inX2.FreeTensor(x2);
    }

    __aicore__ inline void ProcessFast()
    {
        for (int64_t i = GetBlockIdx() * TILE_SIZE;i < sizeX1;i+=TILE_SIZE * GetBlockNum()) {
            CopyInX1(i, MIN(sizeX1 - i, TILE_SIZE));
            CopyInX2(i, MIN(sizeX1 - i, TILE_SIZE));
            Compute();
            CopyOut(i, MIN(sizeX1 - i, TILE_SIZE));
        }
    }

    __aicore__ inline void ProcessSlow()
    {

        if (GetBlockIdx() != 0) return;
        // 预计算步长数组
        int64_t y_stride[5], x2_stride[5];
        y_stride[4] = 1;        // 最内层维度步长为1
        x2_stride[4] = 1;

        for (int i = 3; i >= 0; i--) {
            y_stride[i] = y_stride[i+1] * N[i+1];
            x2_stride[i] = x2_stride[i+1] * M[i+1];
        }
        // for (int i = 0; i < 5;i++) {
        //     printf("y_stride[%d] = %d\n", i, y_stride[i]);
        //     printf("x2_stride[%d] = %d\n", i, x2_stride[i]);
        // }

        // 五维循环
        for (int i0 = 0; i0 < N[0]; i0++) {
            for (int i1 = 0; i1 < N[1]; i1++) {
                for (int i2 = 0; i2 < N[2]; i2++) {
                    for (int i3 = 0; i3 < N[3]; i3++) {
                        for (int i4 = 0; i4 < N[4]; i4++) {
                            // 计算y的线性索引
                            int64_t y_idx = i0*y_stride[0] 
                                    + i1*y_stride[1]
                                    + i2*y_stride[2]
                                    + i3*y_stride[3]
                                    + i4*y_stride[4];
                            
                            // 计算x2的广播索引
                            int64_t x2_idx = (M[0]>1?i0:0)*x2_stride[0]
                                    + (M[1]>1?i1:0)*x2_stride[1]
                                    + (M[2]>1?i2:0)*x2_stride[2]
                                    + (M[3]>1?i3:0)*x2_stride[3]
                                    + (M[4]>1?i4:0)*x2_stride[4];
                            
                            // 计算并存储结果
                            T x1_val = x1Gm.GetValue(y_idx);
                            T x2_val = x2Gm.GetValue(x2_idx);
                            yGm.SetValue(y_idx, gcd_1(x1_val, x2_val));
                            // y[y_idx] = gcd(x1[y_idx], x2[x2_idx]);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void Process() {
        if (sizeX1 == sizeX2) {
            ProcessFast();
        } else {
            ProcessSlow();
        }
    }

    TQue<QuePosition::VECIN, 1> inX1, inX2;
    TQue<QuePosition::VECOUT, 1> outY;
    TBuf<TPosition::VECCALC> tBufNext, tBufMask;

    int64_t N[5], M[5], sizeX1, sizeX2;
    GlobalTensor<T> x1Gm, x2Gm, yGm;
};


extern "C" __global__ __aicore__ void gcd(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGcd<DTYPE_X1> op;
    TPipe pipe;
    op.Init(
        tiling_data.N0, tiling_data.N1, tiling_data.N2, tiling_data.N3, tiling_data.N4, tiling_data.broadcast_mask, &pipe,
        x1, x2, y
    );
    op.Process();
}