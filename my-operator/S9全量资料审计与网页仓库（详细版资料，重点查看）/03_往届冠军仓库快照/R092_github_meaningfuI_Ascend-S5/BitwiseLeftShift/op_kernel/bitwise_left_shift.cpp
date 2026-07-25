#include "kernel_operator.h"

using namespace AscendC;
#define MAX_DIM_NUMBER 3

template<typename T> class LeftShiftScalar {
    public:
        __aicore__ inline LeftShiftScalar() {}
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, int32_t size, int32_t length) {

            unsigned L = GetBlockIdx() * length;
            unsigned R = (GetBlockIdx() + 1) * length;
            if (L > size) {
                L = size;
            }
            if (R > size) {
                R = size;
            }
            x1Gm.SetGlobalBuffer((__gm__ T*)x1);
            x2Gm.SetGlobalBuffer((__gm__ T*)x2);
            yGm.SetGlobalBuffer((__gm__ T*)y);
            // x1Gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            // x2Gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            // yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            this->L = L;
            this->R = R;
        }
        __aicore__ inline void Process() {
            for (int i = L; i < R; i++) {
                T a = x1Gm.GetValue(i), b = x2Gm.GetValue(i);
                yGm.SetValue(i, a << b);
            }
        }
    
    private:
        GlobalTensor<T> x1Gm, x2Gm, yGm;
        int32_t L, R;
};

constexpr int32_t BUFFER_NUM = 2;
template<typename T> class LeftShiftVector {
    public:
        __aicore__ inline LeftShiftVector() {}
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, int32_t size, int32_t length, TPipe* pi) {
            pipe = pi;

            x1Gm.SetGlobalBuffer((__gm__ T*)x1);
            x2Gm.SetGlobalBuffer((__gm__ T*)x2);
            yGm.SetGlobalBuffer((__gm__ T*)y);
            // x1Gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            // x2Gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            // yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            this->L = GetBlockIdx() * LEN;
            this->R = size;
            pipe->InitBuffer(inQueueX1, BUFFER_NUM, LEN * sizeof(T));
            pipe->InitBuffer(inQueueX2, BUFFER_NUM, LEN * sizeof(T));
            pipe->InitBuffer(outQueueY, BUFFER_NUM, LEN * sizeof(T));

            pipe->InitBuffer(X1Buf, LEN * sizeof(int32_t));
            pipe->InitBuffer(X2Buf, LEN * sizeof(int32_t));
            if constexpr(std::is_same_v<T, int8_t>)
            {
                pipe->InitBuffer(tmpBuf, LEN * sizeof(int32_t));
            }
        }
        __aicore__ inline void Process() {
            int i;
            // AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>(LEN);
            // AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(LEN);
            for (i = L; i < R; i += LEN * GetBlockNum()) {
                int t;
                t = (((R-i)<LEN)?R-i:LEN);
                if (t<32/sizeof(T)) t=32/sizeof(T);
                if (t<8) t=8;
                CopyIn(i,t);
                Compute(i,t);
                CopyOut(i,t);
            }
        }
    
    private:
        __aicore__ inline void CopyIn(int32_t progress, int l)
        {
            AscendC::LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
            AscendC::LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
            AscendC::DataCopy(x1Local, x1Gm[progress], l);
            AscendC::DataCopy(x2Local, x2Gm[progress], l);
            inQueueX1.EnQue(x1Local);
            inQueueX2.EnQue(x2Local);
        }
        __aicore__ inline void Compute(int32_t progress, int length)
        {
            AscendC::LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
            AscendC::LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
            AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

            AscendC::LocalTensor<int32_t> X1 = X1Buf.Get<int32_t>();
            AscendC::LocalTensor<int32_t> X2 = X2Buf.Get<int32_t>();

            AscendC::LocalTensor<int16_t> X116 = X1.template ReinterpretCast<int16_t>();
            AscendC::LocalTensor<int16_t> X216 = X2.template ReinterpretCast<int16_t>();
            AscendC::SetMaskCount();

            if constexpr(std::is_same_v<T, int8_t>)
            {
                AscendC::LocalTensor<float> X2float = X2.template ReinterpretCast<float>();
                AscendC::LocalTensor<half> tmp = tmpBuf.Get<half>();
                AscendC::Cast(tmp, x1Local, AscendC::RoundMode::CAST_NONE, length);
                AscendC::Cast(X1, tmp, AscendC::RoundMode::CAST_TRUNC, length);
                AscendC::Cast(tmp, x2Local, AscendC::RoundMode::CAST_NONE, length);
                AscendC::Cast(X2, tmp, AscendC::RoundMode::CAST_TRUNC, length);
                AscendC::Mins(X2, X2, 8, length);
                AscendC::Adds(X2, X2, 127, length);
                AscendC::ShiftLeft(X2, X2, 23, length);
                AscendC::Cast(X2, X2float, AscendC::RoundMode::CAST_TRUNC, length);
                AscendC::Mul(X1, X1, X2, length);
                //if (GetBlockIdx()==0) printf("%d %d\n", X1(1), X2(1));
                //AscendC::Mins(X1, X1, 256, length);
                AscendC::ShiftLeft(X1, X1, 24, length);
                AscendC::ShiftRight(X1, X1, 24, length);
                //if (GetBlockIdx()==0) printf("%d\n", X1(1));
                AscendC::Cast(X2float, X1, AscendC::RoundMode::CAST_TRUNC, length);
                AscendC::Cast(tmp, X2float, AscendC::RoundMode::CAST_TRUNC, length);
                AscendC::Cast(yLocal, tmp, AscendC::RoundMode::CAST_TRUNC, length);
            }
            else if constexpr(std::is_same_v<T, int16_t>)
            {
                AscendC::LocalTensor<half> X2half = x2Local.template ReinterpretCast<half>();
                AscendC::SetVectorMask<int16_t, AscendC::MaskMode::COUNTER>(length);
                AscendC::SetVectorMask<half, AscendC::MaskMode::COUNTER>(length);
                AscendC::Mins<int16_t, false>(x2Local, x2Local, (int16_t)16, 128, 63, {1,1,8,8});
                AscendC::Duplicate<int16_t, false>(X116, (int16_t)15, 128, 63, 1, 8);
                AscendC::And<int16_t, false>(x2Local, x2Local, X116, 128, 63, {1,1,1,8,8,8});
                AscendC::Adds<int16_t, false>(x2Local, x2Local, (int16_t)15, 128, 63, {1,1,8,8});
                AscendC::ShiftLeft<int16_t, false>(x2Local, x2Local, (int16_t)10, 128, 63, {1,1,8,8});
                AscendC::Muls<half, false>(X2half, X2half, (half)-1, 128, 63, {1,1,8,8});
                AscendC::Cast<int16_t, half, false>(x2Local, X2half, AscendC::RoundMode::CAST_TRUNC, 128, 63, {1,1,8,8});

                AscendC::Muls<int16_t, false>(x1Local, x1Local, (int16_t)-1, 128, 63, {1,1,8,8});
                AscendC::Mul<int16_t, false>(yLocal, x1Local, x2Local, 128, 63, {1,1,1,8,8,8});
            }
            else if constexpr(std::is_same_v<T, int32_t>)
            {
                AscendC::LocalTensor<int16_t> x2Local16 = x2Local.template ReinterpretCast<int16_t>();
                AscendC::LocalTensor<float> X1float = x1Local.template ReinterpretCast<float>();
                AscendC::LocalTensor<float> X2float = x2Local.template ReinterpretCast<float>();
                AscendC::SetVectorMask<int32_t, AscendC::MaskMode::COUNTER>(length);
                AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(length);
                
                AscendC::ShiftLeft<int32_t, false>(x2Local, x2Local, 23, 64, 85, {1,1,8,8});
                AscendC::Cast<float, int32_t, false>(X1float, x1Local, AscendC::RoundMode::CAST_NONE, 64, 85, {1,1,8,8});

                AscendC::Add<int32_t, false>(x1Local, x1Local, x2Local, 64, 85, {1,1,1,8,8,8});
                AscendC::Cast<int32_t, float, false>(yLocal, X1float, AscendC::RoundMode::CAST_TRUNC, 64, 85, {1,1,8,8});
            }
            else if constexpr(std::is_same_v<T, int64_t>)
            {
                AscendC::LocalTensor<float> X2float = X2.template ReinterpretCast<float>();
                AscendC::Cast(X2, x2Local, AscendC::RoundMode::CAST_NONE, length);
                AscendC::Mins(X2, X2, 32, length);
                AscendC::Duplicate(X1, 31, length);
                AscendC::And(X216, X216, X116, length * 2);
                AscendC::Adds(X2, X2, 127, length);
                AscendC::ShiftLeft(X2, X2, 23, length);
                AscendC::Muls(X2float, X2float, (float)-1, length);
                AscendC::Cast(X2, X2float, AscendC::RoundMode::CAST_TRUNC, length);

                AscendC::Cast(X1, x1Local, AscendC::RoundMode::CAST_NONE, length);
                AscendC::Muls(X1, X1, -1, length);
                AscendC::Mul(X1, X1, X2, length);
                AscendC::Cast(yLocal, X1, AscendC::RoundMode::CAST_NONE, length);
            }


            outQueueY.EnQue<T>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }
        __aicore__ inline void CopyOut(int32_t progress, int l)
        {
            AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();  
            AscendC::DataCopy(yGm[progress], yLocal, l);
            outQueueY.FreeTensor(yLocal);
        }
    private:
        static constexpr int32_t LEN = ((std::is_same_v<T, int8_t>)?10752:
                                (std::is_same_v<T, int16_t>)?8064:
                                (std::is_same_v<T, int32_t>)?5376:3008);
        TPipe* pipe;
        GlobalTensor<T> x1Gm, x2Gm, yGm;
        int32_t L, R;
        //create queue for input, in this case depth is equal to buffer num
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
        //create queue for output, in this case depth is equal to buffer num
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
        TBuf<AscendC::TPosition::VECCALC> X1Buf, X2Buf, X3Buf, tmpBuf;
};

template<typename T> class LeftShiftBroad {
    public:
        __aicore__ inline LeftShiftBroad() {}
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, int32_t size, int32_t length, int32_t* shape, int32_t* n1, int32_t* n2, TPipe* pi) {
            pipe = pi;
            unsigned L = GetBlockIdx() * length;
            unsigned R = (GetBlockIdx() + 1) * length;
            if (L > size) {
                L = size;
            }
            if (R > size) {
                R = size;
            }

            x1Gm.SetGlobalBuffer((__gm__ T*)x1);
            x2Gm.SetGlobalBuffer((__gm__ T*)x2);
            yGm.SetGlobalBuffer((__gm__ T*)y);
            // x1Gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            // x2Gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            // yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            this->L = L;
            this->R = R;
            pipe->InitBuffer(inQueueX1, BUFFER_NUM, LEN * sizeof(T));
            pipe->InitBuffer(inQueueX2, BUFFER_NUM, LEN * sizeof(T));
            pipe->InitBuffer(outQueueY, BUFFER_NUM, LEN * sizeof(T));

            pipe->InitBuffer(X1Buf, LEN * sizeof(int32_t));
            pipe->InitBuffer(X2Buf, LEN * sizeof(int32_t));
            if constexpr(std::is_same_v<T, int8_t>)
            {
                pipe->InitBuffer(tmpBuf, LEN * sizeof(int32_t));
            }

            for (int i = 0; i < MAX_DIM_NUMBER; i++)
            {
                this->shape[i] = shape[i];
                this->n1[i] = n1[i];
                this->n2[i] = n2[i];
            }
            int id = L;
            idx1 = idx2 = 0;
            for (int i = MAX_DIM_NUMBER - 1; i >= 0; i--)
            {
                idx[i] = id % shape[i];
                id /= shape[i];
                idx1 += idx[i] * n1[i];
                idx2 += idx[i] * n2[i];
            }
        }
        __aicore__ inline void Process() {
            int i;
            int len = shape[MAX_DIM_NUMBER - 1];
            for (i = L; i + len <= R; i+=len) {
                CopyIn(len);
                Compute(len);
                CopyOut(i, len);
                idx[1] ++;
                idx1 += n1[1];
                idx2 += n2[1];
                if (idx[1] == shape[1])
                {
                    idx[1] = 0;
                    idx1 -= n1[1] * shape[1];
                    idx2 -= n2[1] * shape[1];
                    idx[0] ++;
                    idx1 += n1[0];
                    idx2 += n2[0];
                }
                
            }
            if (i < R)
            {
                int t = R-i;
                if (t<32/sizeof(T)) t=32/sizeof(T);
                if (t<8) t=8;
                CopyIn(t);
                Compute(t);
                CopyOut(i, t);
            }
        }
    
    private:
        __aicore__ inline void CopyIn(uint32_t l)
        {
            AscendC::LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
            AscendC::LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
            AscendC::DataCopyExtParams copyParams{1, 0, 0, 0, 0};
            copyParams.blockLen = l * sizeof(T);
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(x1Local, x1Gm[idx1], copyParams, padParams);
            AscendC::DataCopyPad(x2Local, x2Gm[idx2], copyParams, padParams);
            inQueueX1.EnQue(x1Local);
            inQueueX2.EnQue(x2Local);
        }
        __aicore__ inline void Compute(int length)
        {
            AscendC::LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
            AscendC::LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
            AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

            AscendC::LocalTensor<int32_t> X1 = X1Buf.Get<int32_t>();
            AscendC::LocalTensor<int32_t> X2 = X2Buf.Get<int32_t>();

            AscendC::LocalTensor<int16_t> X116 = X1.template ReinterpretCast<int16_t>();
            AscendC::LocalTensor<int16_t> X216 = X2.template ReinterpretCast<int16_t>();
            AscendC::SetMaskCount();

            if constexpr(std::is_same_v<T, int8_t>)
            {
                AscendC::LocalTensor<float> X2float = X2.template ReinterpretCast<float>();
                AscendC::LocalTensor<half> tmp = tmpBuf.Get<half>();
                AscendC::Cast(tmp, x1Local, AscendC::RoundMode::CAST_NONE, length);
                AscendC::Cast(X1, tmp, AscendC::RoundMode::CAST_TRUNC, length);
                AscendC::Cast(tmp, x2Local, AscendC::RoundMode::CAST_NONE, length);
                AscendC::Cast(X2, tmp, AscendC::RoundMode::CAST_TRUNC, length);
                AscendC::Mins(X2, X2, 8, length);
                AscendC::Adds(X2, X2, 127, length);
                AscendC::ShiftLeft(X2, X2, 23, length);
                AscendC::Cast(X2, X2float, AscendC::RoundMode::CAST_TRUNC, length);
                AscendC::Mul(X1, X1, X2, length);
                //if (GetBlockIdx()==0) printf("%d %d\n", X1(1), X2(1));
                //AscendC::Mins(X1, X1, 256, length);
                AscendC::ShiftLeft(X1, X1, 24, length);
                AscendC::ShiftRight(X1, X1, 24, length);
                //if (GetBlockIdx()==0) printf("%d\n", X1(1));
                AscendC::Cast(X2float, X1, AscendC::RoundMode::CAST_TRUNC, length);
                AscendC::Cast(tmp, X2float, AscendC::RoundMode::CAST_TRUNC, length);
                AscendC::Cast(yLocal, tmp, AscendC::RoundMode::CAST_TRUNC, length);
            }
            else if constexpr(std::is_same_v<T, int16_t>)
            {
                AscendC::LocalTensor<half> X2half = x2Local.template ReinterpretCast<half>();
                AscendC::SetVectorMask<int16_t, AscendC::MaskMode::COUNTER>(length);
                AscendC::SetVectorMask<half, AscendC::MaskMode::COUNTER>(length);
                AscendC::Mins<int16_t, false>(x2Local, x2Local, (int16_t)16, 128, 63, {1,1,8,8});
                AscendC::Duplicate<int16_t, false>(X116, (int16_t)15, 128, 63, 1, 8);
                AscendC::And<int16_t, false>(x2Local, x2Local, X116, 128, 63, {1,1,1,8,8,8});
                AscendC::Adds<int16_t, false>(x2Local, x2Local, (int16_t)15, 128, 63, {1,1,8,8});
                AscendC::ShiftLeft<int16_t, false>(x2Local, x2Local, (int16_t)10, 128, 63, {1,1,8,8});
                AscendC::Muls<half, false>(X2half, X2half, (half)-1, 128, 63, {1,1,8,8});
                AscendC::Cast<int16_t, half, false>(x2Local, X2half, AscendC::RoundMode::CAST_TRUNC, 128, 63, {1,1,8,8});

                AscendC::Muls<int16_t, false>(x1Local, x1Local, (int16_t)-1, 128, 63, {1,1,8,8});
                AscendC::Mul<int16_t, false>(yLocal, x1Local, x2Local, 128, 63, {1,1,1,8,8,8});
            }
            else if constexpr(std::is_same_v<T, int32_t>)
            {
                AscendC::LocalTensor<int16_t> x2Local16 = x2Local.template ReinterpretCast<int16_t>();
                AscendC::LocalTensor<float> X2float = x2Local.template ReinterpretCast<float>();
                AscendC::SetVectorMask<int32_t, AscendC::MaskMode::COUNTER>(length);
                AscendC::SetVectorMask<int16_t, AscendC::MaskMode::COUNTER>(length * 2);
                AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(length);
                AscendC::Mins<int32_t, false>(x2Local, x2Local, 32, 64, 85, {1,1,8,8});
                AscendC::Duplicate<int32_t, false>(X1, 31, 64, 85, 1, 8);
                AscendC::And<int16_t, false>(x2Local16, x2Local16, X116, 128, 85, {1,1,1,8,8,8});
                AscendC::Adds<int32_t, false>(x2Local, x2Local, 127, 64, 85, {1,1,8,8});
                AscendC::ShiftLeft<int32_t, false>(x2Local, x2Local, 23, 64, 85, {1,1,8,8});
                AscendC::Muls<float, false>(X2float, X2float, (float)-1, 64, 85, {1,1,8,8});
                AscendC::Cast<int32_t, float, false>(x2Local, X2float, AscendC::RoundMode::CAST_TRUNC, 64, 85, {1,1,8,8});

                AscendC::Muls<int32_t, false>(x1Local, x1Local, -1, 64, 85, {1,1,8,8});
                AscendC::Mul<int32_t, false>(yLocal, x1Local, x2Local, 64, 85, {1,1,1,8,8,8});
            }
            else if constexpr(std::is_same_v<T, int64_t>)
            {
                AscendC::LocalTensor<float> X2float = X2.template ReinterpretCast<float>();
                AscendC::Cast(X2, x2Local, AscendC::RoundMode::CAST_NONE, length);
                AscendC::Mins(X2, X2, 32, length);
                AscendC::Duplicate(X1, 31, length);
                AscendC::And(X216, X216, X116, length * 2);
                AscendC::Adds(X2, X2, 127, length);
                AscendC::ShiftLeft(X2, X2, 23, length);
                AscendC::Muls(X2float, X2float, (float)-1, length);
                AscendC::Cast(X2, X2float, AscendC::RoundMode::CAST_TRUNC, length);

                AscendC::Cast(X1, x1Local, AscendC::RoundMode::CAST_NONE, length);
                AscendC::Muls(X1, X1, -1, length);
                AscendC::Mul(X1, X1, X2, length);
                AscendC::Cast(yLocal, X1, AscendC::RoundMode::CAST_NONE, length);
            }


            outQueueY.EnQue<T>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }
        __aicore__ inline void CopyOut(int32_t progress, uint32_t l)
        {
            AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
            AscendC::DataCopyExtParams copyParams{1, 0, 0, 0, 0};
            copyParams.blockLen = l * sizeof(T);
            AscendC::DataCopyPad(yGm[progress], yLocal, copyParams);
            outQueueY.FreeTensor(yLocal);
        }
    private:
        static constexpr int32_t LEN = ((std::is_same_v<T, int8_t>)?10752:
                                (std::is_same_v<T, int16_t>)?8064:
                                (std::is_same_v<T, int32_t>)?5376:3008);
        TPipe* pipe;
        GlobalTensor<T> x1Gm, x2Gm, yGm;
        int32_t L, R, idx1, idx2;
        //create queue for input, in this case depth is equal to buffer num
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
        //create queue for output, in this case depth is equal to buffer num
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
        TBuf<AscendC::TPosition::VECCALC> X1Buf, X2Buf, X3Buf, tmpBuf;

        int32_t idx[MAX_DIM_NUMBER], shape[MAX_DIM_NUMBER], n1[MAX_DIM_NUMBER], n2[MAX_DIM_NUMBER];
};

extern "C" __global__ __aicore__ void bitwise_left_shift(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1))
    {
        LeftShiftScalar<DTYPE_X1> op;
        op.Init(x1, x2, y, tiling_data.size, tiling_data.length);
        op.Process();
    }
    else if (TILING_KEY_IS(2))
    {
        LeftShiftVector<DTYPE_X1> op;
        op.Init(x1, x2, y, tiling_data.size, tiling_data.length, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(3))
    {
        LeftShiftBroad<DTYPE_X1> op;
        op.Init(x1, x2, y, tiling_data.size, tiling_data.length, tiling_data.shape, tiling_data.n1, tiling_data.n2, &pipe);
        op.Process();
    }
    // TODO: user kernel impl
}

/*
Case1: Wrong answer
Case2: Pass, Result: 8.1036
Case3: Run failed!
Case4: Pass, Result: 13.6588
Case5: Pass, Result: 333.731
*/