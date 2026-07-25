#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;
using namespace AscendC;

constexpr uint64_t pre[65] = {18446744073709551615ull,1ull,3ull,5ull,11ull,17ull,39ull,65ull,139ull,261ull,531ull,1025ull,2095ull,4097ull,8259ull,16405ull,32907ull,65537ull,131367ull,262145ull,524827ull,1048645ull,2098179ull,4194305ull,8390831ull,16777233ull,33558531ull,67109125ull,134225995ull,268435457ull,536887863ull,1073741825ull,2147516555ull,4294968325ull,8590000131ull,17179869265ull,34359871791ull,68719476737ull,137439215619ull,274877911045ull,549756338843ull,1099511627777ull,2199024312423ull,4398046511105ull,8796095120395ull,17592186061077ull,35184376283139ull,70368744177665ull,140737496778927ull,281474976710721ull,562949970199059ull,1125899906908165ull,2251799847243787ull,4503599627370497ull,9007199321981223ull,18014398509483025ull,36028797153190091ull,72057594038190085ull,144115188344291331ull,288230376151711745ull,576460752840837695ull,1152921504606846977ull,2305843010287435779ull,4611686018428436805ull,9223372039002292363ull};
template<typename T> class LCMKernalFast {
    public:
        __aicore__ inline LCMKernalFast() {}
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, int32_t size, int32_t length) {
            ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
            const unsigned num_cores = GetBlockNum();
            unsigned L = GetBlockIdx() * length;
            unsigned R = (GetBlockIdx() + 1) * length;
            if (L > size) {
                L = size;
            }
            if (R > size) {
                R = size;
            }
            x1Gm.SetGlobalBuffer((__gm__ T*)x1 + L, length);
            x2Gm.SetGlobalBuffer((__gm__ T*)x2 + L, length);
            yGm.SetGlobalBuffer((__gm__ T*)y + L, length);
            // x1Gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            // x2Gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            // yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            this->L = 0;
            this->R = R - L;
        }
        __aicore__ inline void Process() {
            for (int i = L; i < R; ++i) {
                T a = x1Gm.GetValue(i);
                T b = x2Gm.GetValue(i);
                a = (a > 0 ? a : -a);
                b = (b > 0 ? b : -b);
                T c = a;
                T d = b;
                if (a == 0 || b == 0) yGm.SetValue(i, 0);
                else
                {
                    T shift = ScalarGetSFFValue<1>(a | b);
                    a >>= ScalarGetSFFValue<1>(a);
                    do {
                        b >>= ScalarGetSFFValue<1>(b);
                        if(a <= 64 && b <= 64){
                            a = 64 - ScalarCountLeadingZero(pre[a] & pre[b]);
                            break;
                        }
                        if (a > b) {
                            a ^= b ^= a ^= b;
                        }
                        b -= a;
                    } while (b);
                    // if constexpr(std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>) yGm.SetValue(i, c / (a << shift) * d);
                    // else
                    // {
                    //     c = c / (a << shift) * d;
                    //     if (c < 0) yGm.SetValue(i, -c);
                    //     else yGm.SetValue(i, c);
                    // }
                    yGm.SetValue(i, c / (a << shift) * d);
                }
                
                
                
            }
            //AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(yGm);
            //DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE>(yGm);
        }
    
    private:
        GlobalTensor<T> x1Gm, x2Gm, yGm;
        int32_t L, R;
};

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t LEN = 7008;


template<typename T> class LCMVector {
    
    public:
        __aicore__ inline LCMVector() {}
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, int32_t size, int32_t length, TPipe* pi) {
            pipe = pi;
            ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
            const unsigned num_cores = GetBlockNum();
            unsigned L = GetBlockIdx() * length;
            unsigned R = (GetBlockIdx() + 1) * length;
            if (L > size) {
                L = size;
            }
            if (R > size) {
                R = size;
            }
            x1Gm.SetGlobalBuffer((__gm__ T*)x1 + L, length);
            x2Gm.SetGlobalBuffer((__gm__ T*)x2 + L, length);
            yGm.SetGlobalBuffer((__gm__ T*)y + L, length);
            // x1Gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            // x2Gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            // yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            this->L = 0;
            this->R = R - L;
            pipe->InitBuffer(inQueueX1, BUFFER_NUM, LEN * sizeof(T));
            pipe->InitBuffer(inQueueX2, BUFFER_NUM, LEN * sizeof(T));
            pipe->InitBuffer(outQueueY, BUFFER_NUM, LEN * sizeof(T));

            pipe->InitBuffer(X1Buf, LEN * sizeof(float));
            pipe->InitBuffer(X2Buf, LEN * sizeof(float));
            pipe->InitBuffer(X3Buf, LEN * sizeof(float));
            pipe->InitBuffer(tmpBuf, LEN * sizeof(T));
        }
        __aicore__ inline void Process() {
            int i;
            AscendC::SetMaskCount();
            // AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>(LEN);
            // AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(LEN);
            for (i = L; i + LEN <= R; i+=LEN) {
                CopyIn(i, LEN);
                Compute(i, LEN);
                CopyOut(i, LEN);
            }
            if (i < R)
            {
                // AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>(R-i);
                // AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(R-i);
                int t = R-i;
                if (t<32/sizeof(T)) t=32/sizeof(T);
                CopyIn(i, t);
                Compute(i, t);
                CopyOut(i, t);
            }
        }
    
    private:
        __aicore__ inline void CopyIn(int32_t progress, int l)
        {
            //鑰冪敓琛ュ厖绠楀瓙浠ｇ爜
            AscendC::LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
            AscendC::LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
            AscendC::DataCopy(x1Local, x1Gm[progress], l);
            AscendC::DataCopy(x2Local, x2Gm[progress], l);
            inQueueX1.EnQue(x1Local);
            inQueueX2.EnQue(x2Local);
        }
        __aicore__ inline void Compute(int32_t progress, int l)
        {
            //鑰冪敓琛ュ厖绠楀瓙璁＄畻浠ｇ爜
            AscendC::LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
            AscendC::LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
            AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

            AscendC::LocalTensor<float> X1 = X1Buf.Get<float>();
            AscendC::LocalTensor<float> X2 = X2Buf.Get<float>();
            AscendC::LocalTensor<float> X3 = X3Buf.Get<float>();
            AscendC::LocalTensor<T> tmp = tmpBuf.Get<T>();

            AscendC::BinaryRepeatParams binaryParams;
            AscendC::UnaryRepeatParams unaryParams;

            AscendC::Cast(X1, x1Local, AscendC::RoundMode::CAST_NONE, l);
            AscendC::Cast(X2, x2Local, AscendC::RoundMode::CAST_NONE, l);
            AscendC::Abs(X1, X1, l);
            AscendC::Abs(X2, X2, l);
            AscendC::Cast(x1Local, X1, AscendC::RoundMode::CAST_TRUNC, l);
            AscendC::Cast(x2Local, X2, AscendC::RoundMode::CAST_TRUNC, l);
            //oeis A001333

                AscendC::Div(X3, X1, X2, l);
                AscendC::Cast(tmp, X3, AscendC::RoundMode::CAST_ROUND, l);
                AscendC::Cast(X3, tmp, AscendC::RoundMode::CAST_NONE, l);
                AscendC::Mul(X3, X2, X3, l);
                AscendC::Sub(X3, X1, X3, l);

                AscendC::Div(X1, X2, X3, l);
                AscendC::Cast(tmp, X1, AscendC::RoundMode::CAST_ROUND, l);
                AscendC::Cast(X1, tmp, AscendC::RoundMode::CAST_NONE, l);
                AscendC::Mul(X1, X3, X1, l);
                AscendC::Sub(X1, X2, X1, l);

                AscendC::Div(X2, X3, X1, l);
                AscendC::Cast(tmp, X2, AscendC::RoundMode::CAST_ROUND, l);
                AscendC::Cast(X2, tmp, AscendC::RoundMode::CAST_NONE, l);
                AscendC::Mul(X2, X1, X2, l);
                AscendC::Sub(X2, X3, X2, l);
                //----------------------------------------------------------

                AscendC::Div(X3, X1, X2, l);
                AscendC::Cast(tmp, X3, AscendC::RoundMode::CAST_ROUND, l);
                AscendC::Cast(X3, tmp, AscendC::RoundMode::CAST_NONE, l);
                AscendC::Mul(X3, X2, X3, l);
                AscendC::Sub(X3, X1, X3, l);

                AscendC::Div(X1, X2, X3, l);
                AscendC::Cast(tmp, X1, AscendC::RoundMode::CAST_ROUND, l);
                AscendC::Cast(X1, tmp, AscendC::RoundMode::CAST_NONE, l);
                AscendC::Mul(X1, X3, X1, l);
                AscendC::Sub(X1, X2, X1, l);

                AscendC::Div(X2, X3, X1, l);
                AscendC::Cast(tmp, X2, AscendC::RoundMode::CAST_ROUND, l);
                AscendC::Cast(X2, tmp, AscendC::RoundMode::CAST_NONE, l);
                AscendC::Mul(X2, X1, X2, l);
                AscendC::Sub(X2, X3, X2, l);
                //----------------------------------------------------------

                AscendC::Div(X3, X1, X2, l);
                AscendC::Cast(tmp, X3, AscendC::RoundMode::CAST_ROUND, l);
                AscendC::Cast(X3, tmp, AscendC::RoundMode::CAST_NONE, l);
                AscendC::Mul(X3, X2, X3, l);
                AscendC::Sub(X3, X1, X3, l);

                AscendC::Div(X1, X2, X3, l);
                AscendC::Cast(tmp, X1, AscendC::RoundMode::CAST_ROUND, l);
                AscendC::Cast(X1, tmp, AscendC::RoundMode::CAST_NONE, l);
                AscendC::Mul(X1, X3, X1, l);
                AscendC::Sub(X1, X2, X1, l);

                AscendC::Div(X2, X3, X1, l);
                AscendC::Cast(tmp, X2, AscendC::RoundMode::CAST_ROUND, l);
                AscendC::Cast(X2, tmp, AscendC::RoundMode::CAST_NONE, l);
                AscendC::Mul(X2, X1, X2, l);
                AscendC::Sub(X2, X3, X2, l);
                //----------------------------------------------------------
                
                AscendC::Div(X3, X1, X2, l);
                AscendC::Cast(tmp, X3, AscendC::RoundMode::CAST_ROUND, l);
                AscendC::Cast(X3, tmp, AscendC::RoundMode::CAST_NONE, l);
                AscendC::Mul(X3, X2, X3, l);
                AscendC::Sub(X3, X1, X3, l);

                AscendC::Div(X1, X2, X3, l);
                AscendC::Cast(tmp, X1, AscendC::RoundMode::CAST_ROUND, l);
                AscendC::Cast(X1, tmp, AscendC::RoundMode::CAST_NONE, l);
                AscendC::Mul(X1, X3, X1, l);
                AscendC::Sub(X1, X2, X1, l);

                AscendC::Div(X2, X3, X1, l);
                AscendC::Cast(tmp, X2, AscendC::RoundMode::CAST_ROUND, l);
                AscendC::Cast(X2, tmp, AscendC::RoundMode::CAST_NONE, l);
                AscendC::Mul(X2, X1, X2, l);
                AscendC::Sub(X2, X3, X2, l);
                //----------------------------------------------------------
            
            AscendC::Add(X1, X2, X1, l);
            AscendC::Abs(X1, X1, l);
            
            AscendC::Cast(X2, x2Local, AscendC::RoundMode::CAST_NONE, l);
            AscendC::Div(X2, X2, X1, l);
            AscendC::Cast(x2Local, X2, AscendC::RoundMode::CAST_TRUNC, l);
            AscendC::Mul(yLocal, x1Local, x2Local, l);
            
            outQueueY.EnQue<T>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }
        __aicore__ inline void CopyOut(int32_t progress, int l)
        {
            //鑰冪敓琛ュ厖绠楀瓙浠ｇ爜
            AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();  
            AscendC::DataCopy(yGm[progress], yLocal, l);
            outQueueY.FreeTensor(yLocal);
        }

    private:
        TPipe* pipe;
        GlobalTensor<T> x1Gm, x2Gm, yGm;
        //create queue for input, in this case depth is equal to buffer num
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
        //create queue for output, in this case depth is equal to buffer num
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
        TBuf<AscendC::TPosition::VECCALC> X1Buf, X2Buf, X3Buf, tmpBuf;
        int32_t L, R, processDataNum;
};


constexpr int LEN2 = 1024;
template<typename T> class LCMKernalScalar {
    public:
        __aicore__ inline LCMKernalScalar() {}
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, int32_t size, int32_t length, int32_t n1[3], int32_t n2[3], TPipe *pi) {
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
            for (int i = 0; i < 3; i++)
            {
                this->n1[i] = n1[i];
                this->n2[i] = n2[i];
            }
            pipe->InitBuffer(inQueueX1, BUFFER_NUM, LEN * sizeof(T));
            pipe->InitBuffer(inQueueX2, BUFFER_NUM, LEN * sizeof(T));
            pipe->InitBuffer(outQueueY, BUFFER_NUM, LEN2 * sizeof(T));

            pipe->InitBuffer(X1Buf, LEN2 * sizeof(int32_t));
            pipe->InitBuffer(X2Buf, LEN2 * sizeof(int32_t));
            pipe->InitBuffer(X3Buf, LEN2 * sizeof(int32_t));
            pipe->InitBuffer(X4Buf, LEN2 * sizeof(int32_t));
            pipe->InitBuffer(X5Buf, LEN2 * sizeof(int32_t));
            pipe->InitBuffer(X6Buf, LEN2 * sizeof(int32_t));
            pipe->InitBuffer(X7Buf, LEN2 * sizeof(int32_t));
            pipe->InitBuffer(X8Buf, LEN2 * sizeof(int32_t));
            pipe->InitBuffer(indexBuf, LEN2 * sizeof(int32_t));
            pipe->InitBuffer(X1hBuf, LEN2 * sizeof(int32_t));
            pipe->InitBuffer(X2hBuf, LEN2 * sizeof(int32_t));
            pipe->InitBuffer(X3hBuf, LEN2 * sizeof(int32_t));
            pipe->InitBuffer(X4hBuf, LEN2 * sizeof(int32_t));
            pipe->InitBuffer(X5hBuf, LEN2 * sizeof(int32_t));
            pipe->InitBuffer(X6hBuf, LEN2 * sizeof(int32_t));
            pipe->InitBuffer(X7hBuf, LEN2 * sizeof(int32_t));
            pipe->InitBuffer(X8hBuf, LEN2 * sizeof(int32_t));
            AscendC::LocalTensor<int32_t> index = indexBuf.Get<int32_t>();
            for (int i=0;i<LEN2;i+=4)
            {
                index.SetValue(i,i);
                index.SetValue(i+1,i+1);
                index.SetValue(i+2,i+2);
                index.SetValue(i+3,i+3);
            }
        }
        __aicore__ inline void Process() {
            constexpr T neg = ((std::is_same_v<T, int8_t>)?-128:
                            (std::is_same_v<T, int16_t>)?-32768:
                            (std::is_same_v<T, int32_t>)?-2147483648:-9223372036854775808LL);
            int A = L / (n1[1] * n2[2]), C = L % n2[2], B = L / n2[2] - A * n1[1], d1 = n1[1], d2 = n2[2];
            A *= d2;
            int id = 0, lastL = L;


            for (int i = L; i < R; i++) {
                T x1,x2;
                x1 = x1Gm.GetValue(i);
                x2 = x2Gm.GetValue(A + C);
                C++;if (C == d2) {B++;C=0;if (B == d1){A+=d2;B=0;}}
                x1 = (x1 > 0 ? x1 : -x1);
                x2 = (x2 > 0 ? x2 : -x2);

                T a = x1, b = x2;
                x1 >>= ScalarGetSFFValue<1>(a | b);
                a >>= ScalarGetSFFValue<1>(a);
                if (a == 0 || b == 0) yGm.SetValue(i, 0);
                else
                {
                    T shift = ScalarGetSFFValue<1>(a | b);
                    a >>= ScalarGetSFFValue<1>(a);
                    do {
                        b >>= ScalarGetSFFValue<1>(b);
                        if (a > b) {
                            a ^= b ^= a ^= b;
                        }
                        b -= a;
                    } while (b);
                    // if constexpr(std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>) yGm.SetValue(i, c / (a << shift) * d);
                    // else
                    // {
                    //     c = c / (a << shift) * d;
                    //     if (c < 0) yGm.SetValue(i, -c);
                    //     else yGm.SetValue(i, c);
                    // }
                    yGm.SetValue(i, c / (a << shift) * d);
                }

                }

            }
        }
    
    private:
        #define LTsr const LocalTensor<int32_t>
        __aicore__ inline void TWO(LTsr &X1h, LTsr &X1)//X1 = 2
        {
            AscendC::Duplicate<int32_t, false>(X1h, 0, AscendC::MASK_PLACEHOLDER, 0, 1, 8);\
            AscendC::Duplicate<int32_t, false>(X1, 2, AscendC::MASK_PLACEHOLDER, 0, 1, 8);
        }
        __aicore__ inline void mySub(LTsr &X1h, LTsr &X1, LTsr &X2h, LTsr &X2, LTsr &X3h, LTsr &X3, LTsr &tmp, LTsr &tmp2)//X3 = X1 - X2
        {
            AscendC::Sub<int32_t, false>(X3h, X1h, X2h, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
            AscendC::Sub<int32_t, false>(X3, X1, X2, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\

            AscendC::ShiftRight<int32_t, false>(tmp, X1, 31, LEN2);\
            AscendC::ShiftRight<int32_t, false>(tmp2, X2, 31, LEN2);\
            AscendC::Sub<int32_t, false>(tmp, tmp2, tmp, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
            AscendC::ShiftLeft<int32_t, false>(tmp, tmp, 1, LEN2);\

            AscendC::ShiftRight<int32_t, false>(tmp2, X3, 31, LEN2);\
            AscendC::Add<int32_t, false>(tmp, tmp2, tmp, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
            AscendC::ShiftRight<int32_t, false>(tmp, tmp, 31, LEN2);\
            AscendC::Add<int32_t, false>(X3h, X3h, tmp, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
        }
        __aicore__ inline void myAdd(LTsr &X1h, LTsr &X1, LTsr &X2h, LTsr &X2, LTsr &X3h, LTsr &X3, LTsr &tmp, LTsr &tmp2)//X3 = X1 + X2
        {
            AscendC::Add<int32_t, false>(X3h, X1h, X2h, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
            AscendC::Add<int32_t, false>(X3, X1, X2, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\

            AscendC::ShiftRight<int32_t, false>(tmp, X1, 31, LEN2);\
            AscendC::ShiftRight<int32_t, false>(tmp2, X2, 31, LEN2);\
            AscendC::Add<int32_t, false>(tmp, tmp2, tmp, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
            //0,-1,-2

            AscendC::ShiftRight<int32_t, false>(tmp2, X3, 31, LEN2);\
            AscendC::Sub<int32_t, false>(tmp, tmp, tmp2, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
            AscendC::ShiftRight<int32_t, false>(tmp, tmp, 31, LEN2);\
            AscendC::Sub<int32_t, false>(X3h, X3h, tmp, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
        }
        __aicore__ inline void myMul(LTsr &X1h, LTsr &X1, LTsr &X2h, LTsr &X2, LTsr &X3h, LTsr &X3, LTsr &tmp, LTsr &tmp2)//X3 = X1 * X2. note: X1 and X2 will destroy.
        {
            auto uX1h = X1h.template ReinterpretCast<uint32_t>();
            auto uX1 = X1.template ReinterpretCast<uint32_t>();
            auto uX2h = X2h.template ReinterpretCast<uint32_t>();
            auto uX2 = X2.template ReinterpretCast<uint32_t>();
            //auto uX3h = X3h.template ReinterpretCast<uint32_t>();
            //auto uX3 = X3.template ReinterpretCast<uint32_t>();
            auto utmp = tmp.template ReinterpretCast<uint32_t>();
            auto utmp2 = tmp2.template ReinterpretCast<uint32_t>();
            AscendC::Mul<int32_t, false>(X3h, X1h, X2, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
            AscendC::Mul<int32_t, false>(X3, X1, X2h, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
            AscendC::Add<int32_t, false>(X3h, X3h, X3, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
            AscendC::Mul<int32_t, false>(X3, X1, X2, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
            //if (GetBlockIdx()==0) printf("mul%d %d %d\n", X3(1),X1(1), X2(1));
            //X1h, X2h, tmp, tmp2
            AscendC::ShiftLeft<int32_t, false>(tmp, X1, 16, LEN2);\
            AscendC::ShiftRight<uint32_t, false>(utmp, utmp, 16, LEN2);\
            AscendC::ShiftLeft<int32_t, false>(tmp2, X2, 16, LEN2);\
            AscendC::ShiftRight<uint32_t, false>(utmp2, utmp2, 16, LEN2);\
            //tmp is low
            AscendC::ShiftRight<uint32_t, false>(uX1, uX1, 16, LEN2);\
            AscendC::ShiftRight<uint32_t, false>(uX2, uX2, 16, LEN2);\
            //X1,X2 is high

            AscendC::Mul<int32_t, false>(X1h, X1, X2, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
            AscendC::Add<int32_t, false>(X3h, X3h, X1h, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\

            AscendC::Mul<int32_t, false>(X2h, tmp, tmp2, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
            AscendC::ShiftRight<uint32_t, false>(uX2h, uX2h, 16, LEN2);\
            AscendC::Mul<int32_t, false>(tmp, tmp, X2, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
            AscendC::Mul<int32_t, false>(tmp2, tmp2, X1, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
            AscendC::Add<int32_t, false>(tmp, tmp, X2h, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
            AscendC::Add<int32_t, false>(X1h, tmp, tmp2, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
            //X1,X2,X2h. X1h = tmp + tmp2;

            AscendC::ShiftRight<int32_t, false>(X1, tmp, 31, LEN2);\
            AscendC::ShiftRight<int32_t, false>(X2, tmp2, 31, LEN2);\
            AscendC::Add<int32_t, false>(X1, X1, X2, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
            //0,-1,-2

            AscendC::ShiftRight<int32_t, false>(X2, X1h, 31, LEN2);\
            AscendC::Sub<int32_t, false>(X1, X1, X2, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
            AscendC::ShiftRight<int32_t, false>(X1, X1, 31, LEN2);\
            AscendC::ShiftLeft<int32_t, false>(X1, X1, 16, LEN2);\
            AscendC::Sub<int32_t, false>(X3h, X3h, X1, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
            AscendC::ShiftRight<uint32_t, false>(uX1h, uX1h, 16, LEN2);\
            AscendC::Add<int32_t, false>(X3h, X3h, X1h, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });\
        }
        TPipe *pipe;
        GlobalTensor<T> x1Gm, x2Gm, yGm;
        int32_t L, R, n1[3], n2[3];
        int tag1, tag2;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
        TBuf<AscendC::TPosition::VECCALC> X1Buf, X2Buf, X3Buf, X4Buf, X5Buf, X6Buf, X7Buf, X8Buf, X1hBuf, X2hBuf, X3hBuf, X4hBuf, X5hBuf, X6hBuf, X7hBuf, X8hBuf, indexBuf;
};

template<typename T> class LCMKernalScalarSlow {
    public:
        __aicore__ inline LCMKernalScalarSlow() {}
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, int32_t size, int32_t length, int tag1, int tag2, int32_t n1[3], int32_t n2[3]) {

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
            for (int i = 0; i < 3; i++)
            {
                this->n1[i] = n1[i];
                this->n2[i] = n2[i];
            }
            this->tag1 = tag1;
            this->tag2 = tag2;
        }
        __aicore__ inline void Process() {
            int i;
            constexpr T neg = ((std::is_same_v<T, int8_t>)?-128:
                            (std::is_same_v<T, int16_t>)?-32768:
                            (std::is_same_v<T, int32_t>)?-2147483648:-9223372036854775808LL);
            for (int i = L; i < R; i++) {
                T a,b;
                switch (tag1){
                    case 0: a = x1Gm.GetValue(i);break;
                    case 1: a = x1Gm.GetValue(i / (n2[2]));break;
                    case 2: a = x1Gm.GetValue(i / (n2[1] * n1[2]) * n1[2] + i % n1[2]);break;//a*d1*d2+b*d2+c -> a*d2+c
                    case 3: a = x1Gm.GetValue(i / (n2[1] * n2[2]));break;
                    case 4: a = x1Gm.GetValue(i % (n1[1] * n1[2]));break;
                    case 5: a = x1Gm.GetValue((i / n2[2]) % n1[1]);break;
                    case 6: a = x1Gm.GetValue(i % n2[2]);break;
                    default: a = x1Gm.GetValue(0);break;
                }
                switch (tag2){
                    case 0: b = x2Gm.GetValue(i);break;
                    case 1: b = x2Gm.GetValue(i / (n1[2]));break;
                    case 2: b = x2Gm.GetValue(i / (n1[1] * n2[2]) * n2[2] + i % n2[2]);break;//a*d1*d2+b*d2+c -> a*d2+c
                    case 3: b = x2Gm.GetValue(i / (n1[1] * n1[2]));break;
                    case 4: b = x2Gm.GetValue(i % (n2[1] * n2[2]));break;
                    case 5: b = x2Gm.GetValue((i / n1[2]) % n2[1]);break;
                    case 6: b = x2Gm.GetValue(i % n1[2]);break;
                    default: b = x2Gm.GetValue(0);break;
                }
                a = (a > 0 ? a : -a);
                b = (b > 0 ? b : -b);
                T c = a;
                T d = b;
                if (a == neg || b == neg) yGm.SetValue(i, neg);
                else
                {
                    T shift = ScalarGetSFFValue<1>(a | b);
                    a >>= ScalarGetSFFValue<1>(a);
                    do {
                        b >>= ScalarGetSFFValue<1>(b);
                        if(a <= 64 && b <= 64){
                            a = 64 - ScalarCountLeadingZero(pre[a] & pre[b]);
                            break;
                        }
                        if (a > b) {
                            a ^= b ^= a ^= b;
                        }
                        b -= a;
                    } while (b);
                    c = c / (a << shift) * d;
                    if (c < 0) yGm.SetValue(i, -c);
                    else yGm.SetValue(i, c);
                }
            }
        }
    
    private:
        GlobalTensor<T> x1Gm, x2Gm, yGm;
        int32_t L, R, n1[3], n2[3];
        int tag1, tag2;
};


extern "C" __global__ __aicore__ void lcm(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    //KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)
    TPipe pipe;
    // TODO: user kernel impl
    if constexpr(std::is_same_v<DTYPE_X1, int16_t> || std::is_same_v<DTYPE_X1, int32_t> )
    {
        LCMVector<DTYPE_X1> op;
        op.Init(x1, x2, y, tiling_data.size, tiling_data.length, &pipe);
        op.Process();
    }
    else if constexpr(std::is_same_v<DTYPE_X1, int8_t>)
    {
        LCMKernalFast<DTYPE_X1> op;
        op.Init(x1, x2, y, tiling_data.size, tiling_data.length);
        op.Process();
    }
    else 
    {
        if (tiling_data.tag1==0 && tiling_data.tag2==2)
        {
            LCMKernalScalar<DTYPE_X1> op;
            op.Init(x1, x2, y, tiling_data.size, tiling_data.length, tiling_data.n1, tiling_data.n2, &pipe);
            op.Process();
        }
        else
        {
            LCMKernalScalarSlow<DTYPE_X1> op;
            op.Init(x1, x2, y, tiling_data.size, tiling_data.length, tiling_data.tag1, tiling_data.tag2, tiling_data.n1, tiling_data.n2);
            op.Process();
        }
    }
    // if (TILING_KEY_IS(3))
    // {
    //     LCMKernalScalar<DTYPE_X1> op;
    //     op.Init(x1, x2, y, tiling_data.size, tiling_data.length, tiling_data.n1, tiling_data.n2);
    //     op.Process();
    // }
    // else if (TILING_KEY_IS(4))
    // {
    //     LCMKernalScalarSlow<DTYPE_X1> op;
    //     op.Init(x1, x2, y, tiling_data.size, tiling_data.length, tiling_data.tag1, tiling_data.tag2, tiling_data.n1, tiling_data.n2);
    //     op.Process();
    // }
    // else if (TILING_KEY_IS(1))
    // {
    //     LCMVector<DTYPE_X1> op;
    //     op.Init(x1, x2, y, tiling_data.size, tiling_data.length, &pipe);
    //     op.Process();
    // }
    // else if (TILING_KEY_IS(2))
    // {
    //     LCMKernalFast<DTYPE_X1> op;
    //     op.Init(x1, x2, y, tiling_data.size, tiling_data.length);
    //     op.Process();
    // }
}
/*
Case1: Pass, Result: 7.1868
Case2: Pass, Result: 52.648
Case3: Pass, Result: 12738.625
Case4: Pass, Result: 360.119
Case5: Pass, Result: 31059.3328
prof_sum: 44217.9116


Case1: Pass, Result: 7.1116
Case2: Pass, Result: 9.5332
Case3: Pass, Result: 7318.965
Case4: Pass, Result: 17.6072
Case5: Pass, Result: 757.826
prof_sum: 8111.043
*/