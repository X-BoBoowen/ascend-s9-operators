#include "kernel_operator.h"

using namespace AscendC;
#define MAX_DIM_NUMBER 4

template<typename T> class CopysignScalar {
    public:
        __aicore__ inline CopysignScalar() {}
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
                if constexpr(std::is_same_v<T, float32_t>)
                {
                    int32_t *p1=reinterpret_cast<int32_t*>(&a), *p2=reinterpret_cast<int32_t*>(&b);
	                int32_t d=(*p2)&0x80000000;
                    (*p1)&=0x7fffffff;
                    (*p1)+=d;
                    yGm.SetValue(i, a);
                }
                else
                {
                    int16_t *p1=reinterpret_cast<int16_t*>(&a), *p2=reinterpret_cast<int16_t*>(&b);
	                int16_t d=(*p2)&((int16_t)0x8000);
                    (*p1)&=((int16_t)0x7fff);
                    (*p1)+=d;
                    yGm.SetValue(i, a);
                }
            }
        }
    
    private:
        GlobalTensor<T> x1Gm, x2Gm, yGm;
        int32_t L, R;
};

constexpr int32_t BUFFER_NUM = 2;
template<typename T> class CopysignVector {
    public:
        __aicore__ inline CopysignVector() {}
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, int32_t size, int32_t length, TPipe* pi) {
            pipe = pi;

            x1Gm.SetGlobalBuffer((__gm__ T*)x1);
            x2Gm.SetGlobalBuffer((__gm__ T*)x2);
            yGm.SetGlobalBuffer((__gm__ T*)y);
            x1Gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            x2Gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            this->L = GetBlockIdx() * LEN;
            this->R = size;
            pipe->InitBuffer(inQueueX1, BUFFER_NUM, LEN * sizeof(T));
            pipe->InitBuffer(inQueueX2, BUFFER_NUM, LEN * sizeof(T));
            pipe->InitBuffer(outQueueY, BUFFER_NUM, LEN * sizeof(T));

            pipe->InitBuffer(X1Buf, LEN * sizeof(T));
            pipe->InitBuffer(X2Buf, LEN * sizeof(T));
            if constexpr(std::is_same_v<T, float32_t>)
            {
                AscendC::LocalTensor<int32_t> X1 = X1Buf.Get<int32_t>();
                AscendC::LocalTensor<int32_t> X2 = X2Buf.Get<int32_t>();
                AscendC::Duplicate(X1, (int32_t)0x80000000, LEN);
                AscendC::Duplicate(X2, (int32_t)0x7fffffff, LEN);
            }
            else
            {
                AscendC::LocalTensor<int16_t> X1 = X1Buf.Get<int16_t>();
                AscendC::LocalTensor<int16_t> X2 = X2Buf.Get<int16_t>();
                AscendC::Duplicate(X1, (int16_t)0x8000, LEN);
                AscendC::Duplicate(X2, (int16_t)0x7fff, LEN);
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
            

            AscendC::LocalTensor<int16_t> x116 = x1Local.template ReinterpretCast<int16_t>();
            AscendC::LocalTensor<int16_t> x216 = x2Local.template ReinterpretCast<int16_t>();
            AscendC::LocalTensor<int16_t> X1 = X1Buf.Get<int16_t>();
            AscendC::LocalTensor<int16_t> X2 = X2Buf.Get<int16_t>();


            if constexpr(std::is_same_v<T, float32_t>)
            {
                AscendC::LocalTensor<int32_t> x132 = x1Local.template ReinterpretCast<int32_t>();
                AscendC::LocalTensor<int32_t> x232 = x2Local.template ReinterpretCast<int32_t>();
                AscendC::LocalTensor<int32_t> yl = yLocal.template ReinterpretCast<int32_t>();

                AscendC::And(x116, x116, X2, length * 2);
                AscendC::And(x216, x216, X1, length * 2);
                AscendC::Add(yl, x132, x232, length);
            }
            else
            {
                AscendC::LocalTensor<int16_t> yl = yLocal.template ReinterpretCast<int16_t>();

                AscendC::And(x116, x116, X2, length);
                AscendC::And(x216, x216, X1, length);
                AscendC::Add(yl, x116, x216, length);
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
        static constexpr int32_t LEN = (std::is_same_v<T, float32_t>)?3064:6128;
        TPipe* pipe;
        GlobalTensor<T> x1Gm, x2Gm, yGm;
        int32_t L, R;
        //create queue for input, in this case depth is equal to buffer num
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
        //create queue for output, in this case depth is equal to buffer num
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
        TBuf<AscendC::TPosition::VECCALC> X1Buf, X2Buf;
};

template<typename T> class CopysignBroad {
    public:
        __aicore__ inline CopysignBroad() {}
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

            pipe->InitBuffer(X1Buf, LEN * sizeof(T));
            pipe->InitBuffer(X2Buf, LEN * sizeof(T));
            if constexpr(std::is_same_v<T, float32_t>)
            {
                AscendC::LocalTensor<int32_t> X1 = X1Buf.Get<int32_t>();
                AscendC::LocalTensor<int32_t> X2 = X2Buf.Get<int32_t>();
                AscendC::Duplicate(X1, (int32_t)0x80000000, LEN);
                AscendC::Duplicate(X2, (int32_t)0x7fffffff, LEN);
            }
            else
            {
                AscendC::LocalTensor<int16_t> X1 = X1Buf.Get<int16_t>();
                AscendC::LocalTensor<int16_t> X2 = X2Buf.Get<int16_t>();
                AscendC::Duplicate(X1, (int16_t)0x8000, LEN);
                AscendC::Duplicate(X2, (int16_t)0x7fff, LEN);
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
        }/*
        shape1[10,1,5,1000]
        shape2[1,20,5,1000]
        shape:[10,20,5,1000]
        n1:[50000,5000,5000,1000]
        n2:[]
        900000 [9,0,0,0]
        */
        __aicore__ inline void Process() {
            int i;
            int len = shape[MAX_DIM_NUMBER - 1];
            for (i = L; i + len <= R; i+=len) {
                CopyIn(len);
                Compute(len);
                CopyOut(i, len);
                
                idx[2] ++;
                idx1 += n1[2];
                idx2 += n2[2];
                if (idx[2] == shape[2])
                {
                    idx[2] = 0;
                    idx1 -= n1[2] * shape[2];
                    idx2 -= n2[2] * shape[2];
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
            

            AscendC::LocalTensor<int16_t> x116 = x1Local.template ReinterpretCast<int16_t>();
            AscendC::LocalTensor<int16_t> x216 = x2Local.template ReinterpretCast<int16_t>();
            AscendC::LocalTensor<int16_t> X1 = X1Buf.Get<int16_t>();
            AscendC::LocalTensor<int16_t> X2 = X2Buf.Get<int16_t>();


            if constexpr(std::is_same_v<T, float32_t>)
            {
                AscendC::LocalTensor<int32_t> x132 = x1Local.template ReinterpretCast<int32_t>();
                AscendC::LocalTensor<int32_t> x232 = x2Local.template ReinterpretCast<int32_t>();
                AscendC::LocalTensor<int32_t> yl = yLocal.template ReinterpretCast<int32_t>();

                AscendC::And(x116, x116, X2, length * 2);
                AscendC::And(x216, x216, X1, length * 2);
                AscendC::Add(yl, x132, x232, length);
            }
            else
            {
                AscendC::LocalTensor<int16_t> yl = yLocal.template ReinterpretCast<int16_t>();

                AscendC::And(x116, x116, X2, length);
                AscendC::And(x216, x216, X1, length);
                AscendC::Add(yl, x116, x216, length);
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
        static constexpr int32_t LEN = ((std::is_same_v<T, float32_t>)?3064:6128);
        TPipe* pipe;
        GlobalTensor<T> x1Gm, x2Gm, yGm;
        int32_t L, R, idx1, idx2;
        //create queue for input, in this case depth is equal to buffer num
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
        //create queue for output, in this case depth is equal to buffer num
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
        TBuf<AscendC::TPosition::VECCALC> X1Buf, X2Buf;

        int32_t idx[MAX_DIM_NUMBER], shape[MAX_DIM_NUMBER], n1[MAX_DIM_NUMBER], n2[MAX_DIM_NUMBER];
};

extern "C" __global__ __aicore__ void copysign(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1))
    {
        CopysignScalar<DTYPE_X1> op;
        op.Init(x1, x2, y, tiling_data.size, tiling_data.length);
        op.Process();
    }
    else if (TILING_KEY_IS(2))
    {
        CopysignVector<DTYPE_X1> op;
        op.Init(x1, x2, y, tiling_data.size, tiling_data.length, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(3))
    {
        CopysignBroad<DTYPE_X1> op;
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