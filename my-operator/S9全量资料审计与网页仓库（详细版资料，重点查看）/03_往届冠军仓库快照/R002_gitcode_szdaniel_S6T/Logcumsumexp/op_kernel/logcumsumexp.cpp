#include "kernel_operator.h"
using namespace AscendC;
#include "mymath.h"

// always broad cast y
#define BUFFER_NUM 2
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define ROUND_UP(a, b) (((a) + (b) - 1) / (b) * (b))
#define ROUND_DOWN(a, b) ((a) / (b) * (b))
// std::log1p(std::exp(min - max)) + max;

template <typename dataType, uint32_t tileKey> class KernelLogcumsumexp;

template <typename T> class KernelLogcumsumexp<T, 0> {
public:
    __aicore__ inline KernelLogcumsumexp() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, LogcumsumexpTilingData tiling)
    {
        this->ADimLength = tiling.ADimLength;
        this->PDimLength = tiling.PDimLength;

        uint64_t eachBlockEle = tiling.ADimLength * tiling.PDimLength;

        if (GetBlockIdx() < tiling.formerNum) {
            this->tileNum = tiling.formerTileNum;
            this->tileLength = tiling.formerTileLength;
            this->lastTileLength = tiling.formerLastTileLength;
            this->blockLength = tiling.formerLength;

            xGm.SetGlobalBuffer((__gm__ T *)x + tiling.formerLength * GetBlockIdx() * eachBlockEle,
                                tiling.formerLength * eachBlockEle);
            zGm.SetGlobalBuffer((__gm__ T *)z + tiling.formerLength * GetBlockIdx() * eachBlockEle,
                                tiling.formerLength * eachBlockEle);
        } else {
            this->tileNum = tiling.tailTileNum;
            this->tileLength = tiling.tailTileLength;
            this->lastTileLength = tiling.tailLastTileLength;
            this->blockLength = tiling.tailLength;

            xGm.SetGlobalBuffer((__gm__ T *)x + tiling.formerLength * tiling.formerNum * eachBlockEle +
                                    tiling.tailLength * (GetBlockIdx() - tiling.formerNum) * eachBlockEle,
                                tiling.tailLength * eachBlockEle);
            zGm.SetGlobalBuffer((__gm__ T *)z + tiling.formerLength * tiling.formerNum * eachBlockEle +
                                    tiling.tailLength * (GetBlockIdx() - tiling.formerNum) * eachBlockEle,
                                tiling.tailLength * eachBlockEle);
        }

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(T));

        pipe.InitBuffer(tBuf[0], this->tileLength * sizeof(float));
        pipe.InitBuffer(tBuf[1], this->tileLength * sizeof(float));
        pipe.InitBuffer(tBuf[2], this->tileLength * sizeof(float));
        pipe.InitBuffer(tBuf[3], this->tileLength * sizeof(float));
        pipe.InitBuffer(accBuf, this->tileLength * sizeof(float));
    }

    __aicore__ inline void CopyIn(int64_t offset, int len)
    {
        LocalTensor<T> x = inQueueX.AllocTensor<T>();

        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x, xGm[offset], copyParams, padParams);

        inQueueX.EnQue(x);
    }

    __aicore__ inline void CopyOut(int64_t offset, int len)
    {
        LocalTensor<T> z = outQueueZ.DeQue<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPad(zGm[offset], z, copyParams);
        outQueueZ.FreeTensor(z);
    }

    __aicore__ inline void Compute1st(uint32_t len)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        // if constexpr (std::is_same_v<T, half>){
        //     printf("x[0]: %f\n", (float)xLocal(0));
        // }

        if constexpr (std::is_same_v<T, float>) {
            AscendC::LocalTensor<T> v_acc = accBuf.Get<T>();

            // Copy(v_acc, xLocal, len, 1, {0, 0, 0, 0});
            // Copy(zLocal, xLocal, len, 1, {0, 0, 0, 0});
            Adds(v_acc, xLocal, (float)0, len);
            Adds(zLocal, xLocal, (float)0, len);
        } else {
            AscendC::LocalTensor<float> v_acc = accBuf.Get<float>();
            AscendC::LocalTensor<float> vx_f32 = tBuf[3].Get<float>();


            Cast(vx_f32, xLocal, AscendC::RoundMode::CAST_NONE, len);
            Adds(v_acc, vx_f32, (float)0, len);
            // Copy(v_acc, vx_f32, len, 1, {0, 0, 0, 0});
            // Copy(zLocal, xLocal, len, 1, {0, 0, 0, 0});
            Cast(zLocal, vx_f32, AscendC::RoundMode::CAST_RINT, len);

            // if constexpr (std::is_same_v<T, half>){
            //     printf("vacc[0]: %f\n", (float)v_acc(0));
            //     printf("zLocal[0]: %f\n", (float)zLocal(0));
            // }
        }

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        // if constexpr (std::is_same_v<T, half>){
        //     printf("x[0]: %f\n", (float)xLocal(0));
        // }

        if constexpr (std::is_same_v<T, float>) {
            AscendC::LocalTensor<T> v_max = tBuf[0].Get<T>();
            AscendC::LocalTensor<T> v_min = tBuf[1].Get<T>();
            AscendC::LocalTensor<T> v_exp = tBuf[2].Get<T>();
            AscendC::LocalTensor<T> v_acc = accBuf.Get<T>();

            Max(v_max, xLocal, v_acc, len);
            Min(v_min, xLocal, v_acc, len);
            Sub(v_exp, v_min, v_max, len);
            Exp(v_exp, v_exp, len);
#if 1
            Adds(v_exp, v_exp, (T)1, len);
            Ln(v_exp, v_exp, len);
#else // cannot cal host api(log1p) from device
            for (int i = 0; i < len; i++) {
                float src = v_exp(i);
                v_exp(i) = mylog1pf(src);
            }
#endif
            Add(v_acc, v_exp, v_max, len);
            // Copy(zLocal, v_acc, len, 1, {0, 0, 0, 0});
            Adds(zLocal, v_acc, 0.f, len);

        } else {
            AscendC::LocalTensor<float> v_max = tBuf[0].Get<float>();
            AscendC::LocalTensor<float> v_min = tBuf[1].Get<float>();
            AscendC::LocalTensor<float> v_exp = tBuf[2].Get<float>();
            AscendC::LocalTensor<float> v_acc = accBuf.Get<float>();
            AscendC::LocalTensor<float> vx_f32 = tBuf[3].Get<float>();

            Cast(vx_f32, xLocal, AscendC::RoundMode::CAST_NONE, len);
            Max(v_max, vx_f32, v_acc, len);
            Min(v_min, vx_f32, v_acc, len);
            Sub(v_exp, v_min, v_max, len);
            Exp(v_exp, v_exp, len);
#if 1
            Adds(v_exp, v_exp, (float)1, len);
            Ln(v_exp, v_exp, len);
#else // cannot cal host api(log1p) from device
            for (int i = 0; i < len; i++) {
                float src = v_exp(i);
                v_exp(i) = mylog1pf(src);
            }
#endif
            Add(v_acc, v_exp, v_max, len);
            Cast(zLocal, v_acc, AscendC::RoundMode::CAST_RINT, len);

            // if constexpr (std::is_same_v<T, half>){
            //     printf("v_acc[0]: %f\n", (float)v_acc(0));
            // }
        }

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessLine(uint64_t base, int len)
    {
        // printf("base:0x%llx,ADimLength:%d,len:%d\n", base,ADimLength,len);
        {
            uint64_t offset = base;
            CopyIn(offset, len);
            Compute1st(len);
            CopyOut(offset, len);
        }
        for (int i = 1; i < ADimLength; i++) {
            uint64_t offset = base + i * PDimLength;
            // printf("i:%d, offset:%d\n",i,offset);
            CopyIn(offset, len);
            Compute(len);
            CopyOut(offset, len);
        }
    }

    __aicore__ inline void ProcessTile(uint64_t base)
    {
        for (int i = 0; i < tileNum - 1; i++) {
            uint64_t offset = base + i * tileLength;
            ProcessLine(offset, tileLength);
        }
        if (tileNum >= 1) {
            uint64_t offset = base + (tileNum - 1) * tileLength;
            ProcessLine(offset, lastTileLength);
        }
    }

    __aicore__ inline void Process()
    {
        // printf("blocklen:%d,tileNum:%d,ADimLength:%d\n", blockLength, tileNum, ADimLength);
        for (int i = 0; i < blockLength; i++) {
            uint64_t offset = i * ADimLength * PDimLength;
            ProcessTile(offset);
        }
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    TBuf<TPosition::VECCALC> tBuf[4];
    TBuf<TPosition::VECCALC> accBuf;

    GlobalTensor<T> xGm;
    GlobalTensor<T> zGm;

    uint32_t blockLength; // cur block length
    uint32_t tileLength;
    uint32_t lastTileLength;
    uint32_t tileNum;
    uint32_t ADimLength; //  along dim shape
    uint32_t PDimLength; //  [dim,dimnum-1] shapeSize
};


template <typename T> class KernelLogcumsumexp<T, 1> {
public:
    __aicore__ inline KernelLogcumsumexp() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, LogcumsumexpTilingData tiling)
    {
        this->ADimLength = tiling.ADimLength;
        this->PDimLength = tiling.PDimLength;

        uint64_t eachBlockEle = tiling.ADimLength * tiling.PDimLength;

        if (GetBlockIdx() < tiling.formerNum) {
            this->tileNum = tiling.formerTileNum;
            this->tileLength = tiling.formerTileLength;
            this->lastTileLength = tiling.formerLastTileLength;
            this->blockLength = tiling.formerLength;

            xGm.SetGlobalBuffer((__gm__ T *)x + tiling.formerLength * GetBlockIdx() * eachBlockEle,
                                tiling.formerLength * eachBlockEle);
            zGm.SetGlobalBuffer((__gm__ T *)z + tiling.formerLength * GetBlockIdx() * eachBlockEle,
                                tiling.formerLength * eachBlockEle);
        } else {
            this->tileNum = tiling.tailTileNum;
            this->tileLength = tiling.tailTileLength;
            this->lastTileLength = tiling.tailLastTileLength;
            this->blockLength = tiling.tailLength;

            xGm.SetGlobalBuffer((__gm__ T *)x + tiling.formerLength * tiling.formerNum * eachBlockEle +
                                    tiling.tailLength * (GetBlockIdx() - tiling.formerNum) * eachBlockEle,
                                tiling.tailLength * eachBlockEle);
            zGm.SetGlobalBuffer((__gm__ T *)z + tiling.formerLength * tiling.formerNum * eachBlockEle +
                                    tiling.tailLength * (GetBlockIdx() - tiling.formerNum) * eachBlockEle,
                                tiling.tailLength * eachBlockEle);
        }

        tileLength = 4096;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(T));

        pipe.InitBuffer(tBuf[0], this->tileLength * sizeof(float));
        pipe.InitBuffer(tBuf[1], this->tileLength * sizeof(float));
        pipe.InitBuffer(tBuf[2], this->tileLength * sizeof(float));
        pipe.InitBuffer(accBuf, this->tileLength * sizeof(float));
    }

    __aicore__ inline void CopyIn(int64_t offset, int len)
    {
        LocalTensor<T> x = inQueueX.AllocTensor<T>();

        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x, xGm[offset], copyParams, padParams);

        inQueueX.EnQue(x);
    }

    __aicore__ inline void CopyOut(int64_t offset, int len)
    {
        LocalTensor<T> z = outQueueZ.DeQue<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPad(zGm[offset], z, copyParams);
        outQueueZ.FreeTensor(z);
    }

    __aicore__ inline void Compute1st(uint32_t len)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        AscendC::LocalTensor<float> v_acc = accBuf.Get<float>();

        zLocal(0) = xLocal(0);
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            // Cast(v_acc, xLocal, AscendC::RoundMode::CAST_NONE, 1);
            T x = xLocal(0);
            v_acc(0) = ToFloat(x);
        } else {
            v_acc(0) = (float)xLocal(0);
        }

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        AscendC::LocalTensor<float> v_acc = accBuf.Get<float>();
        AscendC::LocalTensor<float> v_exp = tBuf[2].Get<float>();

        if constexpr (std::is_same_v<T, float>) {
            for (int i = 0; i < len; i++) {
                float src = xLocal(i);
                float acc = v_acc(0);
                float min = MIN(acc, src);
                float max = MAX(acc, src);
#if 0
                float t_exp = myexpf(min - max);
                acc = max + mylog1pf(t_exp);
#else
                v_exp(0) = min - max;
                Exp(v_exp, v_exp, 1);
                acc = max + mylog1pf(v_exp(0));
#endif
                v_acc(0) = acc;
                zLocal(i) = acc;
            }
        } else {
            AscendC::LocalTensor<float> xF32 = tBuf[0].Get<float>();
            AscendC::LocalTensor<float> zF32 = tBuf[1].Get<float>();

            Cast(xF32, xLocal, AscendC::RoundMode::CAST_NONE, len);

            for (int i = 0; i < len; i++) {
                float src = xF32(i);
                float acc = v_acc(0);
                float min = MIN(acc, src);
                float max = MAX(acc, src);
#if 0
                float t_exp = myexpf(min - max);
                acc = max + mylog1pf(t_exp);
#else
                v_exp(0) = min - max;
                Exp(v_exp, v_exp, 1);
                acc = max + mylog1pf(v_exp(0));
#endif
                v_acc(0) = acc;
                zF32(i) = acc;
            }
            Cast(zLocal, zF32, AscendC::RoundMode::CAST_RINT, len);
        }
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessLine(uint64_t base)
    {
        {
            uint64_t offset = base;
            uint32_t len = 1;

            CopyIn(offset, len);
            Compute1st(len);
            CopyOut(offset, len);
        }
        for (int i = 1; i < ADimLength; i += tileLength) {
            uint64_t offset = base + i;
            uint32_t len = MIN(tileLength, ADimLength - i);
            // printf("i:%d,offset:%lld,len:%d\n",i,offset,len);

            CopyIn(offset, len);
            Compute(len);
            CopyOut(offset, len);
        }
    }


    __aicore__ inline void Process()
    {
        // printf("blocklen:%d,ADimLength:%d\n", blockLength, ADimLength);
        for (int i = 0; i < blockLength; i++) {
            uint64_t offset = i * ADimLength * PDimLength;
            ProcessLine(offset);
        }
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    TBuf<TPosition::VECCALC> tBuf[4];
    TBuf<TPosition::VECCALC> accBuf;

    GlobalTensor<T> xGm;
    GlobalTensor<T> zGm;

    uint32_t blockLength; // cur block length
    uint32_t tileLength;
    uint32_t lastTileLength;
    uint32_t tileNum;
    uint32_t ADimLength; //  along dim shape
    uint32_t PDimLength; //  [dim,dimnum-1] shapeSize
};


#if 0
template <typename T> class KernelLogcumsumexp<T, 1> {
public:
    __aicore__ inline KernelLogcumsumexp() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, LogcumsumexpTilingData tiling)
    {
        this->ADimLength = tiling.ADimLength;
        this->PDimLength = tiling.PDimLength;

        uint64_t eachBlockEle = tiling.ADimLength * tiling.PDimLength;

        if (GetBlockIdx() < tiling.formerNum) {
            this->tileNum = tiling.formerTileNum;
            this->tileLength = tiling.formerTileLength;
            this->lastTileLength = tiling.formerLastTileLength;
            this->blockLength = tiling.formerLength;

            xGm.SetGlobalBuffer((__gm__ T *)x + tiling.formerLength * GetBlockIdx() * eachBlockEle,
                                tiling.formerLength * eachBlockEle);
            zGm.SetGlobalBuffer((__gm__ T *)z + tiling.formerLength * GetBlockIdx() * eachBlockEle,
                                tiling.formerLength * eachBlockEle);
        } else {
            this->tileNum = tiling.tailTileNum;
            this->tileLength = tiling.tailTileLength;
            this->lastTileLength = tiling.tailLastTileLength;
            this->blockLength = tiling.tailLength;

            xGm.SetGlobalBuffer((__gm__ T *)x + tiling.formerLength * tiling.formerNum * eachBlockEle +
                                    tiling.tailLength * (GetBlockIdx() - tiling.formerNum) * eachBlockEle,
                                tiling.tailLength * eachBlockEle);
            zGm.SetGlobalBuffer((__gm__ T *)z + tiling.formerLength * tiling.formerNum * eachBlockEle +
                                    tiling.tailLength * (GetBlockIdx() - tiling.formerNum) * eachBlockEle,
                                tiling.tailLength * eachBlockEle);
        }

        tileLength = 1024; // 4096;
        lastAccVal = 0;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(T));

        pipe.InitBuffer(tBuf[0], this->tileLength * sizeof(float));
        pipe.InitBuffer(tBuf[1], this->tileLength * sizeof(float));
        pipe.InitBuffer(tBuf[2], this->tileLength * sizeof(float));
        pipe.InitBuffer(tBuf[3], this->tileLength * sizeof(float));
        pipe.InitBuffer(tBuf[4], this->tileLength * sizeof(float) * 32);
        pipe.InitBuffer(accBuf, this->tileLength * sizeof(float));
    }

    __aicore__ inline void CopyIn(int64_t offset, int len)
    {
        LocalTensor<T> x = inQueueX.AllocTensor<T>();

        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x, xGm[offset], copyParams, padParams);

        inQueueX.EnQue(x);
    }

    __aicore__ inline void CopyOut(int64_t offset, int len)
    {
        LocalTensor<T> z = outQueueZ.DeQue<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPad(zGm[offset], z, copyParams);
        outQueueZ.FreeTensor(z);
    }

    __aicore__ inline void ComputeLn(uint32_t len)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        AscendC::LocalTensor<float> vx_f32 = tBuf[0].Get<float>();
        AscendC::LocalTensor<float> v_max_val = tBuf[1].Get<float>();
        AscendC::LocalTensor<float> v_exp = tBuf[2].Get<float>();
        AscendC::LocalTensor<float> v_shift = tBuf[3].Get<float>();
        AscendC::LocalTensor<float> v_acc = accBuf.Get<float>();
        AscendC::LocalTensor<uint8_t> v_tmp = tBuf[4].Get<uint8_t>();

        Duplicate(v_max_val, (float)max_val, len);

        if constexpr (std::is_same_v<T, float>) {
            Sub(xLocal, xLocal, v_max_val, len);
            Exp(v_exp, xLocal, len);

            v_acc(0) = v_exp(0) + lastAccVal;
            for (int i = 1; i < len; i++) {
                v_acc(i) = v_acc(i - 1) + v_exp(i);
            }
            lastAccVal = v_acc(len - 1);

            Ln(v_acc, v_acc, len);
            Adds(zLocal, v_acc, (float)max_val, len);
        } else {
            Cast(vx_f32, xLocal, AscendC::RoundMode::CAST_NONE, len);
            Sub(vx_f32, vx_f32, v_max_val, len);
            // for(int i = 0;i<len;i++){
            //     float tt = vx_f32(i);
            //     if (tt<0) tt = -tt;//#define __FLT_EPSILON__ 1.19209289550781250000000000000000000e-7F
            //     if(tt<1.19209289550781250000000000000000000e-7F) vx_f32(i) = 0;
            // }
            Exp(v_exp, vx_f32, len);
            // for(int i = 0;i<len;i++){
            //     float tt = v_exp(i);
            //     if (tt<0) tt = -tt;//#define __FLT_EPSILON__ 1.19209289550781250000000000000000000e-7F
            //     if(tt<1.19209289550781250000000000000000000e-7F) v_exp(i) = 0;
            // }

            // v_acc(0) = v_exp(0) + lastAccVal;
            // for (int i = 1; i < len; i++) {
            //     v_acc(i) = v_acc(i - 1) + v_exp(i);
            // }
            // lastAccVal = v_acc(len - 1);

            // for (int i = 0; i < len; i++) {
            //     v_acc(i) = lastAccVal;
            //     lastAccVal += v_exp(i);
            // }

            static constexpr AscendC::CumSumConfig cumSumConfig{true, false, true};
            const AscendC::CumSumInfo cumSumInfo{1, len};
            AscendC::CumSum<float, cumSumConfig>(v_acc, vx_f32, v_exp, v_tmp, cumSumInfo);
            Adds(v_acc, v_acc, lastAccVal, len);
            lastAccVal = v_acc(len - 1);

            Ln(v_acc, v_acc, len);
            Adds(v_acc, v_acc, (float)max_val, len);
            Cast(zLocal, v_acc, AscendC::RoundMode::CAST_RINT, len);
        }

        outQueueZ.EnQue<T>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeMax(int32_t len)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();

        if constexpr (std::is_same_v<T, bfloat16_t>) {
            AscendC::LocalTensor<float> v_max = tBuf[0].Get<float>();
            AscendC::LocalTensor<float> work_local = tBuf[1].Get<float>();
            AscendC::LocalTensor<float> xF32 = tBuf[2].Get<float>();

            Cast(xF32, xLocal, AscendC::RoundMode::CAST_NONE, len);
            AscendC::ReduceMax(v_max, xF32, work_local, len);
            float tmp_max = v_max(0);
            if (tmp_max > max_val) {
                max_val = tmp_max;
            }
        } else {
            AscendC::LocalTensor<T> v_max = tBuf[0].Get<T>();
            AscendC::LocalTensor<T> work_local = tBuf[1].Get<T>();

            AscendC::ReduceMax(v_max, xLocal, work_local, len);
            float tmp_max = v_max(0);
            if (tmp_max > max_val) {
                max_val = (T)tmp_max;
            }
        }


        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessLine(uint64_t base)
    {
        // find max
        for (int i = 0; i < ADimLength; i += tileLength) {
            uint64_t offset = base + i;
            uint32_t len = MIN(tileLength, ADimLength - i);

            CopyIn(offset, len);
            ComputeMax(len);
        }
        max_val = 0;
        // printf("max val:%f\n", max_val);
        // calc cumsum
        for (int i = 0; i < ADimLength; i += tileLength) {
            uint64_t offset = base + i;
            uint32_t len = MIN(tileLength, ADimLength - i);

            CopyIn(offset, len);
            ComputeLn(len);
            CopyOut(offset, len);
        }
    }


    __aicore__ inline void Process()
    {
        // printf("blocklen:%d,ADimLength:%d\n", blockLength, ADimLength);
        for (int i = 0; i < blockLength; i++) {
            uint64_t offset = i * ADimLength * PDimLength;
            ProcessLine(offset);
        }
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    TBuf<TPosition::VECCALC> tBuf[5];
    TBuf<TPosition::VECCALC> accBuf;

    GlobalTensor<T> xGm;
    GlobalTensor<T> zGm;

    uint32_t blockLength; // cur block length
    uint32_t tileLength;
    uint32_t lastTileLength;
    uint32_t tileNum;
    uint32_t ADimLength; //  along dim shape
    uint32_t PDimLength; //  [dim,dimnum-1] shapeSize


    float max_val;
    float lastAccVal;
};

#endif

extern "C" __global__ __aicore__ void logcumsumexp(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    if (TILING_KEY_IS(0)) {
        KernelLogcumsumexp<DTYPE_X, 0> op;
        op.Init(x, z, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelLogcumsumexp<DTYPE_X, 1> op;
        op.Init(x, z, tiling_data);
        op.Process();
    }
}