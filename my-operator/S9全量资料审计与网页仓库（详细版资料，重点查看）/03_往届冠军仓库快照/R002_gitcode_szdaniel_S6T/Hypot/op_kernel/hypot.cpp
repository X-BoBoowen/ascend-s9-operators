#include "kernel_operator.h"

using namespace AscendC;

// always broad cast y
#define BUFFER_NUM 2
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define ROUND_UP(a, b) (((a) + (b) - 1) / (b) * (b))
#define ROUND_DOWN(a, b) ((a) / (b) * (b))

template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
__aicore__ inline void MyBroadCast(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
                                   const uint32_t dstShape[dim], const uint32_t srcShape[dim])
{
    if constexpr (dim == 2) {
        if constexpr (axis == 0) {
            for (int i = 0; i < dstShape[0]; i++) {
                for (int j = 0; j < dstShape[1]; j++) {
                    dstLocal(i * dstShape[1] + j) = srcLocal(j);
                }
            }
        } else {
            for (int i = 0; i < dstShape[0]; i++) {
                for (int j = 0; j < dstShape[1]; j++) {
                    dstLocal(i * dstShape[1] + j) = srcLocal(i);
                }
            }
        }
    }
}
template <typename dataType, uint32_t tileKey> class KernelHypot;

template <typename T> class KernelHypot<T, 0> {
public:
    __aicore__ inline KernelHypot() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, HypotTilingData tiling, TPipe *pipe)
    {
        if (GetBlockIdx() < tiling.formerNum) {
            this->tileNum = tiling.formerTileNum;
            this->tileLength = tiling.formerTileLength;
            this->lastTileLength = tiling.formerLastTileLength;

            xGm.SetGlobalBuffer((__gm__ T *)x + tiling.formerLength * GetBlockIdx(), tiling.formerLength);
            yGm.SetGlobalBuffer((__gm__ T *)y + tiling.formerLength * GetBlockIdx(), tiling.formerLength);
            zGm.SetGlobalBuffer((__gm__ T *)z + tiling.formerLength * GetBlockIdx(), tiling.formerLength);
        } else {
            this->tileNum = tiling.tailTileNum;
            this->tileLength = tiling.tailTileLength;
            this->lastTileLength = tiling.tailLastTileLength;

            xGm.SetGlobalBuffer((__gm__ T *)x + tiling.formerLength * tiling.formerNum +
                                    tiling.tailLength * (GetBlockIdx() - tiling.formerNum),
                                tiling.tailLength);
            yGm.SetGlobalBuffer((__gm__ T *)y + tiling.formerLength * tiling.formerNum +
                                    tiling.tailLength * (GetBlockIdx() - tiling.formerNum),
                                tiling.tailLength);
            zGm.SetGlobalBuffer((__gm__ T *)z + tiling.formerLength * tiling.formerNum +
                                    tiling.tailLength * (GetBlockIdx() - tiling.formerNum),
                                tiling.tailLength);
        }
        xGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        zGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        pipe->InitBuffer(inQueueX, BUFFER_NUM, 4096 * sizeof(T));
        pipe->InitBuffer(inQueueY, BUFFER_NUM, 4096 * sizeof(T));
        pipe->InitBuffer(outQueueZ, BUFFER_NUM, 4096 * sizeof(T));
    }

    __aicore__ inline void CopyIn(int64_t offset, int len)
    {
        LocalTensor<T> x = inQueueX.AllocTensor<T>();
        LocalTensor<T> y = inQueueY.AllocTensor<T>();

        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x, xGm[offset], copyParams, padParams);
        DataCopyPad(y, yGm[offset], copyParams, padParams);

        inQueueX.EnQue(x);
        inQueueY.EnQue(y);
    }

    __aicore__ inline void CopyOut(int64_t offset, int len)
    {
        LocalTensor<T> z = outQueueZ.DeQue<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPad(zGm[offset], z, copyParams);
        outQueueZ.FreeTensor(z);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        AscendC::Mul<T>(xLocal, xLocal, xLocal, len);
        AscendC::MulAddDst<T>(xLocal, yLocal, yLocal, len);
        AscendC::Sqrt<T>(zLocal, xLocal, len);

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void Process()
    {
        for (int i = 0; i < tileNum - 1; i++) {
            int offset = i * tileLength;
            CopyIn(offset, tileLength);
            Compute(tileLength);
            CopyOut(offset, tileLength);
        }
        int offset = (tileNum - 1) * tileLength;
        CopyIn(offset, lastTileLength);
        Compute(lastTileLength);
        CopyOut(offset, lastTileLength);
    }

private:

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    TBuf<TPosition::VECCALC> tBuf;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;

    int tileLength;
    int tileNum;
    int lastTileLength;
};

template <> class KernelHypot<bfloat16_t, 0> {
public:
    __aicore__ inline KernelHypot() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, HypotTilingData tiling,TPipe *pipe)
    {
        if (GetBlockIdx() < tiling.formerNum) {
            this->tileNum = tiling.formerTileNum;
            this->tileLength = tiling.formerTileLength;
            this->lastTileLength = tiling.formerLastTileLength;

            xGm.SetGlobalBuffer((__gm__ bfloat16_t *)x + tiling.formerLength * GetBlockIdx(), tiling.formerLength);
            yGm.SetGlobalBuffer((__gm__ bfloat16_t *)y + tiling.formerLength * GetBlockIdx(), tiling.formerLength);
            zGm.SetGlobalBuffer((__gm__ bfloat16_t *)z + tiling.formerLength * GetBlockIdx(), tiling.formerLength);
        } else {
            this->tileNum = tiling.tailTileNum;
            this->tileLength = tiling.tailTileLength;
            this->lastTileLength = tiling.tailLastTileLength;

            xGm.SetGlobalBuffer((__gm__ bfloat16_t *)x + tiling.formerLength * tiling.formerNum +
                                    tiling.tailLength * (GetBlockIdx() - tiling.formerNum),
                                tiling.tailLength);
            yGm.SetGlobalBuffer((__gm__ bfloat16_t *)y + tiling.formerLength * tiling.formerNum +
                                    tiling.tailLength * (GetBlockIdx() - tiling.formerNum),
                                tiling.tailLength);
            zGm.SetGlobalBuffer((__gm__ bfloat16_t *)z + tiling.formerLength * tiling.formerNum +
                                    tiling.tailLength * (GetBlockIdx() - tiling.formerNum),
                                tiling.tailLength);
        }

        pipe->InitBuffer(inQueueX, BUFFER_NUM, 4096 * sizeof(bfloat16_t));
        pipe->InitBuffer(inQueueY, BUFFER_NUM, 4096 * sizeof(bfloat16_t));
        pipe->InitBuffer(outQueueZ, BUFFER_NUM, 4096 * sizeof(bfloat16_t));

        pipe->InitBuffer(tBuf[0], 4096 * sizeof(float));
        pipe->InitBuffer(tBuf[1], 4096 * sizeof(float));
        pipe->InitBuffer(tBuf[2], 4096 * sizeof(float));
    }

    __aicore__ inline void CopyIn(int64_t offset, int len)
    {
        LocalTensor<bfloat16_t> x = inQueueX.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> y = inQueueY.AllocTensor<bfloat16_t>();

        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
        DataCopyPad(x, xGm[offset], copyParams, padParams);
        DataCopyPad(y, yGm[offset], copyParams, padParams);

        inQueueX.EnQue(x);
        inQueueY.EnQue(y);
    }

    __aicore__ inline void CopyOut(int64_t offset, int len)
    {
        LocalTensor<bfloat16_t> z = outQueueZ.DeQue<bfloat16_t>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(zGm[offset], z, copyParams);
        outQueueZ.FreeTensor(z);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<bfloat16_t> xLocal = inQueueX.DeQue<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> yLocal = inQueueY.DeQue<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> zLocal = outQueueZ.AllocTensor<bfloat16_t>();

        LocalTensor<float> tmpTensor1 = tBuf[0].Get<float>();
        LocalTensor<float> tmpTensor2 = tBuf[1].Get<float>();
        LocalTensor<float> tmpTensor3 = tBuf[2].Get<float>();

        AscendC::Cast(tmpTensor1, xLocal, AscendC::RoundMode::CAST_NONE, len);
        AscendC::Cast(tmpTensor2, yLocal, AscendC::RoundMode::CAST_NONE, len);
        AscendC::Mul(tmpTensor1, tmpTensor1, tmpTensor1, len);
        AscendC::Mul(tmpTensor2, tmpTensor2, tmpTensor2, len);
        AscendC::Add(tmpTensor3, tmpTensor1, tmpTensor2, len);
        AscendC::Sqrt(tmpTensor3, tmpTensor3, len);
        AscendC::Cast(zLocal, tmpTensor3, AscendC::RoundMode::CAST_RINT, len);

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void Process()
    {
        for (int i = 0; i < tileNum - 1; i++) {
            int offset = i * tileLength;
            CopyIn(offset, tileLength);
            Compute(tileLength);
            CopyOut(offset, tileLength);
        }
        int offset = (tileNum - 1) * tileLength;
        CopyIn(offset, lastTileLength);
        Compute(lastTileLength);
        CopyOut(offset, lastTileLength);
    }

private:
    // TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    TBuf<TPosition::VECCALC> tBuf[3];

    GlobalTensor<bfloat16_t> xGm;
    GlobalTensor<bfloat16_t> yGm;
    GlobalTensor<bfloat16_t> zGm;

    int tileLength;
    int tileNum;
    int lastTileLength;
};

template <typename T> class KernelHypot<T, 1> {
public:
    __aicore__ inline KernelHypot() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, HypotTilingData tiling)
    {
        this->dimnum = tiling.dimnum;
        bool switch_xy = false;
        if (tiling.xShape[0] < tiling.yShape[0]) {
            switch_xy = true;
        }

        GM_ADDR x_new = x;
        GM_ADDR y_new = y;
        if (switch_xy) {
            xShapeNew[0] = tiling.yShape[0];
            yShapeNew[0] = tiling.xShape[0];
            zShapeNew[0] = tiling.zShape[0];
            x_new = y;
            y_new = x;
        } else {
            xShapeNew[0] = tiling.xShape[0];
            yShapeNew[0] = tiling.yShape[0];
            zShapeNew[0] = tiling.zShape[0];
        }

        if (GetBlockIdx() < tiling.formerNum) {
            this->tileNum = tiling.formerTileNum;
            this->tileLength = tiling.formerTileLength;
            this->lastTileLength = tiling.formerLastTileLength;

            xGm.SetGlobalBuffer((__gm__ T *)x_new + tiling.formerLength * GetBlockIdx(), tiling.formerLength);
            yGm.SetGlobalBuffer((__gm__ T *)y_new, 1);
            zGm.SetGlobalBuffer((__gm__ T *)z + tiling.formerLength * GetBlockIdx(), tiling.formerLength);
        } else {
            this->tileNum = tiling.tailTileNum;
            this->tileLength = tiling.tailTileLength;
            this->lastTileLength = tiling.tailLastTileLength;

            xGm.SetGlobalBuffer((__gm__ T *)x_new + tiling.formerLength * tiling.formerNum +
                                    tiling.tailLength * (GetBlockIdx() - tiling.formerNum),
                                tiling.tailLength);
            yGm.SetGlobalBuffer((__gm__ T *)y_new, 1);
            zGm.SetGlobalBuffer((__gm__ T *)z + tiling.formerLength * tiling.formerNum +
                                    tiling.tailLength * (GetBlockIdx() - tiling.formerNum),
                                tiling.tailLength);
        }

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(T));

        pipe.InitBuffer(tBuf[0], this->tileLength * sizeof(T));
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe.InitBuffer(tBuf[1], this->tileLength * sizeof(float));
            pipe.InitBuffer(tBuf[2], this->tileLength * sizeof(float));
            pipe.InitBuffer(tBuf[3], this->tileLength * sizeof(float));
        }
    }

    __aicore__ inline void CopyInX(int64_t offset, int len)
    {
        LocalTensor<T> x = inQueueX.AllocTensor<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(x);
    }

    __aicore__ inline void CopyInY()
    {
        LocalTensor<T> y = inQueueY.AllocTensor<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(1 * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(y, yGm[0], copyParams, padParams);
        inQueueY.EnQue(y);
    }

    __aicore__ inline void CopyOut(int64_t offset, int len)
    {
        LocalTensor<T> z = outQueueZ.DeQue<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPad(zGm[offset], z, copyParams);
        outQueueZ.FreeTensor(z);
    }

    __aicore__ inline void Compute1D(uint32_t len)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        AscendC::LocalTensor<T> broadcastTmpTensor = tBuf[0].Get<T>();

        uint32_t dstShape[] = {1, len};
        uint32_t srcShape[] = {1, 1};
        MyBroadCast<T, 2, 1>(broadcastTmpTensor, yLocal, dstShape, srcShape);

        if constexpr (std::is_same_v<T, bfloat16_t>) {
            AscendC::LocalTensor<float> tmpx = tBuf[1].Get<float>();
            AscendC::LocalTensor<float> tmpy = tBuf[2].Get<float>();
            AscendC::LocalTensor<float> tmpz = tBuf[3].Get<float>();

            AscendC::Cast(tmpx, xLocal, AscendC::RoundMode::CAST_NONE, len);
            AscendC::Cast(tmpy, broadcastTmpTensor, AscendC::RoundMode::CAST_NONE, len);
            AscendC::Mul(tmpx, tmpx, tmpx, len);
            AscendC::Mul(tmpy, tmpy, tmpy, len);
            AscendC::Add(tmpz, tmpy, tmpx, len);
            AscendC::Sqrt(tmpz, tmpz, len);
            AscendC::Cast(zLocal, tmpz, AscendC::RoundMode::CAST_RINT, len);
        } else {
            AscendC::Mul<T>(xLocal, xLocal, xLocal, len);
            AscendC::Mul<T>(broadcastTmpTensor, broadcastTmpTensor, broadcastTmpTensor, len);
            AscendC::Add<T>(zLocal, xLocal, broadcastTmpTensor, len);
            AscendC::Sqrt<T>(zLocal, zLocal, len);
        }
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void Process()
    {
        for (int i = 0; i < tileNum - 1; i++) {
            int offset = i * tileLength;
            CopyInX(offset, tileLength);
            CopyInY();
            Compute1D(tileLength);
            CopyOut(offset, tileLength);
        }
        int offset = (tileNum - 1) * tileLength;
        CopyInX(offset, lastTileLength);
        CopyInY();
        Compute1D(lastTileLength);
        CopyOut(offset, lastTileLength);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    TBuf<TPosition::VECCALC> tBuf[4];

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;

    int tileLength;
    int tileNum;
    int lastTileLength;
    int dimnum;
    int xShapeNew[4];
    int yShapeNew[4];
    int zShapeNew[4];
};


template <typename T> class KernelHypot<T, 2> {
public:
    __aicore__ inline KernelHypot() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, HypotTilingData tiling)
    {
        this->dimnum = tiling.dimnum;
        bool switch_xy = false;
        if (tiling.xShape[dimnum - 1] < tiling.yShape[dimnum - 1]) {
            switch_xy = true;
        }

        GM_ADDR x_new = x;
        GM_ADDR y_new = y;
        if (switch_xy) {
            for (int i = 0; i < dimnum; i++) {
                xShapeNew[i] = tiling.yShape[i];
                yShapeNew[i] = tiling.xShape[i];
                zShapeNew[i] = tiling.zShape[i];
            }
            x_new = y;
            y_new = x;
        } else {
            for (int i = 0; i < dimnum; i++) {
                xShapeNew[i] = tiling.xShape[i];
                yShapeNew[i] = tiling.yShape[i];
                zShapeNew[i] = tiling.zShape[i];
            }
        }
        lastDimLength = zShapeNew[dimnum - 1];

        int xLen = 1;
        int yLen = 1;
        int zLen = 1;
        for (int i = 1; i < dimnum; i++) {
            xLen *= xShapeNew[i];
            yLen *= yShapeNew[i];
            zLen *= zShapeNew[i];
        }

        if (GetBlockIdx() < tiling.formerNum) {
            this->tileNum = tiling.formerTileNum;
            this->tileLength = tiling.formerTileLength;
            this->lastTileLength = tiling.formerLastTileLength;

            xGm.SetGlobalBuffer((__gm__ T *)x_new + (GetBlockIdx() * tiling.formerLength) % xShapeNew[0] * xLen,
                                xLen * tiling.formerLength);
            yGm.SetGlobalBuffer((__gm__ T *)y_new + (GetBlockIdx() * tiling.formerLength) % yShapeNew[0] * yLen,
                                yLen * tiling.formerLength);
            zGm.SetGlobalBuffer((__gm__ T *)z + GetBlockIdx() * tiling.formerLength * zLen, zLen * tiling.formerLength);
        } else {
            this->tileNum = tiling.tailTileNum;
            this->tileLength = tiling.tailTileLength;
            this->lastTileLength = tiling.tailLastTileLength;

            xGm.SetGlobalBuffer((__gm__ T *)x_new + (tiling.formerLength * tiling.formerNum +
                                                     (GetBlockIdx() - tiling.formerNum) * tiling.tailLength) %
                                                        xShapeNew[0] * xLen,
                                xLen * MIN(xShapeNew[0], tiling.tailLength));
            yGm.SetGlobalBuffer((__gm__ T *)y_new + (tiling.formerLength * tiling.formerNum +
                                                     (GetBlockIdx() - tiling.formerNum) * tiling.tailLength) %
                                                        yShapeNew[0] * yLen,
                                yLen * MIN(yShapeNew[0], tiling.tailLength));
            zGm.SetGlobalBuffer((__gm__ T *)z + (tiling.formerLength * tiling.formerNum +
                                                 (GetBlockIdx() - tiling.formerNum) * tiling.tailLength) *
                                                    zLen,
                                zLen * tiling.tailLength);
        }

        pipe.InitBuffer(inQueueX, BUFFER_NUM,
                        MIN(tileLength, xShapeNew[dimnum - 2]) * xShapeNew[dimnum - 1] * sizeof(T));
        pipe.InitBuffer(inQueueY, BUFFER_NUM,
                        MIN(tileLength, yShapeNew[dimnum - 2]) * yShapeNew[dimnum - 1] * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * zShapeNew[dimnum - 1] * sizeof(T));

        pipe.InitBuffer(tBuf[0], this->tileLength * lastDimLength * sizeof(T));
        pipe.InitBuffer(tBuf[1], this->tileLength * lastDimLength * sizeof(T));
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe.InitBuffer(tBuf[2], this->tileLength * lastDimLength * sizeof(float));
            pipe.InitBuffer(tBuf[3], this->tileLength * lastDimLength * sizeof(float));
            pipe.InitBuffer(tBuf[4], this->tileLength * lastDimLength * sizeof(float));
        }
    }

    __aicore__ inline void CopyInX(int64_t offset, int len)
    {
        LocalTensor<T> x = inQueueX.AllocTensor<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(x);
    }

    __aicore__ inline void CopyInY(int64_t offset, int len)
    {
        LocalTensor<T> y = inQueueY.AllocTensor<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(y, yGm[offset], copyParams, padParams);
        inQueueY.EnQue(y);
    }

    __aicore__ inline void CopyOut(int64_t offset, int len)
    {
        LocalTensor<T> z = outQueueZ.DeQue<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPad(zGm[offset], z, copyParams);
        outQueueZ.FreeTensor(z);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        AscendC::LocalTensor<T> broadcastTmpTensor = tBuf[0].Get<T>();

        uint32_t dstShape[] = {len, lastDimLength};
        uint32_t srcShape[] = {len, 1};
        MyBroadCast<T, 2, 1>(broadcastTmpTensor, yLocal, dstShape, srcShape);

        uint32_t compute_len = len * lastDimLength;

        if constexpr (std::is_same_v<T, bfloat16_t>) {
            AscendC::LocalTensor<float> xLocalFloat = tBuf[2].Get<float>();
            AscendC::LocalTensor<float> yLocalFloat = tBuf[3].Get<float>();
            AscendC::LocalTensor<float> zLocalFloat = tBuf[4].Get<float>();

            AscendC::Cast(xLocalFloat, xLocal, AscendC::RoundMode::CAST_NONE, compute_len);
            AscendC::Cast(yLocalFloat, broadcastTmpTensor, AscendC::RoundMode::CAST_NONE, compute_len);
            AscendC::Mul(xLocalFloat, xLocalFloat, xLocalFloat, compute_len);
            AscendC::Mul(yLocalFloat, yLocalFloat, yLocalFloat, compute_len);
            AscendC::Add(zLocalFloat, xLocalFloat, yLocalFloat, compute_len);
            AscendC::Sqrt(zLocalFloat, zLocalFloat, compute_len);
            AscendC::Cast(zLocal, zLocalFloat, AscendC::RoundMode::CAST_RINT, compute_len);
        } else {
            AscendC::Mul(xLocal, xLocal, xLocal, compute_len);
            AscendC::Mul(broadcastTmpTensor, broadcastTmpTensor, broadcastTmpTensor, compute_len);
            AscendC::Add(zLocal, xLocal, broadcastTmpTensor, compute_len);
            AscendC::Sqrt(zLocal, zLocal, compute_len);
        }
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void ComputeB2(uint32_t len)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        AscendC::LocalTensor<T> xBroadcast = tBuf[0].Get<T>();
        AscendC::LocalTensor<T> yBroadcast = tBuf[1].Get<T>();

        {
            uint32_t dstShape[] = {len, lastDimLength};
            uint32_t srcShape[] = {1, lastDimLength};
            MyBroadCast<T, 2, 0>(xBroadcast, xLocal, dstShape, srcShape);
        }
        {
            uint32_t dstShape[] = {len, lastDimLength};
            uint32_t srcShape[] = {len, 1};
            MyBroadCast<T, 2, 1>(yBroadcast, yLocal, dstShape, srcShape);
        }

        uint32_t compute_len = len * lastDimLength;

        if constexpr (std::is_same_v<T, bfloat16_t>) {
            AscendC::LocalTensor<float> xLocalFloat = tBuf[2].Get<float>();
            AscendC::LocalTensor<float> yLocalFloat = tBuf[3].Get<float>();
            AscendC::LocalTensor<float> zLocalFloat = tBuf[4].Get<float>();

            AscendC::Cast(xLocalFloat, xBroadcast, AscendC::RoundMode::CAST_NONE, compute_len);
            AscendC::Cast(yLocalFloat, yBroadcast, AscendC::RoundMode::CAST_NONE, compute_len);
            AscendC::Mul(xLocalFloat, xLocalFloat, xLocalFloat, compute_len);
            AscendC::Mul(yLocalFloat, yLocalFloat, yLocalFloat, compute_len);
            AscendC::Add(zLocalFloat, xLocalFloat, yLocalFloat, compute_len);
            AscendC::Sqrt(zLocalFloat, zLocalFloat, compute_len);
            AscendC::Cast(zLocal, zLocalFloat, AscendC::RoundMode::CAST_RINT, compute_len);
        } else {
            AscendC::Mul(xBroadcast, xBroadcast, xBroadcast, compute_len);
            AscendC::Mul(yBroadcast, yBroadcast, yBroadcast, compute_len);
            AscendC::Add(zLocal, xBroadcast, yBroadcast, compute_len);
            AscendC::Sqrt(zLocal, zLocal, compute_len);
        }

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void Process()
    {
        if (xShapeNew[0] < yShapeNew[0]) //
        {
            for (int i = 0; i < tileNum - 1; i++) {
                CopyInX(0, lastDimLength);
                CopyInY(i * tileLength, tileLength);
                ComputeB2(tileLength);
                CopyOut(i * tileLength * lastDimLength, tileLength * lastDimLength);
            }

            CopyInX(0, lastDimLength);
            CopyInY((tileNum - 1) * tileLength, lastTileLength);
            ComputeB2(lastTileLength);
            CopyOut((tileNum - 1) * tileLength * lastDimLength, lastTileLength * lastDimLength);
        } else {
            for (int i = 0; i < tileNum - 1; i++) {
                int offset = i * tileLength * lastDimLength;
                CopyInX(offset, tileLength * lastDimLength);
                CopyInY(i * tileLength, tileLength);
                Compute(tileLength);
                CopyOut(offset, tileLength * lastDimLength);
            }
            int offset = (tileNum - 1) * tileLength * lastDimLength;
            CopyInX(offset, lastTileLength * lastDimLength);
            CopyInY((tileNum - 1) * tileLength, lastTileLength);
            Compute(lastTileLength);
            CopyOut(offset, lastTileLength * lastDimLength);
        }
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    TBuf<TPosition::VECCALC> tBuf[2 + 3];

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;

    int tileLength;
    int tileNum;
    int lastTileLength;
    int dimnum;
    int xShapeNew[4];
    int yShapeNew[4];
    int zShapeNew[4];
    uint32_t lastDimLength;
};
template <typename T> class KernelHypot<T, 3> {
public:
    __aicore__ inline KernelHypot() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, HypotTilingData tiling)
    {
        this->dimnum = tiling.dimnum;
        bool switch_xy = false;
        if (tiling.xShape[dimnum - 1] < tiling.yShape[dimnum - 1]) {
            switch_xy = true;
        }

        GM_ADDR x_new = x;
        GM_ADDR y_new = y;
        if (switch_xy) {
            for (int i = 0; i < dimnum; i++) {
                xShapeNew[i] = tiling.yShape[i];
                yShapeNew[i] = tiling.xShape[i];
                zShapeNew[i] = tiling.zShape[i];
            }
            x_new = y;
            y_new = x;
        } else {
            for (int i = 0; i < dimnum; i++) {
                xShapeNew[i] = tiling.xShape[i];
                yShapeNew[i] = tiling.yShape[i];
                zShapeNew[i] = tiling.zShape[i];
            }
        }
        lastDimLength = zShapeNew[dimnum - 1];

        int xLen = 1;
        int yLen = 1;
        int zLen = 1;
        for (int i = 1; i < dimnum; i++) {
            xLen *= xShapeNew[i];
            yLen *= yShapeNew[i];
            zLen *= zShapeNew[i];
        }

        if (GetBlockIdx() < tiling.formerNum) {
            this->tileNum = tiling.formerTileNum;
            this->tileLength = tiling.formerTileLength;
            this->lastTileLength = tiling.formerLastTileLength;
            this->blockLength = tiling.formerLength;

            xGm.SetGlobalBuffer((__gm__ T *)x_new + (GetBlockIdx() * tiling.formerLength) % xShapeNew[0] * xLen,
                                xLen * tiling.formerLength);
            yGm.SetGlobalBuffer((__gm__ T *)y_new + (GetBlockIdx() * tiling.formerLength) % yShapeNew[0] * yLen,
                                yLen * tiling.formerLength);
            zGm.SetGlobalBuffer((__gm__ T *)z + GetBlockIdx() * tiling.formerLength * zLen, zLen * tiling.formerLength);
        } else {
            this->tileNum = tiling.tailTileNum;
            this->tileLength = tiling.tailTileLength;
            this->lastTileLength = tiling.tailLastTileLength;
            this->blockLength = tiling.tailLength;

            xGm.SetGlobalBuffer((__gm__ T *)x_new + (tiling.formerLength * tiling.formerNum +
                                                     (GetBlockIdx() - tiling.formerNum) * tiling.tailLength) %
                                                        xShapeNew[0] * xLen,
                                xLen * MIN(xShapeNew[0], tiling.tailLength));
            yGm.SetGlobalBuffer((__gm__ T *)y_new + (tiling.formerLength * tiling.formerNum +
                                                     (GetBlockIdx() - tiling.formerNum) * tiling.tailLength) %
                                                        yShapeNew[0] * yLen,
                                yLen * MIN(yShapeNew[0], tiling.tailLength));
            zGm.SetGlobalBuffer((__gm__ T *)z + (tiling.formerLength * tiling.formerNum +
                                                 (GetBlockIdx() - tiling.formerNum) * tiling.tailLength) *
                                                    zLen,
                                zLen * tiling.tailLength);
        }

        pipe.InitBuffer(inQueueX, BUFFER_NUM,
                        MIN(tileLength, xShapeNew[dimnum - 2]) * xShapeNew[dimnum - 1] * sizeof(T));
        pipe.InitBuffer(inQueueY, BUFFER_NUM,
                        MIN(tileLength, yShapeNew[dimnum - 2]) * yShapeNew[dimnum - 1] * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * zShapeNew[dimnum - 1] * sizeof(T));

        pipe.InitBuffer(tBuf[0], this->tileLength * lastDimLength * sizeof(T));
        pipe.InitBuffer(tBuf[1], this->tileLength * lastDimLength * sizeof(T));
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe.InitBuffer(tBuf[2], this->tileLength * lastDimLength * sizeof(float));
            pipe.InitBuffer(tBuf[3], this->tileLength * lastDimLength * sizeof(float));
            pipe.InitBuffer(tBuf[4], this->tileLength * lastDimLength * sizeof(float));
        }
    }

    __aicore__ inline void CopyInX(int64_t offset, int len)
    {
        LocalTensor<T> x = inQueueX.AllocTensor<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(x);
    }

    __aicore__ inline void CopyInY(int64_t offset, int len)
    {
        LocalTensor<T> y = inQueueY.AllocTensor<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(y, yGm[offset], copyParams, padParams);
        inQueueY.EnQue(y);
    }

    __aicore__ inline void CopyOut(int64_t offset, int len)
    {
        LocalTensor<T> z = outQueueZ.DeQue<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPad(zGm[offset], z, copyParams);
        outQueueZ.FreeTensor(z);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        AscendC::LocalTensor<T> broadcastTmpTensor = tBuf[0].Get<T>();

        uint32_t dstShape[] = {len, lastDimLength};
        uint32_t srcShape[] = {len, 1};
        MyBroadCast<T, 2, 1>(broadcastTmpTensor, yLocal, dstShape, srcShape);

        uint32_t compute_len = len * lastDimLength;

        if constexpr (std::is_same_v<T, bfloat16_t>) {
            AscendC::LocalTensor<float> xLocalFloat = tBuf[2].Get<float>();
            AscendC::LocalTensor<float> yLocalFloat = tBuf[3].Get<float>();
            AscendC::LocalTensor<float> zLocalFloat = tBuf[4].Get<float>();

            AscendC::Cast(xLocalFloat, xLocal, AscendC::RoundMode::CAST_NONE, compute_len);
            AscendC::Cast(yLocalFloat, broadcastTmpTensor, AscendC::RoundMode::CAST_NONE, compute_len);
            AscendC::Mul(xLocalFloat, xLocalFloat, xLocalFloat, compute_len);
            AscendC::Mul(yLocalFloat, yLocalFloat, yLocalFloat, compute_len);
            AscendC::Add(zLocalFloat, xLocalFloat, yLocalFloat, compute_len);
            AscendC::Sqrt(zLocalFloat, zLocalFloat, compute_len);
            AscendC::Cast(zLocal, zLocalFloat, AscendC::RoundMode::CAST_RINT, compute_len);
        } else {
            AscendC::Mul(xLocal, xLocal, xLocal, compute_len);
            AscendC::Mul(broadcastTmpTensor, broadcastTmpTensor, broadcastTmpTensor, compute_len);
            AscendC::Add(zLocal, xLocal, broadcastTmpTensor, compute_len);
            AscendC::Sqrt(zLocal, zLocal, compute_len);
        }
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void ComputeB2(uint32_t len)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        AscendC::LocalTensor<T> xBroadcast = tBuf[0].Get<T>();
        AscendC::LocalTensor<T> yBroadcast = tBuf[1].Get<T>();

        {
            uint32_t dstShape[] = {len, lastDimLength};
            uint32_t srcShape[] = {1, lastDimLength};
            MyBroadCast<T, 2, 0>(xBroadcast, xLocal, dstShape, srcShape);
        }
        {
            uint32_t dstShape[] = {len, lastDimLength};
            uint32_t srcShape[] = {len, 1};
            MyBroadCast<T, 2, 1>(yBroadcast, yLocal, dstShape, srcShape);
        }

        uint32_t compute_len = len * lastDimLength;

        if constexpr (std::is_same_v<T, bfloat16_t>) {
            AscendC::LocalTensor<float> xLocalFloat = tBuf[2].Get<float>();
            AscendC::LocalTensor<float> yLocalFloat = tBuf[3].Get<float>();
            AscendC::LocalTensor<float> zLocalFloat = tBuf[4].Get<float>();

            AscendC::Cast(xLocalFloat, xBroadcast, AscendC::RoundMode::CAST_NONE, compute_len);
            AscendC::Cast(yLocalFloat, yBroadcast, AscendC::RoundMode::CAST_NONE, compute_len);
            AscendC::Mul(xLocalFloat, xLocalFloat, xLocalFloat, compute_len);
            AscendC::Mul(yLocalFloat, yLocalFloat, yLocalFloat, compute_len);
            AscendC::Add(zLocalFloat, xLocalFloat, yLocalFloat, compute_len);
            AscendC::Sqrt(zLocalFloat, zLocalFloat, compute_len);
            AscendC::Cast(zLocal, zLocalFloat, AscendC::RoundMode::CAST_RINT, compute_len);
        } else {
            AscendC::Mul(xBroadcast, xBroadcast, xBroadcast, compute_len);
            AscendC::Mul(yBroadcast, yBroadcast, yBroadcast, compute_len);
            AscendC::Add(zLocal, xBroadcast, yBroadcast, compute_len);
            AscendC::Sqrt(zLocal, zLocal, compute_len);
        }

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void Process2D(uint32_t basex, uint32_t basey, uint32_t basez)
    {
        if (xShapeNew[dimnum - 2] < yShapeNew[dimnum - 2]) //
        {
            for (int i = 0; i < tileNum - 1; i++) {
                CopyInX(basex + 0, lastDimLength);
                CopyInY(basey + i * tileLength, tileLength);
                ComputeB2(tileLength);
                CopyOut(basez + i * tileLength * lastDimLength, tileLength * lastDimLength);
            }
            CopyInX(basex + 0, lastDimLength);
            CopyInY(basey + (tileNum - 1) * tileLength, lastTileLength);
            ComputeB2(lastTileLength);
            CopyOut(basez + (tileNum - 1) * tileLength * lastDimLength, lastTileLength * lastDimLength);
        } else {
            for (int i = 0; i < tileNum - 1; i++) {
                int offset = i * tileLength * lastDimLength;
                CopyInX(basex + offset, tileLength * lastDimLength);
                CopyInY(basey + i * tileLength, tileLength);
                Compute(tileLength);
                CopyOut(basez + offset, tileLength * lastDimLength);
            }
            int offset = (tileNum - 1) * tileLength * lastDimLength;
            CopyInX(basex + offset, lastTileLength * lastDimLength);
            CopyInY(basey + (tileNum - 1) * tileLength, lastTileLength);
            Compute(lastTileLength);
            CopyOut(basez + offset, lastTileLength * lastDimLength);
        }
    }

    __aicore__ inline void Process()
    {
        if (dimnum == 2) {
            Process2D(0, 0, 0);
        } else if (dimnum == 3) {
            for (int i = 0; i < blockLength; i++) {
                Process2D((i % xShapeNew[0]) * xShapeNew[1] * xShapeNew[2],
                          (i % yShapeNew[0]) * yShapeNew[1] * yShapeNew[2], i * zShapeNew[1] * zShapeNew[2]);
            }
        } else {
            for (int i = 0; i < blockLength; i++) {
                int x_offset_0 = i % xShapeNew[0];
                int y_offset_0 = i % yShapeNew[0];

                for (int j = 0; j < zShapeNew[1]; j++) {
                    int x_offset_1 = j % xShapeNew[1];
                    int y_offset_1 = j % yShapeNew[1];

                    Process2D((x_offset_0 * xShapeNew[1] + x_offset_1) * xShapeNew[2] * xShapeNew[3], //
                              (y_offset_0 * yShapeNew[1] + y_offset_1) * yShapeNew[2] * yShapeNew[3], //
                              (i * zShapeNew[1] + j) * zShapeNew[2] * zShapeNew[3]);
                }
            }
        }
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    TBuf<TPosition::VECCALC> tBuf[2 + 3];

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;

    int blockLength;
    int tileLength;
    int tileNum;
    int lastTileLength;
    int dimnum;
    int xShapeNew[4];
    int yShapeNew[4];
    int zShapeNew[4];
    uint32_t lastDimLength;
};

template <typename T> class KernelHypot<T, 4> {
public:
    __aicore__ inline KernelHypot() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, HypotTilingData tiling)
    {
        this->dimnum = tiling.dimnum;
        bool switch_xy = false;
        if (tiling.xShape[dimnum - 2] < tiling.yShape[dimnum - 2]) { //
            switch_xy = true;
        }

        GM_ADDR x_new = x;
        GM_ADDR y_new = y;
        if (switch_xy) {
            for (int i = 0; i < dimnum; i++) {
                xShapeNew[i] = tiling.yShape[i];
                yShapeNew[i] = tiling.xShape[i];
                zShapeNew[i] = tiling.zShape[i];
            }
            x_new = y;
            y_new = x;
        } else {
            for (int i = 0; i < dimnum; i++) {
                xShapeNew[i] = tiling.xShape[i];
                yShapeNew[i] = tiling.yShape[i];
                zShapeNew[i] = tiling.zShape[i];
            }
        }
        lastDimLength = zShapeNew[dimnum - 1];
        lastDimStride = ROUND_UP(lastDimLength * sizeof(T), 32) / sizeof(T);
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            lastDimStride = ROUND_UP(lastDimLength * sizeof(float), 32) / sizeof(float);
        }

        int xLen = 1;
        int yLen = 1;
        int zLen = 1;
        for (int i = 1; i < dimnum; i++) {
            xLen *= xShapeNew[i];
            yLen *= yShapeNew[i];
            zLen *= zShapeNew[i];
        }

        if (GetBlockIdx() < tiling.formerNum) {
            this->tileNum = tiling.formerTileNum;
            this->tileLength = tiling.formerTileLength;
            this->lastTileLength = tiling.formerLastTileLength;
            this->blockLength = tiling.formerLength;

            xGm.SetGlobalBuffer((__gm__ T *)x_new + (GetBlockIdx() * tiling.formerLength) % xShapeNew[0] * xLen,
                                xLen * MIN(xShapeNew[0], tiling.formerLength));
            yGm.SetGlobalBuffer((__gm__ T *)y_new + (GetBlockIdx() * tiling.formerLength) % yShapeNew[0] * yLen,
                                yLen * MIN(yShapeNew[0], tiling.formerLength));
            zGm.SetGlobalBuffer((__gm__ T *)z + GetBlockIdx() * tiling.formerLength * zLen, zLen * tiling.formerLength);
        } else {
            this->tileNum = tiling.tailTileNum;
            this->tileLength = tiling.tailTileLength;
            this->lastTileLength = tiling.tailLastTileLength;
            this->blockLength = tiling.tailLength;

            xGm.SetGlobalBuffer((__gm__ T *)x_new + (tiling.formerLength * tiling.formerNum +
                                                     (GetBlockIdx() - tiling.formerNum) * tiling.tailLength) %
                                                        xShapeNew[0] * xLen,
                                xLen * MIN(xShapeNew[0], tiling.tailLength));
            yGm.SetGlobalBuffer((__gm__ T *)y_new + (tiling.formerLength * tiling.formerNum +
                                                     (GetBlockIdx() - tiling.formerNum) * tiling.tailLength) %
                                                        yShapeNew[0] * yLen,
                                yLen * MIN(yShapeNew[0], tiling.tailLength));
            zGm.SetGlobalBuffer((__gm__ T *)z + (tiling.formerLength * tiling.formerNum +
                                                 (GetBlockIdx() - tiling.formerNum) * tiling.tailLength) *
                                                    zLen,
                                zLen * tiling.tailLength);
        }

        pipe.InitBuffer(inQueueX, 4, tileLength * xShapeNew[dimnum - 1] * sizeof(T));
        pipe.InitBuffer(inQueueY, 4, yShapeNew[dimnum - 1] * sizeof(T));
        pipe.InitBuffer(outQueueZ, 4, tileLength * zShapeNew[dimnum - 1] * sizeof(T));

        // printf("tile*lastdim:%d\n",tileLength*lastDimLength);
        pipe.InitBuffer(tBuf[0], this->tileLength * lastDimStride * sizeof(float));
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe.InitBuffer(tBuf[1], this->tileLength * lastDimStride * sizeof(float));
            pipe.InitBuffer(tBuf[2], this->tileLength * lastDimStride * sizeof(float));
            pipe.InitBuffer(tBuf[3], this->tileLength * lastDimStride * sizeof(float));
        }
    }

    __aicore__ inline void CopyInX(int64_t offset, uint16_t height)
    {
        if(lastDimLength == lastDimStride){
            LocalTensor<T> x = inQueueX.AllocTensor<T>();
            DataCopyExtParams copyParams = {1, (uint32_t)(height * lastDimLength * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(x, xGm[offset], copyParams, padParams);
            inQueueX.EnQue(x);
        }else{
            LocalTensor<T> x = inQueueX.AllocTensor<T>();
            DataCopyExtParams copyParams = {height, (uint32_t)(lastDimLength * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(x, xGm[offset], copyParams, padParams);
            inQueueX.EnQue(x);
        }
    }

    __aicore__ inline void CopyInY(int64_t offset)
    {
        LocalTensor<T> y = inQueueY.AllocTensor<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(lastDimLength * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(y, yGm[offset], copyParams, padParams);
        inQueueY.EnQue(y);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint16_t height)
    {
        if(lastDimLength == lastDimStride){
            LocalTensor<T> z = outQueueZ.DeQue<T>();
            DataCopyExtParams copyParams = {1, (uint32_t)(height*lastDimLength * sizeof(T)), 0, 0, 0};
            DataCopyPad(zGm[offset], z, copyParams);
            outQueueZ.FreeTensor(z);
        }else{
            LocalTensor<T> z = outQueueZ.DeQue<T>();
            DataCopyExtParams copyParams = {height, (uint32_t)(lastDimLength * sizeof(T)), 0, 0, 0};
            DataCopyPad(zGm[offset], z, copyParams);
            outQueueZ.FreeTensor(z);
        }
    }

    __aicore__ inline void Compute(LocalTensor<T> &yBroadCast2, uint32_t compute_len)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        AscendC::Mul(xLocal, xLocal, xLocal, compute_len);
        AscendC::Add(xLocal, xLocal, yBroadCast2, compute_len);
        AscendC::Sqrt(zLocal, xLocal, compute_len);

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeBF16(LocalTensor<float> &yBroadCast2, uint32_t compute_len)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        AscendC::LocalTensor<float> xLocalFloat = tBuf[1].Get<float>();
        AscendC::LocalTensor<float> zLocalFloat = tBuf[3].Get<float>();

        AscendC::Cast(xLocalFloat, xLocal, AscendC::RoundMode::CAST_NONE, compute_len);

        AscendC::Mul(xLocalFloat, xLocalFloat, xLocalFloat, compute_len);
        AscendC::Add(xLocalFloat, xLocalFloat, yBroadCast2, compute_len);
        AscendC::Sqrt(zLocalFloat, xLocalFloat, compute_len);
        AscendC::Cast(zLocal, zLocalFloat, AscendC::RoundMode::CAST_RINT, compute_len);

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
    }


    __aicore__ inline void Process2D(uint32_t basex, uint32_t basey, uint32_t basez)
    {
        CopyInY(basey);
        AscendC::LocalTensor<T> yLocal = inQueueY.DeQue<T>();

        if constexpr (std::is_same_v<T, bfloat16_t>) {
            AscendC::LocalTensor<float> broadcastTmpTensor = tBuf[0].Get<float>();
            AscendC::LocalTensor<float> yLocalFloat = tBuf[2].Get<float>();

            AscendC::Cast(yLocalFloat, yLocal, AscendC::RoundMode::CAST_NONE, lastDimLength);

            uint32_t dstShape[] = {(uint32_t)tileLength, lastDimStride};
            uint32_t srcShape[] = {1, lastDimStride};
            AscendC::Mul(yLocalFloat, yLocalFloat, yLocalFloat, lastDimLength);
            BroadCast<float, 2, 0>(broadcastTmpTensor, yLocalFloat, dstShape, srcShape);

            for (int i = 0; i < tileNum - 1; i++) {
                int offset = i * tileLength * lastDimLength;
                CopyInX(basex + offset, tileLength);
                ComputeBF16(broadcastTmpTensor, tileLength * lastDimStride);
                CopyOut(basez + offset, tileLength);
            }
            int offset = (tileNum - 1) * tileLength * lastDimLength;
            CopyInX(basex + offset, lastTileLength);
            ComputeBF16(broadcastTmpTensor, lastTileLength * lastDimStride);
            CopyOut(basez + offset, lastTileLength);

        } else {
            AscendC::LocalTensor<T> broadcastTmpTensor = tBuf[0].Get<T>();
            uint32_t dstShape[] = {(uint32_t)tileLength, lastDimStride};
            uint32_t srcShape[] = {1, lastDimStride};
            Mul(yLocal, yLocal, yLocal, lastDimLength);
            BroadCast<T, 2, 0>(broadcastTmpTensor, yLocal, dstShape, srcShape);

            for (int i = 0; i < tileNum - 1; i++) {
                int offset = i * tileLength * lastDimLength;
                CopyInX(basex + offset, tileLength);
                Compute(broadcastTmpTensor, tileLength * lastDimStride);
                CopyOut(basez + offset, tileLength);
            }
            int offset = (tileNum - 1) * tileLength * lastDimLength;
            CopyInX(basex + offset, lastTileLength);
            Compute(broadcastTmpTensor, lastTileLength * lastDimStride);
            CopyOut(basez + offset, lastTileLength);
        }

        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void Process()
    {
        if (dimnum == 2) {
            Process2D(0, 0, 0);
        } else if (dimnum == 3) {
            uint32_t xLast = xShapeNew[1] * xShapeNew[2];
            uint32_t yLast = yShapeNew[1] * yShapeNew[2];
            uint32_t zLast = zShapeNew[1] * zShapeNew[2];

            for (int i = 0; i < blockLength; i++) {
                Process2D((i % xShapeNew[0]) * xLast, (i % yShapeNew[0]) * yLast, i * zLast);
            }
        } else {
            uint32_t xLast = xShapeNew[2] * xShapeNew[3];
            uint32_t yLast = yShapeNew[2] * yShapeNew[3];
            uint32_t zLast = zShapeNew[2] * zShapeNew[3];

            for (int i = 0; i < blockLength; i++) {
                int x_offset_0 = (i % xShapeNew[0]) * xShapeNew[1];
                int y_offset_0 = (i % yShapeNew[0]) * yShapeNew[1];
                int z_offset_0 = i * zShapeNew[1];

                for (int j = 0; j < zShapeNew[1]; j++) {
                    int x_offset_1 = j % xShapeNew[1] + x_offset_0;
                    int y_offset_1 = j % yShapeNew[1] + y_offset_0;
                    int z_offset_1 = j + z_offset_0;

                    Process2D(x_offset_1 * xLast, //
                              y_offset_1 * yLast, //
                              z_offset_1 * zLast);
                }
            }
        }
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    TBuf<TPosition::VECCALC> tBuf[2 + 3];

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;

    int blockLength;
    int tileLength;
    int tileNum;
    int lastTileLength;
    int dimnum;
    int xShapeNew[4];
    int yShapeNew[4];
    int zShapeNew[4];
    uint32_t lastDimLength;
    uint32_t lastDimStride;
};

template <typename T> class KernelHypot<T, 5> {
public:
    __aicore__ inline KernelHypot() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, HypotTilingData tiling, TPipe *pipe)
    {
        this->dimnum = tiling.dimnum;
        bool switch_xy = false;
        if (tiling.xShape[dimnum - 1] < tiling.yShape[dimnum - 1]) { //
            switch_xy = true;
        }

        GM_ADDR x_new = x;
        GM_ADDR y_new = y;
        if (switch_xy) {
            for (int i = 0; i < dimnum; i++) {
                xShapeNew[i] = tiling.yShape[i];
                yShapeNew[i] = tiling.xShape[i];
                zShapeNew[i] = tiling.zShape[i];
            }
            x_new = y;
            y_new = x;
        } else {
            for (int i = 0; i < dimnum; i++) {
                xShapeNew[i] = tiling.xShape[i];
                yShapeNew[i] = tiling.yShape[i];
                zShapeNew[i] = tiling.zShape[i];
            }
        }
        lastDimLength = zShapeNew[dimnum - 1];

        int xLen = 1;
        int yLen = 1;
        int zLen = 1;
        for (int i = 0; i < dimnum; i++) {
            xLen *= xShapeNew[i];
            yLen *= yShapeNew[i];
            zLen *= zShapeNew[i];
        }

        if (GetBlockIdx() < tiling.formerNum) {
            this->tileNum = tiling.formerTileNum;
            this->tileLength = tiling.formerTileLength;
            this->lastTileLength = tiling.formerLastTileLength;
        } else {
            this->tileNum = tiling.tailTileNum;
            this->tileLength = tiling.tailTileLength;
            this->lastTileLength = tiling.tailLastTileLength;
        }

        this->blockLength = zShapeNew[0];
        {
            int blockNum = GetBlockNum();
            int total = 1;
            for (int i = 0; i < dimnum - 1; i++) {
                total *= zShapeNew[i];
            }
            total *= (zShapeNew[dimnum - 1] + tileLength - 1) / tileLength;

            int each = total / blockNum;
            int rem = total % blockNum;
            if (GetBlockIdx() < rem) {
                startIdx = GetBlockIdx() * (each + 1);
                endIdx = startIdx + each + 1;
            } else {
                startIdx = GetBlockIdx() * each + rem;
                endIdx = startIdx + each;
            }
        }
        curIdx = 0;

        xGm.SetGlobalBuffer((__gm__ T *)x_new, xLen);
        yGm.SetGlobalBuffer((__gm__ T *)y_new, yLen);
        zGm.SetGlobalBuffer((__gm__ T *)z, zLen);

        pipe->InitBuffer(inQueueX, BUFFER_NUM, 4096 * sizeof(T));
        pipe->InitBuffer(inQueueY, BUFFER_NUM, 4096 * sizeof(T));
        pipe->InitBuffer(outQueueZ, BUFFER_NUM, 4096 * sizeof(T));

        pipe->InitBuffer(tBuf[0], 4096 * sizeof(T));
        pipe->InitBuffer(tBuf[1], 4096 * sizeof(T));
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe->InitBuffer(tBuf[2], 4096 * sizeof(float));
            pipe->InitBuffer(tBuf[3], 4096 * sizeof(float));
            pipe->InitBuffer(tBuf[4], 4096 * sizeof(float));
        }
    }

    __aicore__ inline void CopyInX(int64_t offset, int len)
    {
        LocalTensor<T> x = inQueueX.AllocTensor<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(x);
    }

    __aicore__ inline void CopyInY(int64_t offset, int len)
    {
        LocalTensor<T> y = inQueueY.AllocTensor<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(y, yGm[offset], copyParams, padParams);
        inQueueY.EnQue(y);
    }

    __aicore__ inline void CopyOut(int64_t offset, int len)
    {
        LocalTensor<T> z = outQueueZ.DeQue<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPad(zGm[offset], z, copyParams);
        outQueueZ.FreeTensor(z);
    }

    __aicore__ inline void ComputeB(uint32_t len)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        AscendC::LocalTensor<T> broadcastTmpTensor = tBuf[0].Get<T>();

        uint32_t compute_len = len;

        if constexpr (std::is_same_v<T, bfloat16_t>) {
            auto yscale = yLocal(0);
            float yfloat = ToFloat(yscale);
            float yfloat2 = yfloat * yfloat;
            AscendC::LocalTensor<float> xLocalFloat = tBuf[2].Get<float>();
            AscendC::LocalTensor<float> zLocalFloat = tBuf[4].Get<float>();

            AscendC::Cast(xLocalFloat, xLocal, AscendC::RoundMode::CAST_NONE, compute_len);
            AscendC::Mul(xLocalFloat, xLocalFloat, xLocalFloat, compute_len);
            AscendC::Adds(zLocalFloat, xLocalFloat, yfloat2, compute_len);
            AscendC::Sqrt(zLocalFloat, zLocalFloat, compute_len);
            AscendC::Cast(zLocal, zLocalFloat, AscendC::RoundMode::CAST_RINT, compute_len);
        } else {
            auto yscale = yLocal(0);
            float yfloat = (float)(yscale);
            float yfloat2 = yfloat * yfloat;
            AscendC::Mul(xLocal, xLocal, xLocal, compute_len);
            AscendC::Adds(zLocal, xLocal, (T)yfloat2, compute_len);
            AscendC::Sqrt(zLocal, zLocal, compute_len);
        }
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        uint32_t compute_len = len;

        if constexpr (std::is_same_v<T, bfloat16_t>) {
            AscendC::LocalTensor<float> xLocalFloat = tBuf[2].Get<float>();
            AscendC::LocalTensor<float> yLocalFloat = tBuf[3].Get<float>();
            AscendC::LocalTensor<float> zLocalFloat = tBuf[4].Get<float>();

            AscendC::Cast(xLocalFloat, xLocal, AscendC::RoundMode::CAST_NONE, compute_len);
            AscendC::Cast(yLocalFloat, yLocal, AscendC::RoundMode::CAST_NONE, compute_len);
            AscendC::Mul(xLocalFloat, xLocalFloat, xLocalFloat, compute_len);
            AscendC::Mul(yLocalFloat, yLocalFloat, yLocalFloat, compute_len);
            AscendC::Add(zLocalFloat, xLocalFloat, yLocalFloat, compute_len);
            AscendC::Sqrt(zLocalFloat, zLocalFloat, compute_len);
            AscendC::Cast(zLocal, zLocalFloat, AscendC::RoundMode::CAST_RINT, compute_len);
        } else {
            AscendC::Mul(xLocal, xLocal, xLocal, compute_len);
            AscendC::Mul(yLocal, yLocal, yLocal, compute_len);
            AscendC::Add(zLocal, xLocal, yLocal, compute_len);
            AscendC::Sqrt(zLocal, zLocal, compute_len);
        }
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void Process1D(uint32_t basex, uint32_t basey, uint32_t basez)
    {
        if (xShapeNew[dimnum - 1] != yShapeNew[dimnum - 1]) {
            for (int i = 0; i < tileNum - 1; i++) {
                if (curIdx < startIdx || curIdx >= endIdx) {
                    curIdx++;
                    continue;
                }
                curIdx++;

                int offset = i * tileLength;
                CopyInX(basex + offset, tileLength);
                CopyInY(basey, 1);
                ComputeB(tileLength);
                CopyOut(basez + offset, tileLength);
            }
            if(lastTileLength){
                if (curIdx >= startIdx && curIdx < endIdx) {
                    int offset = (tileNum - 1) * tileLength;
                    CopyInX(basex + offset, lastTileLength);
                    CopyInY(basey, 1);
                    ComputeB(lastTileLength);
                    CopyOut(basez + offset, lastTileLength);
                }
                curIdx++;
            }
        } else {
            for (int i = 0; i < tileNum - 1; i++) {
                if (curIdx < startIdx || curIdx >= endIdx) {
                    curIdx++;
                    continue;
                }
                curIdx++;

                int offset = i * tileLength;
                CopyInX(basex + offset, tileLength);
                CopyInY(basey + offset, tileLength);
                Compute(tileLength);
                CopyOut(basez + offset, tileLength);
            }
            if(lastTileLength){
                if (curIdx >= startIdx && curIdx < endIdx) {
                    int offset = (tileNum - 1) * tileLength;
                    CopyInX(basex + offset, lastTileLength);
                    CopyInY(basey + offset, lastTileLength);
                    Compute(lastTileLength);
                    CopyOut(basez + offset, lastTileLength);
                }
                curIdx++;
            }
        }
    }

    __aicore__ inline void Process()
    {
        // printf("last tileLength:%d", lastTileLength);

        if (dimnum == 1) {
            Process1D(0, 0, 0);
        } else if (dimnum == 2) {
            for (int i = 0; i < blockLength; i++) {
                Process1D((i % xShapeNew[0]) * xShapeNew[1], (i % yShapeNew[0]) * yShapeNew[1], i * zShapeNew[1]);
            }
        } else if (dimnum == 3) {
            for (int i = 0; i < blockLength; i++) {
                int x_offset_0 = i % xShapeNew[0];
                int y_offset_0 = i % yShapeNew[0];

                for (int j = 0; j < zShapeNew[1]; j++) {
                    int x_offset_1 = j % xShapeNew[1];
                    int y_offset_1 = j % yShapeNew[1];

                    Process1D(x_offset_0 * xShapeNew[1] * xShapeNew[2] + x_offset_1 * xShapeNew[2],
                              y_offset_0 * yShapeNew[1] * yShapeNew[2] + y_offset_1 * yShapeNew[2],
                              i * zShapeNew[1] * zShapeNew[2] + j * zShapeNew[2]);
                }
            }
        } else {
            for (int i = 0; i < blockLength; i++) {
                int x_offset_0 = i % xShapeNew[0];
                int y_offset_0 = i % yShapeNew[0];

                for (int j = 0; j < zShapeNew[1]; j++) {
                    int x_offset_1 = j % xShapeNew[1];
                    int y_offset_1 = j % yShapeNew[1];

                    for (int k = 0; k < zShapeNew[2]; k++) {
                        int x_offset_2 = k % xShapeNew[2];
                        int y_offset_2 = k % yShapeNew[2];

                        Process1D(x_offset_0 * xShapeNew[1] * xShapeNew[2] * xShapeNew[3] +
                                      x_offset_1 * xShapeNew[2] * xShapeNew[3] + x_offset_2 * xShapeNew[3],
                                  y_offset_0 * yShapeNew[1] * yShapeNew[2] * yShapeNew[3] +
                                      y_offset_1 * yShapeNew[2] * yShapeNew[3] + y_offset_2 * yShapeNew[3],
                                  i * zShapeNew[1] * zShapeNew[2] * zShapeNew[3] + j * zShapeNew[2] * zShapeNew[3] +
                                      k * zShapeNew[3]);
                    }
                }
            }
        }
    }

private:

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    TBuf<TPosition::VECCALC> tBuf[2 + 3];

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;

    int blockLength;
    int tileLength;
    int tileNum;
    int lastTileLength;
    int dimnum;
    int xShapeNew[4];
    int yShapeNew[4];
    int zShapeNew[4];
    uint32_t lastDimLength;

    uint32_t curIdx;
    uint32_t startIdx;
    uint32_t endIdx;
};

extern "C" __global__ __aicore__ void hypot(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    // if (GetBlockIdx() == 0)
    //     printf("kernel print\n");
    // TODO: user kernel impl
    // if constexpr (std::is_same_v<DTYPE_X, half> || std::is_same_v<DTYPE_X, float>)
    if (TILING_KEY_IS(0)) // no broadcast
    {
        TPipe pipe;
        KernelHypot<DTYPE_X, 0> op;
        op.Init(x, y, z, tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelHypot<DTYPE_X, 1> op;
        op.Init(x, y, z, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelHypot<DTYPE_X, 2> op;
        op.Init(x, y, z, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelHypot<DTYPE_X, 3> op;
        op.Init(x, y, z, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        KernelHypot<DTYPE_X, 4> op;
        op.Init(x, y, z, tiling_data);
        op.Process();
        // if (GetBlockIdx() == 0)
        //     printf("tiling key 4\n");
    } else if (TILING_KEY_IS(5)) {
        TPipe pipe;
        KernelHypot<DTYPE_X, 5> op;
        op.Init(x, y, z, tiling_data, &pipe);
        op.Process();
    }
}