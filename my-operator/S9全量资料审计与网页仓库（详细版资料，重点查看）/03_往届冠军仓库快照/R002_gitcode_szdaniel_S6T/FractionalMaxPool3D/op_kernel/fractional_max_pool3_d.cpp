#include "kernel_operator.h"

using namespace AscendC;

// always broad cast y
#define BUFFER_NUM 2
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define ROUND_UP(a, b) (((a) + (b) - 1) / (b) * (b))
#define ROUND_DOWN(a, b) ((a) / (b) * (b))
#define ELE2BLOCK(x) ((x * sizeof(T) + 31) / 32)

template <typename dataType, uint32_t tileKey> class KernelFractionalMaxPool3D;

template <typename T> class KernelFractionalMaxPool3D<T, 0> {
public:
    __aicore__ inline KernelFractionalMaxPool3D() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR random_sample, GM_ADDR out, GM_ADDR indices,
                                FractionalMaxPool3DTilingData tiling, TPipe &pipe)
    {
        this->tStep = tiling.tStep;
        this->hStep = tiling.hStep;
        this->wStep = tiling.wStep;

        this->inputT = tiling.inputT;
        this->outputT = tiling.outputT;
        this->poolSizeT = tiling.poolSizeT;

        this->inputH = tiling.inputH;
        this->outputH = tiling.outputH;
        this->poolSizeH = tiling.poolSizeH;

        this->inputW = tiling.inputW;
        this->outputW = tiling.outputW;
        this->poolSizeW = tiling.poolSizeW;

        int maxLength = tStep * hStep * ROUND_UP(wStep, 32 / sizeof(T));

        this->inBlockSize = inputT * inputH * inputW;
        this->outBlockSize = outputT * outputH * outputW;

        if (GetBlockIdx() < tiling.formerNum) {
            this->block_length = tiling.formerLength;

            inputGm.SetGlobalBuffer((__gm__ T *)input + tiling.formerLength * GetBlockIdx() * inBlockSize,
                                    tiling.formerLength * inBlockSize);
            randomSampleGm.SetGlobalBuffer((__gm__ T *)random_sample + tiling.formerLength * GetBlockIdx() * 3,
                                           tiling.formerLength * 3);
            outGm.SetGlobalBuffer((__gm__ T *)out + tiling.formerLength * GetBlockIdx() * outBlockSize,
                                  tiling.formerLength * outBlockSize);
            indicesGm.SetGlobalBuffer((__gm__ int32_t *)indices + tiling.formerLength * GetBlockIdx() * outBlockSize,
                                      tiling.formerLength * outBlockSize);
        } else {
            this->block_length = tiling.tailLength;

            inputGm.SetGlobalBuffer((__gm__ T *)input + (tiling.formerLength * tiling.formerNum +
                                                         tiling.tailLength * (GetBlockIdx() - tiling.formerNum)) *
                                                            inBlockSize,
                                    tiling.tailLength * inBlockSize);
            randomSampleGm.SetGlobalBuffer(
                (__gm__ T *)random_sample +
                    (tiling.formerLength * tiling.formerNum + tiling.tailLength * (GetBlockIdx() - tiling.formerNum)) *
                        3,
                tiling.tailLength * 3);
            outGm.SetGlobalBuffer((__gm__ T *)out + (tiling.formerLength * tiling.formerNum +
                                                     tiling.tailLength * (GetBlockIdx() - tiling.formerNum)) *
                                                        outBlockSize,
                                  tiling.tailLength * outBlockSize);
            indicesGm.SetGlobalBuffer(
                (__gm__ int32_t *)indices +
                    (tiling.formerLength * tiling.formerNum + tiling.tailLength * (GetBlockIdx() - tiling.formerNum)) *
                        outBlockSize,
                tiling.tailLength * outBlockSize);
        }

        pipe.InitBuffer(inputQueue, BUFFER_NUM, inputW * sizeof(T));
        pipe.InitBuffer(randomSampleQueue, BUFFER_NUM, 3 * sizeof(T));
        pipe.InitBuffer(outQueue, BUFFER_NUM, outputW * sizeof(T));
        pipe.InitBuffer(indicesQueue, BUFFER_NUM, outputW * sizeof(int32_t));

        pipe.InitBuffer(tBuf[0], outputT * sizeof(int32_t));
        pipe.InitBuffer(tBuf[1], outputH * sizeof(int32_t));
        pipe.InitBuffer(tBuf[2], outputW * sizeof(int32_t));
        pipe.InitBuffer(tBuf[3], outputW * sizeof(int32_t));

        pipe.InitBuffer(gatherBuf, outputW * sizeof(float));
        pipe.InitBuffer(maskBuf, outputW * sizeof(uint8_t) + 64);
        pipe.InitBuffer(newIdxBuf, outputW * sizeof(int32_t));
        pipe.InitBuffer(gatherF32Buf, outputW * sizeof(float));
        pipe.InitBuffer(outF32Buf, outputW * sizeof(float));

        const int max_seq_len = MAX(MAX(outputT, outputH), outputW);
        pipe.InitBuffer(sBuf[0], max_seq_len * sizeof(float));
        pipe.InitBuffer(sBuf[1], max_seq_len * sizeof(float));
        pipe.InitBuffer(sBuf[2], max_seq_len * sizeof(float));
    }

    __aicore__ inline void CopyInX(int64_t offset, int len)
    {
        LocalTensor<T> x = inputQueue.AllocTensor<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x, inputGm[offset], copyParams, padParams);
        inputQueue.EnQue(x);
    }

    __aicore__ inline void CopyInR(int64_t offset, int len)
    {
        LocalTensor<T> r = randomSampleQueue.AllocTensor<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(r, randomSampleGm[offset], copyParams, padParams);
        randomSampleQueue.EnQue(r);
    }

    __aicore__ inline void CopyOutZ(int64_t offset, int len)
    {
        LocalTensor<T> z = outQueue.DeQue<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPad(outGm[offset], z, copyParams);
        outQueue.FreeTensor(z);
    }

    __aicore__ inline void CopyOutI(int64_t offset, int len)
    {
        LocalTensor<int32_t> z = indicesQueue.DeQue<int32_t>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(indicesGm[offset], z, copyParams);
        indicesQueue.FreeTensor(z);
    }


    __aicore__ inline void generate_intervals(LocalTensor<int32_t> &sequence, T randomSample, int inputSize,
                                              int outputSize, int poolSize)
    {
        if (outputSize > 1) {
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                LocalTensor<float> iF32 = sBuf[0].Get<float>();
                LocalTensor<T> iBf16 = sBuf[1].Get<T>();
                LocalTensor<int32_t> sL2 = sBuf[2].Get<int32_t>();

                iF32(0) = (float)(inputSize - poolSize);
                Cast(iBf16, iF32, AscendC::RoundMode::CAST_RINT, 1);
                Cast(iF32, iBf16, AscendC::RoundMode::CAST_NONE, 1);
                float a = iF32(0);
                iF32(0) = (float)(outputSize - 1);
                Cast(iBf16, iF32, AscendC::RoundMode::CAST_RINT, 1);
                Cast(iF32, iBf16, AscendC::RoundMode::CAST_NONE, 1);
                float b = iF32(0);
                float alpha = a / b;

                iF32(0) = alpha;
                Cast(iBf16, iF32, AscendC::RoundMode::CAST_RINT, 1);
                Cast(iF32, iBf16, AscendC::RoundMode::CAST_NONE, 1);
                alpha = iF32(0);

                CreateVecIndex(iF32, 0.f, outputSize - 1);
                Cast(iBf16, iF32, AscendC::RoundMode::CAST_RINT, outputSize - 1);
                Cast(iF32, iBf16, AscendC::RoundMode::CAST_NONE, outputSize - 1);

                Adds(iF32, iF32, ToFloat(randomSample), outputSize - 1);
                // 2
                Cast(iBf16, iF32, AscendC::RoundMode::CAST_RINT, outputSize - 1);
                Cast(iF32, iBf16, AscendC::RoundMode::CAST_NONE, outputSize - 1);

                Muls(iF32, iF32, (alpha), outputSize - 1);
                // 3
                Cast(iBf16, iF32, AscendC::RoundMode::CAST_RINT, outputSize - 1);
                Cast(iF32, iBf16, AscendC::RoundMode::CAST_NONE, outputSize - 1);

                Cast(sL2, iF32, AscendC::RoundMode::CAST_TRUNC, outputSize - 1);
                int nrma = -static_cast<int>(ToFloat(ToBfloat16(ToFloat(randomSample) * (alpha))));

                Adds(sequence, sL2, nrma, outputSize - 1);
            } else if constexpr (std::is_same_v<T, half>) {
                LocalTensor<float> iF32 = sBuf[0].Get<float>();
                LocalTensor<T> iF16 = sBuf[1].Get<T>();
                LocalTensor<int32_t> sL2 = sBuf[2].Get<int32_t>();

                float alpha = (float)(inputSize - poolSize) / (float)(outputSize - 1);
                alpha = (float)((half)alpha);

                CreateVecIndex(iF32, 0.f, outputSize - 1);
                Cast(iF16, iF32, AscendC::RoundMode::CAST_RINT, outputSize - 1);
                Cast(iF32, iF16, AscendC::RoundMode::CAST_NONE, outputSize - 1);

                Adds(iF32, iF32, (float)(randomSample), outputSize - 1);
                Cast(iF16, iF32, AscendC::RoundMode::CAST_RINT, outputSize - 1);
                Cast(iF32, iF16, AscendC::RoundMode::CAST_NONE, outputSize - 1);

                Muls(iF32, iF32, (alpha), outputSize - 1);
                Cast(iF16, iF32, AscendC::RoundMode::CAST_RINT, outputSize - 1);
                Cast(iF32, iF16, AscendC::RoundMode::CAST_NONE, outputSize - 1);

                Cast(sL2, iF32, AscendC::RoundMode::CAST_TRUNC, outputSize - 1);
                int nrma = -static_cast<int>((float)((half)((float)(randomSample) * (alpha))));

                Adds(sequence, sL2, nrma, outputSize - 1);
            } else {
                LocalTensor<T> sL0 = sBuf[0].Get<T>();
                LocalTensor<T> sL1 = sBuf[1].Get<T>();
                LocalTensor<int32_t> sL2 = sBuf[2].Get<int32_t>();

                float alpha = (float)(inputSize - poolSize) / (float)(outputSize - 1);

                CreateVecIndex(sL0, (T)0, outputSize - 1);
                Adds(sL0, sL0, (T)randomSample, outputSize - 1);
                Muls(sL0, sL0, alpha, outputSize - 1);

                Cast(sL2, sL0, AscendC::RoundMode::CAST_TRUNC, outputSize - 1);
                int nrma = -static_cast<int>(randomSample * alpha);
                Adds(sequence, sL2, nrma, outputSize - 1);
            }
        }
        if (outputSize > 0) {
            sequence(outputSize - 1) = inputSize - poolSize;
        }
    }

    __aicore__ inline void ComputeSeq()
    {
        AscendC::LocalTensor<T> randomLocal = randomSampleQueue.DeQue<T>();

        /* Generate interval sequence */
        LocalTensor<int32_t> sequenceT = tBuf[0].Get<int32_t>();
        LocalTensor<int32_t> sequenceH = tBuf[1].Get<int32_t>();
        LocalTensor<int32_t> sequenceW = tBuf[2].Get<int32_t>();
        LocalTensor<int32_t> sequenceWmulst = tBuf[3].Get<int32_t>();

        T r[3];
        r[0] = randomLocal(0);
        r[1] = randomLocal(1);
        r[2] = randomLocal(2);

        generate_intervals(sequenceT, r[0], inputT, outputT, poolSizeT);
        generate_intervals(sequenceH, r[1], inputH, outputH, poolSizeH);
        generate_intervals(sequenceW, r[2], inputW, outputW, poolSizeW);
        Add(sequenceWmulst, sequenceW, sequenceW, outputW);
        if constexpr (std::is_same_v<T, float>) {
            Add(sequenceWmulst, sequenceWmulst, sequenceWmulst, outputW);
        }

        // DumpTensor(sequenceT, 0, 32);
        // DumpTensor(sequenceH, 1, 32);
        // DumpTensor(sequenceW, 2, 32);

        randomSampleQueue.FreeTensor(randomLocal);
    }

    __aicore__ inline void Compute(LocalTensor<T> &outLocal, LocalTensor<int32_t> &idxLocal, uint32_t offset,
                                   uint32_t len)
    {
        // comput one row;
        LocalTensor<T> inLocal = inputQueue.DeQue<T>();


        LocalTensor<int32_t> sequenceW = tBuf[2].Get<int32_t>();
        LocalTensor<uint32_t> sequenceWmulst = tBuf[3].Get<uint32_t>();

        LocalTensor<T> gatherLocal = gatherBuf.Get<T>();
        LocalTensor<int32_t> newIdxLocal = newIdxBuf.Get<int32_t>();
        LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
        LocalTensor<float> gatherF32 = gatherF32Buf.Get<float>();
        LocalTensor<float> outF32 = outF32Buf.Get<float>();

        for (uint32_t j = 0; j < poolSizeW; j++) {
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                Gather(gatherLocal, inLocal, sequenceWmulst, j * sizeof(T), len);
                Cast(gatherF32, gatherLocal, AscendC::RoundMode::CAST_NONE, len);

                Compare(mask, outF32, gatherF32, AscendC::CMPMODE::LT, ROUND_UP(len, 256 / sizeof(float)));
                Max(outF32, outF32, gatherF32, len);
                Adds(newIdxLocal, sequenceW, (int32_t)(offset + j), len);

                Select(idxLocal.ReinterpretCast<float>(), mask, newIdxLocal.ReinterpretCast<float>(),
                       idxLocal.ReinterpretCast<float>(), AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, len);

            } else {
                Gather(gatherLocal, inLocal, sequenceWmulst, j * sizeof(T), (uint32_t)len);

                Compare(mask, outLocal, gatherLocal, AscendC::CMPMODE::LT, ROUND_UP(len, 256 / sizeof(T)));

                Max(outLocal, outLocal, gatherLocal, (uint32_t)len);
                // //idx
                Adds(newIdxLocal, sequenceW, (int32_t)(offset + j), len);
                Select(idxLocal.ReinterpretCast<float>(), mask, newIdxLocal.ReinterpretCast<float>(),
                       idxLocal.ReinterpretCast<float>(), AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, len);
            }
        }

        inputQueue.FreeTensor(inLocal);
    }

    __aicore__ inline void Process3D(int block_id)
    {
        LocalTensor<int32_t> sequenceT = tBuf[0].Get<int32_t>();
        LocalTensor<int32_t> sequenceH = tBuf[1].Get<int32_t>();
        LocalTensor<int32_t> sequenceW = tBuf[2].Get<int32_t>();
        LocalTensor<uint32_t> sequenceWmulst = tBuf[3].Get<uint32_t>();

        LocalTensor<float> outF32 = outF32Buf.Get<float>();

        uint32_t ioffset_block = block_id * inputT * inputH * inputW;
        uint32_t ooffset_block = block_id * outputT * outputH * outputW;

        for (int t = 0; t < outputT; t++) {
            int32_t inputTStart = sequenceT(t);
            int32_t inputTEnd = inputTStart + poolSizeT;

            int32_t curTout = 1;
            int32_t curTin = inputTEnd - inputTStart;
            uint32_t ooffset_t = t * outputH * outputW;

            // printf("its:%d, ite:%d, to:%d, ti:%d\n", inputTStart, inputTEnd, curTout, curTin);
            // hStep = 1
            for (int h = 0; h < outputH; h++) {
                int32_t inputHStart = sequenceH(h);
                int32_t inputHEnd = inputHStart + poolSizeH;

                int32_t curHout = 1;
                int32_t curHin = inputHEnd - inputHStart;
                uint32_t ooffset_h = h * outputW;

                // printf("ihs:%d, ihe:%d, ho:%d, hi:%d\n", inputHStart, inputHEnd, curHout, curHin);

                // compute 1-row result
                LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
                LocalTensor<int32_t> idxLocal = indicesQueue.AllocTensor<int32_t>();
                // Duplicate(idxLocal, 0, outputW);

                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    float max_scale;
                    *(uint32_t *)&max_scale = 0xff800000;
                    Duplicate(outF32, max_scale, outputW);
                } else if constexpr (std::is_same_v<T, half>) {
                    uint16_t max_scale;
                    *(uint16_t *)&max_scale = 0xfc00;
                    Duplicate<uint16_t>(outLocal.template ReinterpretCast<uint16_t>(), max_scale, outputW);
                } else if constexpr (std::is_same_v<T, float>) {
                    T max_scale;
                    *(uint32_t *)&max_scale = 0xff800000;
                    Duplicate(outLocal, max_scale, outputW);
                }

                for (int t2 = inputTStart; t2 < inputTEnd; t2++) {
                    uint32_t ioffset_t = t2 * inputH * inputW;
                    // printf("t2:%d\n", t2);

                    for (int h2 = inputHStart; h2 < inputHEnd; h2++) {
                        uint32_t ioffset_h = h2 * inputW;
                        // printf("h2:%d\n", h2);

                        CopyInX(ioffset_block + ioffset_t + ioffset_h, inputW);

                        Compute(outLocal, idxLocal, ioffset_t + ioffset_h, outputW);
                    }
                }
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    Cast(outLocal, outF32, AscendC::RoundMode::CAST_RINT, (uint32_t)outputW);
                }
                outQueue.EnQue(outLocal);
                indicesQueue.EnQue(idxLocal);

                uint32_t ooffset = ooffset_block + ooffset_t + ooffset_h;
                // printf("ooffset:%d, b:%d,t:%d,h:%d\n", ooffset, ooffset_block, ooffset_t, ooffset_h);
                CopyOutZ(ooffset, outputW);
                CopyOutI(ooffset, outputW);
            }
        }
    }
    __aicore__ inline void Process()
    {
        for (int i = 0; i < block_length; i++) {
            CopyInR(i * 3, 3);
            ComputeSeq();
            Process3D(i);
        }
    }

private:
    // TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> randomSampleQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> indicesQueue;

    TBuf<TPosition::VECCALC> tBuf[4];
    TBuf<TPosition::VECCALC> gatherBuf;
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> newIdxBuf;
    TBuf<TPosition::VECCALC> gatherF32Buf;
    TBuf<TPosition::VECCALC> outF32Buf;
    TBuf<TPosition::VECCALC> sBuf[3];

    GlobalTensor<T> inputGm;
    GlobalTensor<T> randomSampleGm;
    GlobalTensor<T> outGm;
    GlobalTensor<int32_t> indicesGm;

    // int tileLength;
    // int tileNum;
    // int lastTileLength;

    int inBlockSize;
    int outBlockSize;
    int block_length;

    int inputT, outputT, poolSizeT;
    int inputH, outputH, poolSizeH;
    int inputW, outputW, poolSizeW;

    int tStep, hStep, wStep;
};

template <typename T> class KernelFractionalMaxPool3D<T, 1> {
public:
    __aicore__ inline KernelFractionalMaxPool3D() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR random_sample, GM_ADDR out, GM_ADDR indices,
                                FractionalMaxPool3DTilingData tiling, TPipe &pipe)
    {
        this->tStep = tiling.tStep;
        this->hStep = tiling.hStep;
        this->wStep = tiling.wStep;

        this->inputT = tiling.inputT;
        this->outputT = tiling.outputT;
        this->poolSizeT = tiling.poolSizeT;

        this->inputH = tiling.inputH;
        this->outputH = tiling.outputH;
        this->poolSizeH = tiling.poolSizeH;

        this->inputW = tiling.inputW;
        this->outputW = tiling.outputW;
        this->poolSizeW = tiling.poolSizeW;

        int maxLength = tStep * hStep * ROUND_UP(wStep, 32 / sizeof(T));

        this->inBlockSize = inputT * inputH * inputW;
        this->outBlockSize = outputT * outputH * outputW;

        if (GetBlockIdx() < tiling.formerNum) {
            this->block_length = tiling.formerLength;

            inputGm.SetGlobalBuffer((__gm__ T *)input + tiling.formerLength * GetBlockIdx() * inBlockSize,
                                    tiling.formerLength * inBlockSize);
            randomSampleGm.SetGlobalBuffer((__gm__ T *)random_sample + tiling.formerLength * GetBlockIdx() * 3,
                                           tiling.formerLength * 3);
            outGm.SetGlobalBuffer((__gm__ T *)out + tiling.formerLength * GetBlockIdx() * outBlockSize,
                                  tiling.formerLength * outBlockSize);
        } else {
            this->block_length = tiling.tailLength;

            inputGm.SetGlobalBuffer((__gm__ T *)input + (tiling.formerLength * tiling.formerNum +
                                                         tiling.tailLength * (GetBlockIdx() - tiling.formerNum)) *
                                                            inBlockSize,
                                    tiling.tailLength * inBlockSize);
            randomSampleGm.SetGlobalBuffer(
                (__gm__ T *)random_sample +
                    (tiling.formerLength * tiling.formerNum + tiling.tailLength * (GetBlockIdx() - tiling.formerNum)) *
                        3,
                tiling.tailLength * 3);
            outGm.SetGlobalBuffer((__gm__ T *)out + (tiling.formerLength * tiling.formerNum +
                                                     tiling.tailLength * (GetBlockIdx() - tiling.formerNum)) *
                                                        outBlockSize,
                                  tiling.tailLength * outBlockSize);
        }

        pipe.InitBuffer(inputQueue, BUFFER_NUM, inputW * sizeof(T));
        pipe.InitBuffer(randomSampleQueue, BUFFER_NUM, 3 * sizeof(T));
        pipe.InitBuffer(outQueue, BUFFER_NUM, outputW * sizeof(T));

        pipe.InitBuffer(tBuf[0], outputT * sizeof(int32_t));
        pipe.InitBuffer(tBuf[1], outputH * sizeof(int32_t));
        pipe.InitBuffer(tBuf[2], outputW * sizeof(int32_t));
        pipe.InitBuffer(tBuf[3], outputW * sizeof(int32_t));

        pipe.InitBuffer(gatherBuf, outputW * sizeof(float));
        pipe.InitBuffer(maskBuf, outputW * sizeof(uint8_t) + 64);
        pipe.InitBuffer(newIdxBuf, outputW * sizeof(int32_t));
        pipe.InitBuffer(gatherF32Buf, outputW * sizeof(float));
        pipe.InitBuffer(outF32Buf, outputW * sizeof(float));

        const int max_seq_len = MAX(MAX(outputT, outputH), outputW);
        pipe.InitBuffer(sBuf[0], max_seq_len * sizeof(float));
        pipe.InitBuffer(sBuf[1], max_seq_len * sizeof(float));
        pipe.InitBuffer(sBuf[2], max_seq_len * sizeof(float));
    }

    __aicore__ inline void CopyInX(int64_t offset, int len)
    {
        LocalTensor<T> x = inputQueue.AllocTensor<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x, inputGm[offset], copyParams, padParams);
        inputQueue.EnQue(x);
    }

    __aicore__ inline void CopyInR(int64_t offset, int len)
    {
        LocalTensor<T> r = randomSampleQueue.AllocTensor<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(r, randomSampleGm[offset], copyParams, padParams);
        randomSampleQueue.EnQue(r);
    }

    __aicore__ inline void CopyOutZ(int64_t offset, int len)
    {
        LocalTensor<T> z = outQueue.DeQue<T>();
        DataCopyExtParams copyParams = {1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPad(outGm[offset], z, copyParams);
        outQueue.FreeTensor(z);
    }

    __aicore__ inline void generate_intervals(LocalTensor<int32_t> &sequence, T randomSample, int inputSize,
                                              int outputSize, int poolSize)
    {
        if (outputSize > 1) {
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                LocalTensor<float> iF32 = sBuf[0].Get<float>();
                LocalTensor<T> iBf16 = sBuf[1].Get<T>();
                LocalTensor<int32_t> sL2 = sBuf[2].Get<int32_t>();

                iF32(0) = (float)(inputSize - poolSize);
                Cast(iBf16, iF32, AscendC::RoundMode::CAST_RINT, 1);
                Cast(iF32, iBf16, AscendC::RoundMode::CAST_NONE, 1);
                float a = iF32(0);
                iF32(0) = (float)(outputSize - 1);
                Cast(iBf16, iF32, AscendC::RoundMode::CAST_RINT, 1);
                Cast(iF32, iBf16, AscendC::RoundMode::CAST_NONE, 1);
                float b = iF32(0);
                float alpha = a / b;

                iF32(0) = alpha;
                Cast(iBf16, iF32, AscendC::RoundMode::CAST_RINT, 1);
                Cast(iF32, iBf16, AscendC::RoundMode::CAST_NONE, 1);
                alpha = iF32(0);

                CreateVecIndex(iF32, 0.f, outputSize - 1);
                Cast(iBf16, iF32, AscendC::RoundMode::CAST_RINT, outputSize - 1);
                Cast(iF32, iBf16, AscendC::RoundMode::CAST_NONE, outputSize - 1);

                Adds(iF32, iF32, ToFloat(randomSample), outputSize - 1);
                // 2
                Cast(iBf16, iF32, AscendC::RoundMode::CAST_RINT, outputSize - 1);
                Cast(iF32, iBf16, AscendC::RoundMode::CAST_NONE, outputSize - 1);

                Muls(iF32, iF32, (alpha), outputSize - 1);
                // 3
                Cast(iBf16, iF32, AscendC::RoundMode::CAST_RINT, outputSize - 1);
                Cast(iF32, iBf16, AscendC::RoundMode::CAST_NONE, outputSize - 1);

                Cast(sL2, iF32, AscendC::RoundMode::CAST_TRUNC, outputSize - 1);
                int nrma = -static_cast<int>(ToFloat(ToBfloat16(ToFloat(randomSample) * (alpha))));

                Adds(sequence, sL2, nrma, outputSize - 1);
            } else if constexpr (std::is_same_v<T, half>) {
                LocalTensor<float> iF32 = sBuf[0].Get<float>();
                LocalTensor<T> iF16 = sBuf[1].Get<T>();
                LocalTensor<int32_t> sL2 = sBuf[2].Get<int32_t>();

                float alpha = (float)(inputSize - poolSize) / (float)(outputSize - 1);
                alpha = (float)((half)alpha);

                CreateVecIndex(iF32, 0.f, outputSize - 1);
                Cast(iF16, iF32, AscendC::RoundMode::CAST_RINT, outputSize - 1);
                Cast(iF32, iF16, AscendC::RoundMode::CAST_NONE, outputSize - 1);

                Adds(iF32, iF32, (float)(randomSample), outputSize - 1);
                Cast(iF16, iF32, AscendC::RoundMode::CAST_RINT, outputSize - 1);
                Cast(iF32, iF16, AscendC::RoundMode::CAST_NONE, outputSize - 1);

                Muls(iF32, iF32, (alpha), outputSize - 1);
                Cast(iF16, iF32, AscendC::RoundMode::CAST_RINT, outputSize - 1);
                Cast(iF32, iF16, AscendC::RoundMode::CAST_NONE, outputSize - 1);

                Cast(sL2, iF32, AscendC::RoundMode::CAST_TRUNC, outputSize - 1);
                int nrma = -static_cast<int>((float)((half)((float)(randomSample) * (alpha))));

                Adds(sequence, sL2, nrma, outputSize - 1);
            } else {
                LocalTensor<T> sL0 = sBuf[0].Get<T>();
                LocalTensor<T> sL1 = sBuf[1].Get<T>();
                LocalTensor<int32_t> sL2 = sBuf[2].Get<int32_t>();

                float alpha = (float)(inputSize - poolSize) / (float)(outputSize - 1);

                CreateVecIndex(sL0, (T)0, outputSize - 1);
                Adds(sL0, sL0, (T)randomSample, outputSize - 1);
                Muls(sL0, sL0, alpha, outputSize - 1);

                Cast(sL2, sL0, AscendC::RoundMode::CAST_TRUNC, outputSize - 1);
                int nrma = -static_cast<int>(randomSample * alpha);
                Adds(sequence, sL2, nrma, outputSize - 1);
            }
        }
        if (outputSize > 0) {
            sequence(outputSize - 1) = inputSize - poolSize;
        }
    }

    __aicore__ inline void ComputeSeq()
    {
        AscendC::LocalTensor<T> randomLocal = randomSampleQueue.DeQue<T>();

        /* Generate interval sequence */
        LocalTensor<int32_t> sequenceT = tBuf[0].Get<int32_t>();
        LocalTensor<int32_t> sequenceH = tBuf[1].Get<int32_t>();
        LocalTensor<int32_t> sequenceW = tBuf[2].Get<int32_t>();
        LocalTensor<int32_t> sequenceWmulst = tBuf[3].Get<int32_t>();

        T r[3];
        r[0] = randomLocal(0);
        r[1] = randomLocal(1);
        r[2] = randomLocal(2);

        generate_intervals(sequenceT, r[0], inputT, outputT, poolSizeT);
        generate_intervals(sequenceH, r[1], inputH, outputH, poolSizeH);
        generate_intervals(sequenceW, r[2], inputW, outputW, poolSizeW);
        Add(sequenceWmulst, sequenceW, sequenceW, outputW);
        if constexpr (std::is_same_v<T, float>) {
            Add(sequenceWmulst, sequenceWmulst, sequenceWmulst, outputW);
        }

        randomSampleQueue.FreeTensor(randomLocal);
    }

    __aicore__ inline void Compute(LocalTensor<T> &outLocal, uint32_t offset, uint32_t len)
    {
        // comput one row;
        LocalTensor<T> inLocal = inputQueue.DeQue<T>();

        LocalTensor<int32_t> sequenceW = tBuf[2].Get<int32_t>();
        LocalTensor<uint32_t> sequenceWmulst = tBuf[3].Get<uint32_t>();

        LocalTensor<T> gatherLocal = gatherBuf.Get<T>();
        LocalTensor<int32_t> newIdxLocal = newIdxBuf.Get<int32_t>();
        LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
        LocalTensor<float> gatherF32 = gatherF32Buf.Get<float>();
        LocalTensor<float> outF32 = outF32Buf.Get<float>();

        for (uint32_t j = 0; j < poolSizeW; j++) {
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                Gather(gatherLocal, inLocal, sequenceWmulst, j * sizeof(T), len);
                Cast(gatherF32, gatherLocal, AscendC::RoundMode::CAST_NONE, len);
                Max(outF32, outF32, gatherF32, len);
            } else {
                Gather(gatherLocal, inLocal, sequenceWmulst, j * sizeof(T), (uint32_t)len);
                Max(outLocal, outLocal, gatherLocal, (uint32_t)len);
            }
        }

        inputQueue.FreeTensor(inLocal);
    }

    __aicore__ inline void Process3D(int block_id)
    {
        LocalTensor<int32_t> sequenceT = tBuf[0].Get<int32_t>();
        LocalTensor<int32_t> sequenceH = tBuf[1].Get<int32_t>();
        LocalTensor<int32_t> sequenceW = tBuf[2].Get<int32_t>();
        LocalTensor<uint32_t> sequenceWmulst = tBuf[3].Get<uint32_t>();

        LocalTensor<float> outF32 = outF32Buf.Get<float>();

        uint32_t ioffset_block = block_id * inputT * inputH * inputW;
        uint32_t ooffset_block = block_id * outputT * outputH * outputW;

        for (int t = 0; t < outputT; t++) {
            int32_t inputTStart = sequenceT(t);
            int32_t inputTEnd = inputTStart + poolSizeT;

            int32_t curTout = 1;
            int32_t curTin = inputTEnd - inputTStart;
            uint32_t ooffset_t = t * outputH * outputW;

            // printf("its:%d, ite:%d, to:%d, ti:%d\n", inputTStart, inputTEnd, curTout, curTin);
            // hStep = 1
            for (int h = 0; h < outputH; h++) {
                int32_t inputHStart = sequenceH(h);
                int32_t inputHEnd = inputHStart + poolSizeH;

                int32_t curHout = 1;
                int32_t curHin = inputHEnd - inputHStart;
                uint32_t ooffset_h = h * outputW;

                // printf("ihs:%d, ihe:%d, ho:%d, hi:%d\n", inputHStart, inputHEnd, curHout, curHin);

                // compute 1-row result
                LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
                // Duplicate(idxLocal, 0, outputW);

                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    float max_scale;
                    *(uint32_t *)&max_scale = 0xff800000;
                    Duplicate(outF32, max_scale, outputW);
                } else if constexpr (std::is_same_v<T, half>) {
                    uint16_t max_scale;
                    *(uint16_t *)&max_scale = 0xfc00;
                    Duplicate<uint16_t>(outLocal.template ReinterpretCast<uint16_t>(), max_scale, outputW);
                } else if constexpr (std::is_same_v<T, float>) {
                    T max_scale;
                    *(uint32_t *)&max_scale = 0xff800000;
                    Duplicate(outLocal, max_scale, outputW);
                }

                for (int t2 = inputTStart; t2 < inputTEnd; t2++) {
                    uint32_t ioffset_t = t2 * inputH * inputW;
                    // printf("t2:%d\n", t2);

                    for (int h2 = inputHStart; h2 < inputHEnd; h2++) {
                        uint32_t ioffset_h = h2 * inputW;
                        // printf("h2:%d\n", h2);

                        CopyInX(ioffset_block + ioffset_t + ioffset_h, inputW);

                        Compute(outLocal, ioffset_t + ioffset_h, outputW);
                    }
                }
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    Cast(outLocal, outF32, AscendC::RoundMode::CAST_RINT, (uint32_t)outputW);
                }
                outQueue.EnQue(outLocal);

                uint32_t ooffset = ooffset_block + ooffset_t + ooffset_h;
                // printf("ooffset:%d, b:%d,t:%d,h:%d\n", ooffset, ooffset_block, ooffset_t, ooffset_h);
                CopyOutZ(ooffset, outputW);
            }
        }
    }
    __aicore__ inline void Process()
    {
        for (int i = 0; i < block_length; i++) {
            CopyInR(i * 3, 3);
            ComputeSeq();
            Process3D(i);
        }
    }

private:
    // TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> randomSampleQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> indicesQueue;

    TBuf<TPosition::VECCALC> tBuf[4];
    TBuf<TPosition::VECCALC> gatherBuf;
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> newIdxBuf;
    TBuf<TPosition::VECCALC> gatherF32Buf;
    TBuf<TPosition::VECCALC> outF32Buf;
    TBuf<TPosition::VECCALC> sBuf[3];

    GlobalTensor<T> inputGm;
    GlobalTensor<T> randomSampleGm;
    GlobalTensor<T> outGm;
    GlobalTensor<int32_t> indicesGm;

    // int tileLength;
    // int tileNum;
    // int lastTileLength;

    int inBlockSize;
    int outBlockSize;
    int block_length;

    int inputT, outputT, poolSizeT;
    int inputH, outputH, poolSizeH;
    int inputW, outputW, poolSizeW;

    int tStep, hStep, wStep;
};

extern "C" __global__ __aicore__ void fractional_max_pool3_d(GM_ADDR input, GM_ADDR random_sample, GM_ADDR out,
                                                             GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl

    AscendC::TPipe pipeIn;
    pipeIn.Init();

    if (TILING_KEY_IS(0)) {
        KernelFractionalMaxPool3D<DTYPE_INPUT, 0> op;
        op.Init(input, random_sample, out, indices, tiling_data, pipeIn);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelFractionalMaxPool3D<DTYPE_INPUT, 1> op;
        op.Init(input, random_sample, out, indices, tiling_data, pipeIn);
        op.Process();
    }
    pipeIn.Destroy();
}
