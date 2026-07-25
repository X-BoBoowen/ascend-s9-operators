#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM(2);

template <class T> class KernelCopysign {

  public:
    __aicore__ inline KernelCopysign() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out,
                                int32_t former_length, int32_t tail_length,
                                int32_t former_num, int32_t tile_length) {
        int32_t coreID = GetBlockIdx();

        coreLength = (coreID < former_num) ? former_length : tail_length;

        tileLength = tile_length;
        tileNum = (coreLength / tileLength);
        lastTileLength = coreLength - tileLength * tileNum;

        if (lastTileLength > 0) tileNum++;
        else lastTileLength = tileLength;

        loopLength = tileLength * GetBlockNum();

        int32_t formerTileNum = former_length / tile_length;
        int32_t formerLastTile = former_length - formerTileNum * tile_length;
        int32_t tailTileNum = tail_length / tile_length;
        int32_t tailLastTile = tail_length - tailTileNum * tile_length;

        if (formerTileNum == tailTileNum) {
            if (coreID < former_num) lastLoopLength = coreID * (formerLastTile - tileLength);
            else lastLoopLength = former_num * (formerLastTile - tileLength) + (coreID - former_num) * (tailLastTile - tileLength);
        } else {
            if (coreID < former_num) lastLoopLength = coreID * (formerLastTile - tileLength);
            else lastLoopLength = 0;
        }
        if (tileNum == 1) lastLoopLength = 0;

        int32_t totalLength = former_length * former_num + tail_length * (GetBlockNum() - former_num);
        int32_t beginLength = (tileNum > 1 ? coreID * tileLength : (
            (coreID < former_num)
                ? (former_length * coreID)
                : (tail_length * coreID +
                   (former_length - tail_length) * former_num)));

        // printf("coreLength: %d\ntileLength: %d\ntileNum: %d\nlastTileLength: %d\nloopLength: %d\nlastLoopLength: %d\nbeginLength: %d\n", coreLength, tileLength, tileNum, lastTileLength, loopLength, lastLoopLength, beginLength);

        pipe.InitBuffer(inputQue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(otherQue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(cmpBuf, tileLength * sizeof(uint8_t));

        inputGM.SetGlobalBuffer((__gm__ T *)input + beginLength,
                                totalLength);
        otherGM.SetGlobalBuffer((__gm__ T *)other + beginLength,
                                totalLength);
        outGM.SetGlobalBuffer((__gm__ T *)out + beginLength, totalLength);
    }
    __aicore__ inline void Process() {
        int32_t loopCount = tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

  private:
    __aicore__ inline void CopyIn(int32_t progress) {
        int32_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        int32_t currentPosition = progress * loopLength;
        if (progress == tileNum - 1) currentPosition += lastLoopLength;

        LocalTensor<T> inputLocal = inputQue.AllocTensor<T>();
        LocalTensor<T> otherLocal = otherQue.AllocTensor<T>();

        DataCopy(inputLocal, inputGM[currentPosition], currentLength);
        DataCopy(otherLocal, otherGM[currentPosition], currentLength);

        inputQue.EnQue(inputLocal);
        otherQue.EnQue(otherLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        int32_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        LocalTensor<T> inputLocal = inputQue.DeQue<T>();
        LocalTensor<T> otherLocal = otherQue.DeQue<T>();
        LocalTensor<T> outLocal = outQue.AllocTensor<T>();
        LocalTensor<uint8_t> cmp = cmpBuf.Get<uint8_t>();

        CompareScalar(cmp, otherLocal, T(0), CMPMODE::GE, currentLength);
        Abs(inputLocal, inputLocal, currentLength);
        Muls(otherLocal, inputLocal, T(-1), currentLength);
        Select(outLocal, cmp, inputLocal, otherLocal,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, currentLength);

        outQue.EnQue<T>(outLocal);
        inputQue.FreeTensor(inputLocal);
        otherQue.FreeTensor(otherLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        int32_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        int32_t currentPosition = progress * loopLength;
        if (progress == tileNum - 1) currentPosition += lastLoopLength;

        LocalTensor<T> outLocal = outQue.DeQue<T>();
        DataCopy(outGM[currentPosition], outLocal, currentLength);
        outQue.FreeTensor(outLocal);
    }

  private:
    TPipe pipe;
    GlobalTensor<T> inputGM;
    GlobalTensor<T> otherGM;
    GlobalTensor<T> outGM;
    TQue<QuePosition::VECIN, 1> inputQue, otherQue;
    TQue<QuePosition::VECOUT, 1> outQue;
    TBuf<TPosition::VECCALC> cmpBuf;

    int32_t coreLength;
    int32_t tileNum;
    int32_t tileLength;
    int32_t lastTileLength;
    int32_t loopLength;
    int32_t lastLoopLength;
};

template <> class KernelCopysign<bfloat16_t> {
    using T = bfloat16_t;
    using F = float16_t;

  public:
    __aicore__ inline KernelCopysign() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out,
                                int32_t former_length, int32_t tail_length,
                                int32_t former_num, int32_t tile_length) {
        int32_t coreID = GetBlockIdx();

        coreLength = (coreID < former_num) ? former_length : tail_length;

        tileLength = tile_length;
        tileNum = (coreLength / tileLength);
        lastTileLength = coreLength - tileLength * tileNum;

        if (lastTileLength > 0) tileNum++;
        else lastTileLength = tileLength;

        loopLength = tileLength * GetBlockNum();

        int32_t formerTileNum = former_length / tile_length;
        int32_t formerLastTile = former_length - formerTileNum * tile_length;
        int32_t tailTileNum = tail_length / tile_length;
        int32_t tailLastTile = tail_length - tailTileNum * tile_length;

        if (formerTileNum == tailTileNum) {
            if (coreID < former_num) lastLoopLength = coreID * (formerLastTile - tileLength);
            else lastLoopLength = former_num * (formerLastTile - tileLength) + (coreID - former_num) * (tailLastTile - tileLength);
        } else {
            if (coreID < former_num) lastLoopLength = coreID * (formerLastTile - tileLength);
            else lastLoopLength = 0;
        }
        if (tileNum == 1) lastLoopLength = 0;

        int32_t totalLength = former_length * former_num + tail_length * (GetBlockNum() - former_num);
        int32_t beginLength = (tileNum > 1 ? coreID * tileLength : (
            (coreID < former_num)
                ? (former_length * coreID)
                : (tail_length * coreID +
                   (former_length - tail_length) * former_num)));

        // printf("coreLength: %d\ntileLength: %d\ntileNum: %d\nlastTileLength: %d\nloopLength: %d\ntotalLength: %d\n", coreLength, tileLength, tileNum, lastTileLength, loopLength, totalLength);

        pipe.InitBuffer(inputQue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(otherQue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(cmpBuf, tileLength * sizeof(uint8_t));

        inputGM.SetGlobalBuffer((__gm__ T *)input + beginLength,
                                totalLength);
        otherGM.SetGlobalBuffer((__gm__ T *)other + beginLength,
                                totalLength);
        outGM.SetGlobalBuffer((__gm__ T *)out + beginLength, totalLength);
    }
    __aicore__ inline void Process() {
        int32_t loopCount = tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

  private:
    __aicore__ inline void CopyIn(int32_t progress) {
        int32_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        int32_t currentPosition = progress * loopLength;
        if (progress == tileNum - 1) currentPosition += lastLoopLength;

        LocalTensor<T> inputLocal = inputQue.AllocTensor<T>();
        LocalTensor<T> otherLocal = otherQue.AllocTensor<T>();

        DataCopy(inputLocal, inputGM[currentPosition], currentLength);
        DataCopy(otherLocal, otherGM[currentPosition], currentLength);

        inputQue.EnQue(inputLocal);
        otherQue.EnQue(otherLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        int32_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        LocalTensor<T> inputLocal = inputQue.DeQue<T>();
        LocalTensor<T> otherLocal = otherQue.DeQue<T>();
        LocalTensor<T> outLocal = outQue.AllocTensor<T>();

        LocalTensor<F> inputF = inputLocal.ReinterpretCast<F>();
        LocalTensor<F> otherF = otherLocal.ReinterpretCast<F>();
        LocalTensor<F> outF = outLocal.ReinterpretCast<F>();
        LocalTensor<uint8_t> cmp = cmpBuf.Get<uint8_t>();

        CompareScalar(cmp, otherF, F(0), CMPMODE::GE, currentLength);
        Abs(inputF, inputF, currentLength);
        Muls(otherF, inputF, F(-1), currentLength);
        Select(outF, cmp, inputF, otherF, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               currentLength);

        outQue.EnQue<T>(outLocal);
        inputQue.FreeTensor(inputLocal);
        otherQue.FreeTensor(otherLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        int32_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        int32_t currentPosition = progress * loopLength;
        if (progress == tileNum - 1) currentPosition += lastLoopLength;

        LocalTensor<T> outLocal = outQue.DeQue<T>();
        DataCopy(outGM[currentPosition], outLocal, currentLength);
        outQue.FreeTensor(outLocal);
    }

  private:
    TPipe pipe;
    GlobalTensor<T> inputGM;
    GlobalTensor<T> otherGM;
    GlobalTensor<T> outGM;
    TQue<QuePosition::VECIN, 1> inputQue, otherQue;
    TQue<QuePosition::VECOUT, 1> outQue;
    TBuf<TPosition::VECCALC> cmpBuf;

    int32_t coreLength;
    int32_t tileNum;
    int32_t tileLength;
    int32_t lastTileLength;
    int32_t loopLength;
    int32_t lastLoopLength;
};

template <class T> class KernelCopysignBoardCastVector {

  public:
    __aicore__ inline KernelCopysignBoardCastVector() {}
    __aicore__ inline void
    Init(GM_ADDR input, GM_ADDR other, GM_ADDR out, int32_t former_vector,
         int32_t tail_vector, int32_t former_num, int32_t tile_vector,
         int32_t vector_length, int32_t align_vector_length, uint8_t board_cast,
         int32_t out_dim1, int32_t out_dim2, int32_t out_dim3) {

        int32_t coreID = GetBlockIdx();
        coreVector = (coreID < former_num) ? former_vector : tail_vector;
        vectorLength = vector_length;
        alignVectorLength = align_vector_length;

        tileVector = tile_vector;
        tileNum = (coreVector / tileVector);
        lastTileVector = coreVector - tileVector * tileNum;

        if (lastTileVector > 0) tileNum++;
        else lastTileVector = tileVector;

        coreFormerVector = ((coreID < former_num)
                                ? (former_vector * coreID)
                                : (tail_vector * coreID +
                                   (former_vector - tail_vector) * former_num));

        // printf("#%d: %d %d %d\n", coreID, coreFormerVector, coreVector,
        //        tileVector);

        otherStride1 = (board_cast & 1) ^ 1;
        otherStride2 = ((board_cast >> 1) & 1) ^ 1;
        otherStride3 = ((board_cast >> 2) & 1) ^ 1;

        outDim1 = out_dim1;
        outDim2 = out_dim2;
        outDim3 = out_dim3;
        otherDim1 = otherStride1 ? outDim1 : 1;
        otherDim2 = otherStride2 ? outDim2 : 1;
        otherDim3 = otherStride3 ? outDim3 : 1;

        int32_t alignTileLength = tileVector * alignVectorLength;

        pipe.InitBuffer(inputQue, 1, alignTileLength * sizeof(T));
        pipe.InitBuffer(otherQue, 1, alignTileLength * sizeof(T));
        pipe.InitBuffer(outQue, 1, alignTileLength * sizeof(T));
        pipe.InitBuffer(cmpBuf, alignTileLength * sizeof(uint8_t));

        inputGM.SetGlobalBuffer((__gm__ T *)input,
                                outDim1 * outDim2 * outDim3 * vectorLength);
        otherGM.SetGlobalBuffer((__gm__ T *)other, otherDim1 * otherDim2 *
                                                       otherDim3 *
                                                       vectorLength);
        outGM.SetGlobalBuffer((__gm__ T *)out,
                              outDim1 * outDim2 * outDim3 * vectorLength);
    }
    __aicore__ inline void Process() {
        int32_t loopCount = tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

  private:
    __aicore__ inline void CopyIn(int32_t progress) {
        int32_t beginVector = progress * tileVector + coreFormerVector;
        int32_t currentVector =
            (progress == tileNum - 1) ? lastTileVector : tileVector;

        LocalTensor<T> inputLocal = inputQue.AllocTensor<T>();
        LocalTensor<T> otherLocal = otherQue.AllocTensor<T>();

        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(vectorLength * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        for (int32_t i = 0; i < currentVector; i++) {
            int32_t inputID = i + beginVector;

            int32_t dim1 = inputID / (outDim2 * outDim3);
            int32_t dim2 = inputID / outDim3 % outDim2;
            int32_t dim3 = inputID % outDim3;

            int32_t otherID = dim1 * otherStride1 * otherDim2 * otherDim3 +
                              dim2 * otherStride2 * otherDim3 +
                              dim3 * otherStride3;

            DataCopyPad(inputLocal[i * alignVectorLength],
                        inputGM[inputID * vectorLength], copyParams, padParams);
            DataCopyPad(otherLocal[i * alignVectorLength],
                        otherGM[otherID * vectorLength], copyParams, padParams);
        }

        inputQue.EnQue(inputLocal);
        otherQue.EnQue(otherLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        int32_t currentLength =
            alignVectorLength *
            ((progress == tileNum - 1) ? lastTileVector : tileVector);

        LocalTensor<T> inputLocal = inputQue.DeQue<T>();
        LocalTensor<T> otherLocal = otherQue.DeQue<T>();
        LocalTensor<T> outLocal = outQue.AllocTensor<T>();
        LocalTensor<uint8_t> cmp = cmpBuf.Get<uint8_t>();

        CompareScalar(cmp, otherLocal, T(0), CMPMODE::GE, currentLength);
        Abs(inputLocal, inputLocal, currentLength);
        Muls(otherLocal, inputLocal, T(-1), currentLength);
        Select(outLocal, cmp, inputLocal, otherLocal,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, currentLength);

        outQue.EnQue<T>(outLocal);
        inputQue.FreeTensor(inputLocal);
        otherQue.FreeTensor(otherLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        int32_t beginVector = progress * tileVector + coreFormerVector;
        int32_t currentVector =
            (progress == tileNum - 1) ? lastTileVector : tileVector;

        LocalTensor<T> outLocal = outQue.DeQue<T>();

        int32_t contentSize = vectorLength * sizeof(T);
        int32_t alignSize = alignVectorLength * sizeof(T);
        int32_t srcStride = (alignSize - (contentSize + 31) / 32 * 32) / 32;

        DataCopyExtParams copyParams{
            static_cast<uint16_t>(currentVector),
            static_cast<uint32_t>(vectorLength * sizeof(T)),
            static_cast<uint32_t>(srcStride), 0, 0};
        DataCopyPad(outGM[beginVector * vectorLength], outLocal, copyParams);

        outQue.FreeTensor(outLocal);
    }

  private:
    TPipe pipe;
    GlobalTensor<T> inputGM;
    GlobalTensor<T> otherGM;
    GlobalTensor<T> outGM;
    TQue<QuePosition::VECIN, 1> inputQue, otherQue;
    TQue<QuePosition::VECOUT, 1> outQue;
    TBuf<TPosition::VECCALC> cmpBuf;

    int32_t coreVector;
    int32_t coreFormerVector;
    int32_t tileNum;
    int32_t tileVector;
    int32_t lastTileVector;

    int32_t vectorLength;
    int32_t alignVectorLength;

    int8_t otherStride1, otherStride2, otherStride3;
    int32_t outDim1, outDim2, outDim3;
    int32_t otherDim1, otherDim2, otherDim3;
};

template <> class KernelCopysignBoardCastVector<bfloat16_t> {
    using T = bfloat16_t;
    using F = float16_t;

  public:
    __aicore__ inline KernelCopysignBoardCastVector() {}
    __aicore__ inline void
    Init(GM_ADDR input, GM_ADDR other, GM_ADDR out, int32_t former_vector,
         int32_t tail_vector, int32_t former_num, int32_t tile_vector,
         int32_t vector_length, int32_t align_vector_length, uint8_t board_cast,
         int32_t out_dim1, int32_t out_dim2, int32_t out_dim3) {

        int32_t coreID = GetBlockIdx();
        coreVector = (coreID < former_num) ? former_vector : tail_vector;
        vectorLength = vector_length;
        alignVectorLength = align_vector_length;

        tileVector = tile_vector;
        tileNum = (coreVector / tileVector);
        lastTileVector = coreVector - tileVector * tileNum;

        if (lastTileVector > 0) tileNum++;
        else lastTileVector = tileVector;

        coreFormerVector = ((coreID < former_num)
                                ? (former_vector * coreID)
                                : (tail_vector * coreID +
                                   (former_vector - tail_vector) * former_num));

        otherStride1 = (board_cast & 1) ^ 1;
        otherStride2 = ((board_cast >> 1) & 1) ^ 1;
        otherStride3 = ((board_cast >> 2) & 1) ^ 1;

        outDim1 = out_dim1;
        outDim2 = out_dim2;
        outDim3 = out_dim3;
        otherDim1 = otherStride1 ? outDim1 : 1;
        otherDim2 = otherStride2 ? outDim2 : 1;
        otherDim3 = otherStride3 ? outDim3 : 1;

        int32_t alignTileLength = tileVector * alignVectorLength;

        pipe.InitBuffer(inputQue, 1, alignTileLength * sizeof(T));
        pipe.InitBuffer(otherQue, 1, alignTileLength * sizeof(T));
        pipe.InitBuffer(outQue, 1, alignTileLength * sizeof(T));
        pipe.InitBuffer(cmpBuf, alignTileLength * sizeof(uint8_t));

        inputGM.SetGlobalBuffer((__gm__ T *)input,
                                outDim1 * outDim2 * outDim3 * vectorLength);
        otherGM.SetGlobalBuffer((__gm__ T *)other, otherDim1 * otherDim2 *
                                                       otherDim3 *
                                                       vectorLength);
        outGM.SetGlobalBuffer((__gm__ T *)out,
                              outDim1 * outDim2 * outDim3 * vectorLength);
    }
    __aicore__ inline void Process() {
        int32_t loopCount = tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

  private:
    __aicore__ inline void CopyIn(int32_t progress) {
        int32_t beginVector = progress * tileVector + coreFormerVector;
        int32_t currentVector =
            (progress == tileNum - 1) ? lastTileVector : tileVector;

        LocalTensor<T> inputLocal = inputQue.AllocTensor<T>();
        LocalTensor<T> otherLocal = otherQue.AllocTensor<T>();

        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(vectorLength * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        for (int32_t i = 0; i < currentVector; i++) {
            int32_t inputID = i + beginVector;

            int32_t dim1 = inputID / (outDim2 * outDim3);
            int32_t dim2 = inputID / outDim3 % outDim2;
            int32_t dim3 = inputID % outDim3;

            int32_t otherID = dim1 * otherStride1 * otherDim2 * otherDim3 +
                              dim2 * otherStride2 * otherDim3 +
                              dim3 * otherStride3;

            DataCopyPad(inputLocal[i * alignVectorLength],
                        inputGM[inputID * vectorLength], copyParams, padParams);
            DataCopyPad(otherLocal[i * alignVectorLength],
                        otherGM[otherID * vectorLength], copyParams, padParams);
        }

        inputQue.EnQue(inputLocal);
        otherQue.EnQue(otherLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        int32_t currentLength =
            alignVectorLength *
            ((progress == tileNum - 1) ? lastTileVector : tileVector);

        LocalTensor<T> inputLocal = inputQue.DeQue<T>();
        LocalTensor<T> otherLocal = otherQue.DeQue<T>();
        LocalTensor<T> outLocal = outQue.AllocTensor<T>();

        LocalTensor<F> inputF = inputLocal.ReinterpretCast<F>();
        LocalTensor<F> otherF = otherLocal.ReinterpretCast<F>();
        LocalTensor<F> outF = outLocal.ReinterpretCast<F>();
        LocalTensor<uint8_t> cmp = cmpBuf.Get<uint8_t>();

        CompareScalar(cmp, otherF, F(0), CMPMODE::GE, currentLength);
        Abs(inputF, inputF, currentLength);
        Muls(otherF, inputF, F(-1), currentLength);
        Select(outF, cmp, inputF, otherF, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               currentLength);

        outQue.EnQue<T>(outLocal);
        inputQue.FreeTensor(inputLocal);
        otherQue.FreeTensor(otherLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        int32_t beginVector = progress * tileVector + coreFormerVector;
        int32_t currentVector =
            (progress == tileNum - 1) ? lastTileVector : tileVector;

        LocalTensor<T> outLocal = outQue.DeQue<T>();

        int32_t contentSize = vectorLength * sizeof(T);
        int32_t alignSize = alignVectorLength * sizeof(T);
        int32_t srcStride = (alignSize - (contentSize + 31) / 32 * 32) / 32;

        DataCopyExtParams copyParams{
            static_cast<uint16_t>(currentVector),
            static_cast<uint32_t>(vectorLength * sizeof(T)),
            static_cast<uint32_t>(srcStride), 0, 0};
        DataCopyPad(outGM[beginVector * vectorLength], outLocal, copyParams);

        outQue.FreeTensor(outLocal);
    }

  private:
    TPipe pipe;
    GlobalTensor<T> inputGM;
    GlobalTensor<T> otherGM;
    GlobalTensor<T> outGM;
    TQue<QuePosition::VECIN, 1> inputQue, otherQue;
    TQue<QuePosition::VECOUT, 1> outQue;
    TBuf<TPosition::VECCALC> cmpBuf;

    int32_t coreVector;
    int32_t coreFormerVector;
    int32_t tileNum;
    int32_t tileVector;
    int32_t lastTileVector;

    int32_t vectorLength;
    int32_t alignVectorLength;

    int8_t otherStride1, otherStride2, otherStride3;
    int32_t outDim1, outDim2, outDim3;
    int32_t otherDim1, otherDim2, otherDim3;
};

extern "C" __global__ __aicore__ void copysign(GM_ADDR input, GM_ADDR other,
                                               GM_ADDR out, GM_ADDR workspace,
                                               GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (tiling_data.board_cast) {
        KernelCopysignBoardCastVector<DTYPE_INPUT> op;
        op.Init(input, other, out, tiling_data.former_length,
                tiling_data.tail_length, tiling_data.former_num,
                tiling_data.tile_length, tiling_data.vector_length,
                tiling_data.align_vector_length, tiling_data.board_cast,
                tiling_data.out_dim1, tiling_data.out_dim2,
                tiling_data.out_dim3);
        op.Process();
    } else {
        KernelCopysign<DTYPE_INPUT> op;
        op.Init(input, other, out, tiling_data.former_length,
                tiling_data.tail_length, tiling_data.former_num,
                tiling_data.tile_length);
        op.Process();
    }
}