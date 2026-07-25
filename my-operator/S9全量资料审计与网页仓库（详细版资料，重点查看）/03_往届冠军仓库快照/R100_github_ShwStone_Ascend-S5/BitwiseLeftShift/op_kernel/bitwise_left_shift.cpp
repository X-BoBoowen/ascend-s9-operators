#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM(2);

template <class T> struct unsigned_type {
    using type = T;
};

template <> struct unsigned_type<int8_t> {
    using type = uint8_t;
};
template <> struct unsigned_type<int64_t> {
    using type = uint64_t;
};

template <class T> class KernelBitwiseLeftShift {
    using U = typename unsigned_type<T>::type;

  public:
    __aicore__ inline KernelBitwiseLeftShift() {}
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

        pipe.InitBuffer(inputQue, BUFFER_NUM, tileLength * sizeof(U));
        pipe.InitBuffer(otherQue, BUFFER_NUM, tileLength * sizeof(U));
        pipe.InitBuffer(outQue, BUFFER_NUM, tileLength * sizeof(U));

        inputGM.SetGlobalBuffer((__gm__ U *)input + beginLength,
                                totalLength);
        otherGM.SetGlobalBuffer((__gm__ U *)other + beginLength,
                                totalLength);
        outGM.SetGlobalBuffer((__gm__ U *)out + beginLength, totalLength);
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

        LocalTensor<U> inputLocal = inputQue.AllocTensor<U>();
        LocalTensor<U> otherLocal = otherQue.AllocTensor<U>();

        DataCopy(inputLocal, inputGM[currentPosition], currentLength);
        DataCopy(otherLocal, otherGM[currentPosition], currentLength);

        inputQue.EnQue(inputLocal);
        otherQue.EnQue(otherLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        int32_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        LocalTensor<U> inputLocal = inputQue.DeQue<U>();
        LocalTensor<U> otherLocal = otherQue.DeQue<U>();
        LocalTensor<U> outLocal = outQue.AllocTensor<U>();

        for (int32_t i = 0; i < currentLength; i++) {
            U val = inputLocal(i), shift = otherLocal(i);
            outLocal(i) = (val << shift);
        }

        outQue.EnQue<U>(outLocal);
        inputQue.FreeTensor(inputLocal);
        otherQue.FreeTensor(otherLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        int32_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        int32_t currentPosition = progress * loopLength;
        if (progress == tileNum - 1) currentPosition += lastLoopLength;

        LocalTensor<U> outLocal = outQue.DeQue<U>();
        DataCopy(outGM[currentPosition], outLocal, currentLength);
        outQue.FreeTensor(outLocal);
    }

  private:
    TPipe pipe;
    GlobalTensor<U> inputGM;
    GlobalTensor<U> otherGM;
    GlobalTensor<U> outGM;
    TQue<QuePosition::VECIN, 1> inputQue, otherQue;
    TQue<QuePosition::VECOUT, 1> outQue;

    int32_t coreLength;
    int32_t tileNum;
    int32_t tileLength;
    int32_t lastTileLength;
    int32_t loopLength;
    int32_t lastLoopLength;
};

template <> class KernelBitwiseLeftShift<int16_t> {
    using T = int16_t;
    using U = uint16_t;
    using F = half;

  public:
    __aicore__ inline KernelBitwiseLeftShift() {}
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

        pipe.InitBuffer(inputQue, BUFFER_NUM, tileLength * sizeof(U));
        pipe.InitBuffer(otherQue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQue, BUFFER_NUM, tileLength * sizeof(U));
        pipe.InitBuffer(cmpBuf, tileLength * sizeof(uint8_t));

        inputGM.SetGlobalBuffer((__gm__ U *)input + beginLength,
                                totalLength);
        otherGM.SetGlobalBuffer((__gm__ T *)other + beginLength,
                                totalLength);
        outGM.SetGlobalBuffer((__gm__ U *)out + beginLength, totalLength);
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

        LocalTensor<U> inputLocal = inputQue.AllocTensor<U>();
        LocalTensor<T> otherLocal = otherQue.AllocTensor<T>();

        DataCopy(inputLocal, inputGM[currentPosition], currentLength);
        DataCopy(otherLocal, otherGM[currentPosition], currentLength);

        inputQue.EnQue(inputLocal);
        otherQue.EnQue(otherLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        int32_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        LocalTensor<U> inputLocal = inputQue.DeQue<U>();
        LocalTensor<T> otherLocal = otherQue.DeQue<T>();
        LocalTensor<U> outLocal = outQue.AllocTensor<U>();
        LocalTensor<uint8_t> cmp = cmpBuf.Get<uint8_t>();

        LocalTensor<F> inputF = inputLocal.ReinterpretCast<F>();
        LocalTensor<F> outF = outLocal.ReinterpretCast<F>();
        LocalTensor<F> otherF = otherLocal.ReinterpretCast<F>();
        LocalTensor<T> outT = outLocal.ReinterpretCast<T>();

#define SEL_TT SELMODE::VSEL_TENSOR_TENSOR_MODE
#define SCAST static_cast
#define LEFT_SHIFT(b)                                                          \
    {                                                                          \
        CompareScalar(cmp, otherF, SCAST<F>((b)), CMPMODE::GE, currentLength); \
        ShiftLeft(outLocal, inputLocal, SCAST<U>((b)), currentLength);         \
        Select(inputF, cmp, outF, inputF, SEL_TT, currentLength);              \
        Adds(outF, otherF, SCAST<F>(-(b)), currentLength);                     \
        Select(otherF, cmp, outF, otherF, SEL_TT, currentLength);              \
    }

        Cast(otherF, otherLocal, RoundMode::CAST_NONE, currentLength);

        LEFT_SHIFT(8);
        LEFT_SHIFT(4);
        LEFT_SHIFT(2);

        CompareScalar(cmp, otherF, SCAST<F>(1), CMPMODE::GE, currentLength);
        ShiftLeft(outLocal, inputLocal, SCAST<U>(1), currentLength);
        Select(outF, cmp, outF, inputF, SEL_TT, currentLength);

#undef SEL_TT
#undef SCAST
#undef LEFT_SHIFT

        outQue.EnQue<U>(outLocal);
        inputQue.FreeTensor(inputLocal);
        otherQue.FreeTensor(otherLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        int32_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        int32_t currentPosition = progress * loopLength;
        if (progress == tileNum - 1) currentPosition += lastLoopLength;

        LocalTensor<U> outLocal = outQue.DeQue<U>();
        DataCopy(outGM[currentPosition], outLocal, currentLength);
        outQue.FreeTensor(outLocal);
    }

  private:
    TPipe pipe;
    GlobalTensor<U> inputGM;
    GlobalTensor<T> otherGM;
    GlobalTensor<U> outGM;
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

template <> class KernelBitwiseLeftShift<int32_t> {
    using T = int32_t;
    using U = uint32_t;
    using F = float;

  public:
    __aicore__ inline KernelBitwiseLeftShift() {}
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

        pipe.InitBuffer(inputQue, BUFFER_NUM, tileLength * sizeof(U));
        pipe.InitBuffer(otherQue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQue, BUFFER_NUM, tileLength * sizeof(U));
        pipe.InitBuffer(cmpBuf, tileLength * sizeof(uint8_t));

        inputGM.SetGlobalBuffer((__gm__ U *)input + beginLength,
                                totalLength);
        otherGM.SetGlobalBuffer((__gm__ T *)other + beginLength,
                                totalLength);
        outGM.SetGlobalBuffer((__gm__ U *)out + beginLength, totalLength);
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

        LocalTensor<U> inputLocal = inputQue.AllocTensor<U>();
        LocalTensor<T> otherLocal = otherQue.AllocTensor<T>();

        DataCopy(inputLocal, inputGM[currentPosition], currentLength);
        DataCopy(otherLocal, otherGM[currentPosition], currentLength);

        inputQue.EnQue(inputLocal);
        otherQue.EnQue(otherLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        int32_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        LocalTensor<U> inputLocal = inputQue.DeQue<U>();
        LocalTensor<T> otherLocal = otherQue.DeQue<T>();
        LocalTensor<U> outLocal = outQue.AllocTensor<U>();
        LocalTensor<uint8_t> cmp = cmpBuf.Get<uint8_t>();

        LocalTensor<F> inputF = inputLocal.ReinterpretCast<F>();
        LocalTensor<F> outF = outLocal.ReinterpretCast<F>();
        LocalTensor<F> otherF = otherLocal.ReinterpretCast<F>();
        LocalTensor<T> outT = outLocal.ReinterpretCast<T>();

#define SEL_TT SELMODE::VSEL_TENSOR_TENSOR_MODE
#define SCAST static_cast
#define LEFT_SHIFT(b)                                                          \
    {                                                                          \
        CompareScalar(cmp, otherF, SCAST<F>((b)), CMPMODE::GE, currentLength); \
        ShiftLeft(outLocal, inputLocal, SCAST<U>((b)), currentLength);         \
        Select(inputF, cmp, outF, inputF, SEL_TT, currentLength);              \
        Adds(outF, otherF, SCAST<F>(-(b)), currentLength);                     \
        Select(otherF, cmp, outF, otherF, SEL_TT, currentLength);              \
    }

        Cast(otherF, otherLocal, RoundMode::CAST_NONE, currentLength);

        LEFT_SHIFT(16);
        LEFT_SHIFT(8);
        LEFT_SHIFT(4);
        LEFT_SHIFT(2);

        CompareScalar(cmp, otherF, SCAST<F>(1), CMPMODE::GE, currentLength);
        ShiftLeft(outLocal, inputLocal, SCAST<U>(1), currentLength);
        Select(outF, cmp, outF, inputF, SEL_TT, currentLength);

#undef SEL_TT
#undef SCAST
#undef LEFT_SHIFT

        outQue.EnQue<U>(outLocal);
        inputQue.FreeTensor(inputLocal);
        otherQue.FreeTensor(otherLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        int32_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        int32_t currentPosition = progress * loopLength;
        if (progress == tileNum - 1) currentPosition += lastLoopLength;

        LocalTensor<U> outLocal = outQue.DeQue<U>();
        DataCopy(outGM[currentPosition], outLocal, currentLength);
        outQue.FreeTensor(outLocal);
    }

  private:
    TPipe pipe;
    GlobalTensor<U> inputGM;
    GlobalTensor<T> otherGM;
    GlobalTensor<U> outGM;
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

template <class T> class KernelBitwiseLeftShiftBoardCast {
    using U = typename unsigned_type<T>::type;
  public:
    __aicore__ inline KernelBitwiseLeftShiftBoardCast() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out,
                                int32_t former_length, int32_t tail_length,
                                int32_t former_num, int32_t tile_length,
                                uint8_t board_cast, int32_t out_dim1,
                                int32_t out_dim2, int32_t out_dim3) {

        int32_t coreID = GetBlockIdx();
        coreLength = (coreID < former_num) ? former_length : tail_length;

        tileLength = tile_length;
        tileNum = (coreLength / tileLength);
        lastTileLength = coreLength - tileLength * tileNum;

        if (lastTileLength > 0) tileNum++;
        else lastTileLength = tileLength;

        coreFormerLength = ((coreID < former_num)
                                ? (former_length * coreID)
                                : (tail_length * coreID +
                                   (former_length - tail_length) * former_num));

        // printf("#%d: %d %d %d\n", coreID, coreFormerLength, coreLength,
        //        tileLength);

        inputStride1 = ((board_cast >> 0) & 1) ^ 1;
        inputStride2 = ((board_cast >> 1) & 1) ^ 1;
        inputStride3 = ((board_cast >> 2) & 1) ^ 1;
        otherStride1 = ((board_cast >> 3) & 1) ^ 1;
        otherStride2 = ((board_cast >> 4) & 1) ^ 1;
        otherStride3 = ((board_cast >> 5) & 1) ^ 1;

        outDim1 = out_dim1;
        outDim2 = out_dim2;
        outDim3 = out_dim3;
        inputDim1 = inputStride1 ? outDim1 : 1;
        inputDim2 = inputStride2 ? outDim2 : 1;
        inputDim3 = inputStride3 ? outDim3 : 1;
        otherDim1 = otherStride1 ? outDim1 : 1;
        otherDim2 = otherStride2 ? outDim2 : 1;
        otherDim3 = otherStride3 ? outDim3 : 1;

        pipe.InitBuffer(inputQue, 1, tileLength * sizeof(U));
        pipe.InitBuffer(otherQue, 1, tileLength * sizeof(U));
        pipe.InitBuffer(outQue, 1, tileLength * sizeof(U));

        inputGM.SetGlobalBuffer((__gm__ U *)input, inputDim1 * inputDim2 *
                                                       inputDim3);
        otherGM.SetGlobalBuffer((__gm__ U *)other, otherDim1 * otherDim2 *
                                                       otherDim3);
        outGM.SetGlobalBuffer((__gm__ U *)out,
                              outDim1 * outDim2 * outDim3);
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
        int32_t beginLength = progress * tileLength + coreFormerLength;
        int32_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        LocalTensor<U> inputLocal = inputQue.AllocTensor<U>();
        LocalTensor<U> otherLocal = otherQue.AllocTensor<U>();

        DataCopy(inputLocal, inputGM[beginLength], currentLength);

        for (int32_t i = 0; i < currentLength; i++) {
            int32_t dim1 = (i + beginLength) / (outDim2 * outDim3);
            int32_t dim2 = (i + beginLength) / outDim3 % outDim2;
            int32_t dim3 = (i + beginLength) % outDim3;

            int32_t otherID = dim1 * otherStride1 * otherDim2 * otherDim3 +
                              dim2 * otherStride2 * otherDim3 +
                              dim3 * otherStride3;
            otherLocal(i) = otherGM(otherID);
        }

        inputQue.EnQue(inputLocal);
        otherQue.EnQue(otherLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        int32_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        LocalTensor<U> inputLocal = inputQue.DeQue<U>();
        LocalTensor<U> otherLocal = otherQue.DeQue<U>();
        LocalTensor<U> outLocal = outQue.AllocTensor<U>();

        for (int32_t i = 0; i < currentLength; i++) {
            U val = inputLocal(i), shift = otherLocal(i);
            outLocal(i) = (val << shift);
        }

        outQue.EnQue<U>(outLocal);
        inputQue.FreeTensor(inputLocal);
        otherQue.FreeTensor(otherLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        int32_t beginLength = progress * tileLength + coreFormerLength;
        int32_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        LocalTensor<U> outLocal = outQue.DeQue<U>();
        
        DataCopy(outGM[beginLength], outLocal, currentLength);

        outQue.FreeTensor(outLocal);
    }

  private:
    TPipe pipe;
    GlobalTensor<U> inputGM;
    GlobalTensor<U> otherGM;
    GlobalTensor<U> outGM;
    TQue<QuePosition::VECIN, 1> inputQue, otherQue;
    TQue<QuePosition::VECOUT, 1> outQue;
    TBuf<TPosition::VECCALC> cmpBuf;

    int32_t coreLength;
    int32_t coreFormerLength;
    int32_t tileNum;
    int32_t tileLength;
    int32_t lastTileLength;

    int8_t inputStride1, inputStride2, inputStride3;
    int8_t otherStride1, otherStride2, otherStride3;
    int32_t outDim1, outDim2, outDim3;
    int32_t inputDim1, inputDim2, inputDim3;
    int32_t otherDim1, otherDim2, otherDim3;
};

extern "C" __global__ __aicore__ void
bitwise_left_shift(GM_ADDR input, GM_ADDR other, GM_ADDR out, GM_ADDR workspace,
                   GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (tiling_data.board_cast) {
        KernelBitwiseLeftShiftBoardCast<DTYPE_INPUT> op;
        op.Init(input, other, out, tiling_data.former_length,
                tiling_data.tail_length, tiling_data.former_num, 
                tiling_data.tile_length, 
                tiling_data.board_cast, tiling_data.out_dim1,
                tiling_data.out_dim2, tiling_data.out_dim3);
        op.Process();
    } else {
        KernelBitwiseLeftShift<DTYPE_INPUT> op;
        op.Init(input, other, out, tiling_data.former_length,
                tiling_data.tail_length, tiling_data.former_num,
                tiling_data.tile_length);
        op.Process();
    }
}