#include "kernel_operator.h"

using namespace AscendC;
constexpr int BUFFER_NUM = 1;
constexpr int BLOCK_SIZE = 32;


template <typename T>
class KernelGcdSimple {

public:
    __aicore__ inline KernelGcdSimple() {}

    __aicore__ inline void Init(
        TPipe* pipe,
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR y,
        GcdTilingData* tiling
    ) {
        int coreIdx = GetBlockIdx();
        int bigCoreNum = tiling->bigCoreNum;
        int bigCoreLen = tiling->bigCoreLen;
        int offset = bigCoreLen * coreIdx;

        // 小核
        if (coreIdx >= bigCoreNum) {
            int smallCoreLen = tiling->smallCoreLen;
            offset -= (coreIdx - bigCoreNum) * (bigCoreLen - smallCoreLen);
            this->totalLen = smallCoreLen;
            this->tileLen = tiling->smallCoreTileLen;
            this->tileNum = tiling->smallCoreTileNum;
            this->tailTileLen = tiling->smallCoreTailTileLen;
        } else {
            this->totalLen = bigCoreLen;
            this->tileLen = tiling->bigCoreTileLen;
            this->tileNum = tiling->bigCoreTileNum;
            this->tailTileLen = tiling->bigCoreTailTileLen;
        }

        this->pipe = pipe;
        this->offset = offset;


        x1Gm.SetGlobalBuffer((__gm__ T *)x1);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2);
        yGm.SetGlobalBuffer((__gm__ T *)y);

        this->pipe->InitBuffer(yQue, BUFFER_NUM, tileLen * sizeof(T));

        this->shapeDim = tiling->shapeDim;
        for (int i = 0; i < shapeDim; i++) {
            this->shapes[0][i] = tiling->x1Shape[i];
            this->shapes[1][i] = tiling->x2Shape[i];
            this->shapes[2][i] = tiling->yShape[i];
        }
        for (int i = 0; i < 3; i++) {
            strides[i][shapeDim - 1] = 1;
        }
        for (int i = shapeDim - 2; i >= 0; i--) {
            strides[0][i] = strides[0][i + 1] * shapes[0][i + 1];
            strides[1][i] = strides[1][i + 1] * shapes[1][i + 1];
            strides[2][i] = strides[2][i + 1] * shapes[2][i + 1];
        }
    }


    __aicore__ inline void Process() {
        for (int i = 0; i < tileNum; i++) {
            int inputOffset = this->offset + i * tileLen;
            LocalTensor<T> y = yQue.AllocTensor<T>();
            for (int j = 0; j < tileLen; j++) {
                int x1Offset = inputOffset + j;
                int x2Offset = 0;
                for (int k = 0; k < shapeDim; k++) {
                    int index = x1Offset / strides[2][k] % shapes[2][k];
                    x2Offset += (index % shapes[1][k]) * strides[1][k];
                }
                T x1Value = x1Gm(x1Offset);
                T x2Value = x2Gm(x2Offset);
                abs(x1Value, x1Value);
                abs(x2Value, x2Value);
                gcd(x1Value, x2Value);
                y.SetValue(j, x1Value);
            }
            yQue.EnQue<T>(y);
            CopyOut(inputOffset, tileLen, 1);
        }

        int inputOffset = this->offset + tileNum * tileLen;
        LocalTensor<T> y = yQue.AllocTensor<T>();
        for (int j = 0; j < tailTileLen; j++) {
            int x1Offset = inputOffset + j;
            int x2Offset = 0;
            for (int k = 0; k < shapeDim; k++) {
                int index = x1Offset / strides[2][k] % shapes[2][k];
                x2Offset += (index % shapes[1][k]) * strides[1][k];
            }
            T x1Value = x1Gm(x1Offset);
            T x2Value = x2Gm(x2Offset);
            abs(x1Value, x1Value);
            abs(x2Value, x2Value);
            gcd(x1Value, x2Value);
            y.SetValue(j, x1Value);
        }
        CopyOut(inputOffset, tailTileLen, 1);
    }

    __aicore__ inline void CopyIn(
        uint64_t offset,
        uint32_t dataLen,
        uint16_t repeatTime
    ) {
        uint32_t dataSize = dataLen * sizeof(T);
        LocalTensor<T> x1 = x1Que.AllocTensor<T>();
        DataCopyExtParams copyParams {repeatTime, dataSize, 0, 0, 0};
        DataCopyPadExtParams<T> padParams {false, 0, 0, 0};
        DataCopyPad(x1, x1Gm[offset], copyParams, padParams);
        x1Que.EnQue<T>(x1);
    }

    __aicore__ inline void CopyOut(
        uint64_t offset,
        uint32_t dataLen,
        uint16_t repeatTime
    ) {
        uint32_t dataSize = dataLen * sizeof(T);
        LocalTensor<T> y = yQue.DeQue<T>();
        DataCopyExtParams copyParams {repeatTime, dataSize, 0, 0, 0};
        DataCopyPad(yGm[offset], y, copyParams);
        yQue.FreeTensor(y);
    }

    __aicore__ inline void PrintArray(int64_t* arr, int64_t len) {
        for (int i = 0; i < len; i++) {
            printf("%ld, ", arr[i]);
        }
        printf("\n");
    }

    __aicore__ inline void gcd(T& x1, T& x2) {
        if (x1 == 0) {
            x1 = x2;
            return;
        }
        if (x2 == 0) {
            return;
        }
        while (x2 != 0) {
            T temp = x2;
            x2 = x1 % x2;
            x1 = temp;
        }
    }

    __aicore__ inline void abs(T x, T& y) {
        if (x < 0) {
            y = -x;
        }
        else {
            y = x;
        }
    }

private:
    TPipe* pipe;
    GlobalTensor<T> x1Gm, x2Gm, yGm;
    
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQue;
    uint32_t totalLen, tileLen, tileNum, tailTileLen;

    uint32_t offset;
    
    int64_t shapes[3][10], strides[3][10];
    int64_t shapeDim;
};


extern "C" __global__ __aicore__ void gcd(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    TPipe pipe;
    KernelGcdSimple<DTYPE_X1> op;
    op.Init(&pipe, x1, x2, y, &tiling_data);
    op.Process();
}