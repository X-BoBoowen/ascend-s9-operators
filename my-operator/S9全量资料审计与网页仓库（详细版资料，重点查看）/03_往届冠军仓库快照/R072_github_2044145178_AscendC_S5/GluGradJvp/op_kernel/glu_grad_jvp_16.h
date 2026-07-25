#include "kernel_operator.h"
#include <type_traits>

using namespace AscendC;
#define BufferNum 1

// const uint16_t BufferNum = 1;
template <typename T>
class KernelGluGradJvp16 {
public:
    __aicore__ inline KernelGluGradJvp16() {}
    __aicore__ inline void Init(GM_ADDR x_grad, GM_ADDR y_grad, GM_ADDR x, GM_ADDR v_y, GM_ADDR v_x, GM_ADDR jvp_out, int HI, int J, int KS, uint32_t smallSize, uint32_t incSize, uint16_t formerNum,
                                TPipe *pipeIn) {
        this->pipe = pipeIn;
        this->HI = HI;
        this->J = J;
        this->KS = KS;
        int totalSize = HI * J * KS;
        this->hi_block_size = J * KS / 2;
        uint32_t beginIndex = 0;
        if (GetBlockIdx() < formerNum) {
            this->size = smallSize + incSize;
            beginIndex = this->size * GetBlockIdx();
        } else {
            this->size = smallSize;
            beginIndex = this->size * GetBlockIdx() + formerNum * incSize;
        }

        this->hi_block_num = this->size / this->hi_block_size;

        x_gradGm.SetGlobalBuffer((__gm__ T *)x_grad + beginIndex * 2, size * 2);
        xGm.SetGlobalBuffer((__gm__ T *)x + beginIndex * 2, size * 2);
        v_xGm.SetGlobalBuffer((__gm__ T *)v_x + beginIndex * 2, size * 2);
        v_yGm.SetGlobalBuffer((__gm__ T *)v_y + beginIndex, size);
        jvp_outGm.SetGlobalBuffer((__gm__ T *)jvp_out + beginIndex * 2, size * 2);

        uint32_t spaceSize = min((uint32_t)(hi_block_size * sizeof(float)), (uint32_t)(160 / 10) * 1024 / BufferNum);
        this->n_elements_per_iter = spaceSize / sizeof(float);
        this->bigLoopTimes = hi_block_num;
        this->smallLoopTimes = (hi_block_size + n_elements_per_iter - 1) / n_elements_per_iter;
        pipe->InitBuffer(x1Buf, BufferNum, spaceSize + 32);
        pipe->InitBuffer(x2Buf, BufferNum, spaceSize + 32);
        pipe->InitBuffer(vx1Buf, BufferNum, spaceSize + 32);
        pipe->InitBuffer(vx2Buf, BufferNum, spaceSize + 32);
        pipe->InitBuffer(v_yBuf, BufferNum, spaceSize + 32);
        pipe->InitBuffer(x1GradBuf, BufferNum, spaceSize + 32);
        pipe->InitBuffer(jvp_out_aBuf, BufferNum, spaceSize + 32);
        pipe->InitBuffer(jvp_out_bBuf, BufferNum, spaceSize + 32);

        pipe->InitBuffer(inputTmpBuf, BufferNum, spaceSize / 2 + 32);
    }

    __aicore__ inline void Process_iter(uint32_t inputIndex, uint32_t glu_outIndex, uint32_t iterSize) {
        uint16_t blockCount = 1;
        uint32_t blockLen = iterSize * sizeof(T);
        DataCopyExtParams copyParams{blockCount, blockLen, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        LocalTensor<float> x1Local = x1Buf.AllocTensor<float>();
        LocalTensor<float> x2Local = x2Buf.AllocTensor<float>();
        LocalTensor<float> v_x1Local = vx1Buf.AllocTensor<float>();
        LocalTensor<float> v_x2Local = vx2Buf.AllocTensor<float>();
        LocalTensor<float> v_yLocal = v_yBuf.AllocTensor<float>();
        LocalTensor<float> x1_gradLocal = x1GradBuf.AllocTensor<float>();
        // 读入
        LocalTensor<T> x1Local_16 = inputTmpBuf.AllocTensor<T>();
        DataCopyPad(x1Local_16, xGm[inputIndex], copyParams, padParams);
        inputTmpBuf.EnQue<T>(x1Local_16);
        x1Local_16 = inputTmpBuf.DeQue<T>();
        Cast(x1Local, x1Local_16, RoundMode::CAST_NONE, iterSize);
        inputTmpBuf.FreeTensor<T>(x1Local_16);

        LocalTensor<T> x2Local_16 = inputTmpBuf.AllocTensor<T>();
        DataCopyPad(x2Local_16, xGm[inputIndex + hi_block_size], copyParams, padParams);
        inputTmpBuf.EnQue<T>(x2Local_16);
        x2Local_16 = inputTmpBuf.DeQue<T>();
        Cast(x2Local, x2Local_16, RoundMode::CAST_NONE, iterSize);
        inputTmpBuf.FreeTensor<T>(x2Local_16);

        LocalTensor<T> v_x1Local_16 = inputTmpBuf.AllocTensor<T>();
        DataCopyPad(v_x1Local_16, v_xGm[inputIndex], copyParams, padParams);
        inputTmpBuf.EnQue<T>(v_x1Local_16);
        v_x1Local_16 = inputTmpBuf.DeQue<T>();
        Cast(v_x1Local, v_x1Local_16, RoundMode::CAST_NONE, iterSize);
        inputTmpBuf.FreeTensor<T>(v_x1Local_16);

        LocalTensor<T> v_x2Local_16 = inputTmpBuf.AllocTensor<T>();
        DataCopyPad(v_x2Local_16, v_xGm[inputIndex + hi_block_size], copyParams, padParams);
        inputTmpBuf.EnQue<T>(v_x2Local_16);
        v_x2Local_16 = inputTmpBuf.DeQue<T>();
        Cast(v_x2Local, v_x2Local_16, RoundMode::CAST_NONE, iterSize);
        inputTmpBuf.FreeTensor<T>(v_x2Local_16);

        LocalTensor<T> v_yLocal_16 = inputTmpBuf.AllocTensor<T>();
        DataCopyPad(v_yLocal_16, v_yGm[glu_outIndex], copyParams, padParams);
        inputTmpBuf.EnQue<T>(v_yLocal_16);
        v_yLocal_16 = inputTmpBuf.DeQue<T>();
        Cast(v_yLocal, v_yLocal_16, RoundMode::CAST_NONE, iterSize);
        inputTmpBuf.FreeTensor<T>(v_yLocal_16);

        LocalTensor<T> x1_gradLocal_16 = inputTmpBuf.AllocTensor<T>();
        DataCopyPad(x1_gradLocal_16, x_gradGm[inputIndex], copyParams, padParams);
        inputTmpBuf.EnQue<T>(x1_gradLocal_16);
        x1_gradLocal_16 = inputTmpBuf.DeQue<T>();
        Cast(x1_gradLocal, x1_gradLocal_16, RoundMode::CAST_NONE, iterSize);
        inputTmpBuf.FreeTensor<T>(x1_gradLocal_16);

        x1Buf.EnQue<float>(x1Local);
        x2Buf.EnQue<float>(x2Local);
        vx1Buf.EnQue<float>(v_x1Local);
        vx2Buf.EnQue<float>(v_x2Local);
        v_yBuf.EnQue<float>(v_yLocal);
        x1GradBuf.EnQue<float>(x1_gradLocal);
        x1Local = x1Buf.DeQue<float>();
        x2Local = x2Buf.DeQue<float>();
        v_x1Local = vx1Buf.DeQue<float>();
        v_x2Local = vx2Buf.DeQue<float>();
        v_yLocal = v_yBuf.DeQue<float>();
        x1_gradLocal = x1GradBuf.DeQue<float>();

        LocalTensor<float> jvp_out_aLocal = jvp_out_aBuf.AllocTensor<float>();
        LocalTensor<float> jvp_out_bLocal = jvp_out_bBuf.AllocTensor<float>();
        LocalTensor<T> jvp_out_aLocal_16 = jvp_out_aLocal.ReinterpretCast<T>();
        LocalTensor<T> jvp_out_bLocal_16 = jvp_out_bLocal.ReinterpretCast<T>();
        Sigmoid(jvp_out_bLocal, x2Local, iterSize);
        Mul(x2Local, x1Local, jvp_out_bLocal, iterSize);
        Mul(jvp_out_aLocal, v_x2Local, jvp_out_bLocal, iterSize);
        Sub(v_x2Local, v_x2Local, jvp_out_aLocal, iterSize);
        Mul(jvp_out_aLocal, v_yLocal, jvp_out_bLocal, iterSize);
        MulAddDst(jvp_out_aLocal, x1_gradLocal, v_x2Local, iterSize);
        Sub(x1Local, x1Local, x2Local, iterSize);
        Mul(x1Local, jvp_out_aLocal, x1Local, iterSize);
        Cast(jvp_out_aLocal_16, jvp_out_aLocal, RoundMode::CAST_RINT, iterSize);
        Mul(jvp_out_bLocal, v_x1Local, jvp_out_bLocal, iterSize);
        Sub(jvp_out_bLocal, v_x1Local, jvp_out_bLocal, iterSize);
        Mul(x2Local, x2Local, v_x2Local, iterSize);
        Sub(jvp_out_bLocal, jvp_out_bLocal, x2Local, iterSize);
        FusedMulAdd(jvp_out_bLocal, x1_gradLocal, x1Local, iterSize);
        x1Buf.FreeTensor<float>(x1Local);
        x2Buf.FreeTensor<float>(x2Local);
        vx1Buf.FreeTensor<float>(v_x1Local);
        vx2Buf.FreeTensor<float>(v_x2Local);
        v_yBuf.FreeTensor<float>(v_yLocal);
        x1GradBuf.FreeTensor<float>(x1_gradLocal);


        Cast(jvp_out_bLocal_16, jvp_out_bLocal, RoundMode::CAST_RINT, iterSize);

        jvp_out_aBuf.EnQue<float>(jvp_out_aLocal);
        jvp_out_bBuf.EnQue<float>(jvp_out_bLocal);
        jvp_out_aLocal = jvp_out_aBuf.DeQue<float>();
        jvp_out_bLocal = jvp_out_bBuf.DeQue<float>();

        // 将结果写回全局内存
        DataCopyExtParams storeParams{1, (uint32_t)(iterSize * sizeof(T)), 0, 0, 0};
        DataCopyPad(jvp_outGm[inputIndex], jvp_out_aLocal_16, storeParams);
        DataCopyPad(jvp_outGm[inputIndex + hi_block_size], jvp_out_bLocal_16, storeParams);
        jvp_out_aBuf.FreeTensor<float>(jvp_out_aLocal);
        jvp_out_bBuf.FreeTensor<float>(jvp_out_bLocal);
    }
    __aicore__ inline void Process() {
        uint32_t blockBeginIndex = 0;
        // printf("KernelGluJvpFp32 Process: size=%d, bigLoopTimes=%d, smallLoopTimes=%d, n_elements_per_iter=%d\n", size, bigLoopTimes, smallLoopTimes, n_elements_per_iter);
        // printf("KernelGluJvpFp32 Process: size=%d, hi_block_num=%d, hi_block_size=%d, n_elements_per_iter=%d\n", size, hi_block_num, hi_block_size, n_elements_per_iter);
        for (uint32_t i = 0; i < bigLoopTimes; i++) {
            uint32_t iterInputIndex = 0;
            for (uint32_t j = 0; j < smallLoopTimes; j++) {
                uint32_t iterSize = min(n_elements_per_iter, hi_block_size - iterInputIndex);
                // if (GetBlockIdx() == 0) {
                //     printf("KernelGluJvpFp32 Process: blockBeginIndex=%d, iterBeginIndex=%d, iterSize=%d\n", blockBeginIndex, iterBeginIndex, iterSize);
                // }
                Process_iter(blockBeginIndex * 2 + iterInputIndex, blockBeginIndex + iterInputIndex, iterSize);
                iterInputIndex += n_elements_per_iter;
            }
            blockBeginIndex += hi_block_size;
        }
    }

private:
    TPipe *pipe;
    GlobalTensor<T> x_gradGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> v_xGm;
    GlobalTensor<T> v_yGm;
    GlobalTensor<T> jvp_outGm;
    TQue<QuePosition::VECIN, 1> x1Buf;
    TQue<QuePosition::VECIN, 1> x2Buf;
    TQue<QuePosition::VECIN, 1> vx1Buf;
    TQue<QuePosition::VECIN, 1> vx2Buf;
    TQue<QuePosition::VECIN, 1> v_yBuf;
    TQue<QuePosition::VECIN, 1> x1GradBuf;

    TQue<QuePosition::VECOUT, 1> jvp_out_aBuf;
    TQue<QuePosition::VECOUT, 1> jvp_out_bBuf;

    TQue<QuePosition::VECIN, 1> inputTmpBuf;

    int KS;
    int J;
    int HI;

    uint32_t size;
    uint32_t smallLoopTimes;
    uint32_t bigLoopTimes;
    uint32_t n_elements_per_iter;
    uint32_t n_hi_blocks_per_iter;
    uint32_t hi_block_size;
    uint32_t hi_block_num;
};
