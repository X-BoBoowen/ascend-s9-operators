#include "kernel_operator.h"
#include <type_traits>

using namespace AscendC;
constexpr uint16_t BufferNumSmall = 1;
class KernelGluGradJvpFp32Small {
public:
    __aicore__ inline KernelGluGradJvpFp32Small() {}
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

        x_gradGm.SetGlobalBuffer((__gm__ float *)x_grad + beginIndex * 2, size * 2);
        xGm.SetGlobalBuffer((__gm__ float *)x + beginIndex * 2, size * 2);
        v_xGm.SetGlobalBuffer((__gm__ float *)v_x + beginIndex * 2, size * 2);
        v_yGm.SetGlobalBuffer((__gm__ float *)v_y + beginIndex, size);
        jvp_outGm.SetGlobalBuffer((__gm__ float *)jvp_out + beginIndex * 2, size * 2);
        uint32_t maxSpaceSize = (uint32_t)(160 / 8) * 1024 / BufferNumSmall;
        blockCount = maxSpaceSize / ((hi_block_size * sizeof(float) + 31) / 32 * 32);
        blockLen = hi_block_size * sizeof(float);
        uint32_t spaceSize = blockCount * (hi_block_size * sizeof(float) + 31) / 32 * 32;
        this->n_elements_per_iter = spaceSize / sizeof(float);
        this->smallLoopTimes = (hi_block_num + blockCount - 1) / blockCount;

        pipe->InitBuffer(x1Buf, BufferNumSmall, spaceSize + 32);
        pipe->InitBuffer(x2Buf, BufferNumSmall, spaceSize + 32);
        pipe->InitBuffer(vx1Buf, BufferNumSmall, spaceSize + 32);
        pipe->InitBuffer(vx2Buf, BufferNumSmall, spaceSize + 32);
        pipe->InitBuffer(v_yBuf, BufferNumSmall, spaceSize + 32);
        pipe->InitBuffer(x1GradBuf, BufferNumSmall, spaceSize + 32);
        pipe->InitBuffer(jvp_out_aBuf, BufferNumSmall, spaceSize + 32);
        pipe->InitBuffer(jvp_out_bBuf, BufferNumSmall, spaceSize + 32);
    }
    __aicore__ inline void Process_iter(uint32_t inputIndex, uint32_t glu_outIndex, uint32_t iterSize) {
        LocalTensor<float> x1Local = x1Buf.AllocTensor<float>();
        LocalTensor<float> x2Local = x2Buf.AllocTensor<float>();
        LocalTensor<float> v_x1Local = vx1Buf.AllocTensor<float>();
        LocalTensor<float> v_x2Local = vx2Buf.AllocTensor<float>();
        LocalTensor<float> v_yLocal = v_yBuf.AllocTensor<float>();
        LocalTensor<float> x1_gradLocal = x1GradBuf.AllocTensor<float>();

        DataCopyExtParams copyParams{blockCount, blockLen, blockLen, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

        DataCopyPad(x1Local, xGm[inputIndex], copyParams, padParams);
        DataCopyPad(x2Local, xGm[inputIndex + hi_block_size], copyParams, padParams);
        DataCopyPad(v_x1Local, v_xGm[inputIndex], copyParams, padParams);
        DataCopyPad(v_x2Local, v_xGm[inputIndex + hi_block_size], copyParams, padParams);
        DataCopyPad(x1_gradLocal, x_gradGm[inputIndex], copyParams, padParams);

        DataCopyExtParams ycopyParams{blockCount, blockLen, 0, 0, 0};
        // 毒瘤是这个！
        DataCopyPad(v_yLocal, v_yGm[glu_outIndex], ycopyParams, padParams);
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

        // uint32_t index = (hi_block_size * sizeof(float) + 31) / 32 * 32 / sizeof(float);
         // 1. 计算sigmoid(b) jvp_out_bLocal
        Sigmoid(jvp_out_bLocal, x2Local, iterSize);

         // 2. 计算 glu_out_val = a * sigmoid(b) x2Local
        Mul(x2Local, x1Local, jvp_out_bLocal, iterSize);

         // 3. 计算 db_neg_sigmoid_b = v_b - v_b*sigmoid(b) v_x2Local
         // v_b*sigmoid(b) jvp_out_aLocal
        Mul(jvp_out_aLocal, v_x2Local, jvp_out_bLocal, iterSize);
        // db_neg_sigmoid_b = v_b - v_b*sigmoid(b) v_x2Local
        Sub(v_x2Local, v_x2Local, jvp_out_aLocal, iterSize);

        // 4. 计算 dgrad_x_a = v_y * sigmoid_b + grad_x_a * db_neg_sigmoid_b jvp_out_aLocal
        // term1 = v_y * sigmoid_b:jvp_out_aLocal
        Mul(jvp_out_aLocal, v_yLocal, jvp_out_bLocal, iterSize);
        // dgrad_x_a=term1+grad_x_a * db_neg_sigmoid_b: jvp_out_aLocal
        MulAddDst(jvp_out_aLocal, x1_gradLocal, v_x2Local, iterSize);

        // 5. 计算 dgrad_x_b
        // term2 = dgrad_x_a * (a - glu_out_val):x1Local
         // (a - glu_out_val):x1Local
        Sub(x1Local, x1Local, x2Local, iterSize);
        // term2 = dgrad_x_a * (a - glu_out_val):x1Local
        Mul(x1Local, jvp_out_aLocal, x1Local, iterSize);

        // term3 = grad_x_a * (v_a - v_a*sigmoid_b - glu_out_val*db_neg_sigmoid_b)
        // v_a * sigmoid_b:jvp_out_bLocal
        Mul(jvp_out_bLocal, v_x1Local, jvp_out_bLocal, iterSize);
         // v_a - v_a*sigmoid_b jvp_out_bLocal
        Sub(jvp_out_bLocal, v_x1Local, jvp_out_bLocal, iterSize);
        // glu_out_val * db_neg_sigmoid_b:jvp_out_bLocal:x2Local
        Mul(x2Local, x2Local, v_x2Local, iterSize);
        // (v_a - v_a*sigmoid_b - glu_out_val*db_neg_sigmoid_b):jvp_out_bLocal
        Sub(jvp_out_bLocal, jvp_out_bLocal, x2Local, iterSize);

        //  dgrad_x_b=grad_x_a * (v_a - v_a*sigmoid_b - glu_out_val*db_neg_sigmoid_b)+dgrad_x_a * (a - glu_out_val):jvp_out_bLocal
        FusedMulAdd(jvp_out_bLocal, x1_gradLocal, x1Local, iterSize);

        x1Buf.FreeTensor<float>(x1Local);
        x2Buf.FreeTensor<float>(x2Local);
        vx1Buf.FreeTensor<float>(v_x1Local);
        vx2Buf.FreeTensor<float>(v_x2Local);
        v_yBuf.FreeTensor<float>(v_yLocal);
        x1GradBuf.FreeTensor<float>(x1_gradLocal);

        jvp_out_aBuf.EnQue<float>(jvp_out_aLocal);
        jvp_out_bBuf.EnQue<float>(jvp_out_bLocal);
        jvp_out_aLocal = jvp_out_aBuf.DeQue<float>();
        jvp_out_bLocal = jvp_out_bBuf.DeQue<float>();

        // 将结果写回全局内存
        DataCopyExtParams storeParams{blockCount, blockLen, 0, blockLen, 0};
        DataCopyPad(jvp_outGm[inputIndex], jvp_out_aLocal, storeParams);
        DataCopyPad(jvp_outGm[inputIndex + hi_block_size], jvp_out_bLocal, storeParams);
        jvp_out_aBuf.FreeTensor<float>(jvp_out_aLocal);
        jvp_out_bBuf.FreeTensor<float>(jvp_out_bLocal);
    }
    __aicore__ inline void Process() {
        uint32_t blockBeginIndex = 0;

        for (uint32_t i = 0; i < smallLoopTimes; i++) {
            Process_iter(blockBeginIndex * 2, blockBeginIndex, n_elements_per_iter);
            blockBeginIndex += blockCount * hi_block_size;
        }
    }

private:
    TPipe *pipe;
    GlobalTensor<float> x_gradGm;
    GlobalTensor<float> xGm;
    GlobalTensor<float> v_xGm;
    GlobalTensor<float> v_yGm;
    GlobalTensor<float> jvp_outGm;
    TQue<QuePosition::VECIN, 1> x1Buf;
    TQue<QuePosition::VECIN, 1> x2Buf;
    TQue<QuePosition::VECIN, 1> vx1Buf;
    TQue<QuePosition::VECIN, 1> vx2Buf;
    TQue<QuePosition::VECIN, 1> v_yBuf;
    TQue<QuePosition::VECIN, 1> x1GradBuf;

    TQue<QuePosition::VECOUT, 1> jvp_out_aBuf;
    TQue<QuePosition::VECOUT, 1> jvp_out_bBuf;

    int KS;
    int J;
    int HI;

    uint32_t size;
    uint32_t smallLoopTimes;
    uint32_t n_elements_per_iter;
    uint32_t n_hi_blocks_per_iter;
    uint32_t hi_block_size;
    uint32_t hi_block_num;
    uint16_t blockCount;
    uint32_t blockLen;
};
