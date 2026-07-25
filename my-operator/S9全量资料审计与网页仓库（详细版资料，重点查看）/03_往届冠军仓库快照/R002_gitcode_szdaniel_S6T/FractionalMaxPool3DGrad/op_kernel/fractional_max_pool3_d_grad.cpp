#include "kernel_operator.h"
using namespace AscendC;
#define BUFFER_NUM 2

#define BUFFER_NUM 2
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define ROUND_UP(a, b) (((a) + (b) - 1) / (b) * (b))
#define ROUND_DOWN(a, b) ((a) / (b) * (b))
#define ELE2BLOCK(x) ((x * sizeof(T) + 31) / 32)


template <typename dataType, uint32_t tileKey> class KernelFractionalMaxPool3DGrad;

template <typename T> class KernelFractionalMaxPool3DGrad<T, 0> {
public:
    __aicore__ inline KernelFractionalMaxPool3DGrad() {}

    __aicore__ inline void Init(GM_ADDR grad_output, GM_ADDR input, GM_ADDR indices, GM_ADDR random_sample, GM_ADDR out,
                                FractionalMaxPool3DGradTilingData tiling)
    {
        this->out_gm_addr = out;

        N = tiling.N;
        C = tiling.C;
        TI = tiling.TI;
        HI = tiling.HI;
        WI = tiling.WI;
        TO = tiling.TO;
        HO = tiling.HO;
        WO = tiling.WO;

        int inBlockSize = N * C * TI * HI * WI;
        int outBlockSize = N * C * TO * HO * WO;

        int coreID = GetBlockIdx();
        int NC = N * C;
        int blockNum = MIN(GetBlockNum(), NC);

        uint32_t formerLength = tiling.formerLength;
        uint32_t formerNum = tiling.formerNum;
        uint32_t tailLength = tiling.tailLength;
        uint32_t tailNum = tiling.tailNum;

        if (coreID < formerNum) {
            curNCstart = coreID * formerLength;
            curNCend = (coreID + 1) * formerLength;
        } else {
            curNCstart = formerNum * formerLength + (coreID - formerNum) * tailLength;
            curNCend = curNCstart + tailLength;
        }


        grad_outputGm.SetGlobalBuffer((__gm__ T *)grad_output, outBlockSize);
        inputGm.SetGlobalBuffer((__gm__ T *)input, inBlockSize);
        indicesGm.SetGlobalBuffer((__gm__ int32_t *)indices, outBlockSize);
        random_sampleGm.SetGlobalBuffer((__gm__ T *)random_sample, N * C * 3);
        outGm.SetGlobalBuffer((__gm__ T *)out, inBlockSize);

        const int maxLength = 16 * 1024;
        pipe.InitBuffer(outQueue, BUFFER_NUM, maxLength * sizeof(T));

        pipe.InitBuffer(tBuf[0], 256);
        pipe.InitBuffer(tBuf[1], 256);
        pipe.InitBuffer(tBuf[2], 256);
    }
    __aicore__ inline void SetZero()
    {
        for (int i = 0; i < N * C * TI * HI * WI; i++) {
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                outGm(i) = ToBfloat16(0.f);
            } else {
                outGm(i) = (T)0.f;
            }
        }
    }
    __aicore__ inline void outGmMemSetZero(uint32_t offset, uint32_t len)
    {
        const int maxLength = 16 * 1024;

        for (uint32_t i = 0; i < len; i += maxLength) {
            uint32_t curLen = MIN(maxLength, len - i);
            uint32_t curOffset = offset + i;

            //
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                AscendC::LocalTensor<uint16_t> outLocal = outQueue.AllocTensor<uint16_t>();
                Duplicate<uint16_t>(outLocal, (uint16_t)0, curLen);
                outQueue.EnQue(outLocal);
            } else {
                AscendC::LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
                Duplicate(outLocal, (T)0.f, curLen);
                outQueue.EnQue(outLocal);
            }


            //
            LocalTensor<T> z = outQueue.DeQue<T>();
            DataCopyExtParams copyParams = {1, (uint32_t)(curLen * sizeof(T)), 0, 0, 0};
            DataCopyPad(outGm[curOffset], z, copyParams);
            outQueue.FreeTensor(z);
        }
    }

    __aicore__ inline void Process()
    {
        // SetZero();

        LocalTensor<float> tLocal0 = tBuf[0].Get<float>();
        LocalTensor<T> tLocal1 = tBuf[1].Get<T>();

        uint32_t iPlaneSize = TI * HI * WI;
        uint32_t oPlaneSize = TO * HO * WO;

        outGmMemSetZero(iPlaneSize * curNCstart, iPlaneSize * (curNCend - curNCstart));
        // T zero;
        // if constexpr (std::is_same_v<T, bfloat16_t>) {
        //     zero = ToBfloat16(0.f);
        // } else {
        //     zero = (T)0.f;
        // }
        // GlobalTensor<T> tmpOutGm;
        // tmpOutGm.SetGlobalBuffer((__gm__ T *)out_gm_addr + iPlaneSize * curNCstart, N * C * TI * HI * WI);
        // InitGlobalMemory(tmpOutGm, iPlaneSize * (curNCend - curNCstart), zero);

        uint32_t HWO = HO * WO;
        uint32_t THWO = TO*HWO;
        for (int i = curNCstart; i < curNCend; i++) {
            uint32_t iPlaneBase = i * iPlaneSize;
            uint32_t oPlaneBase = i * oPlaneSize;

            for (int32_t thw = oPlaneBase; thw < oPlaneBase + THWO; ++thw) {
                int32_t index = indicesGm(thw);
                index +=iPlaneBase;

                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    float grad_o = ToFloat((bfloat16_t)grad_outputGm(thw));
                    float o = ToFloat((bfloat16_t)outGm(index));

                    tLocal0(0) = grad_o + o;
                    Cast(tLocal1, tLocal0, AscendC::RoundMode::CAST_RINT, 1);

                    outGm(index) = tLocal1(0);

                } else {
                    float grad_o = grad_outputGm(thw);
                    float o = outGm(index);
                    outGm(index) = (T)(grad_o + o);
                }                    
            }
        }
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> grad_outputQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> indicesQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> random_sampleQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    GlobalTensor<T> grad_outputGm;
    GlobalTensor<T> inputGm;
    GlobalTensor<int32_t> indicesGm;
    GlobalTensor<T> random_sampleGm;
    GlobalTensor<T> outGm;

    TBuf<TPosition::VECCALC> tBuf[3];

    int N, C, TI, HI, WI;
    int TO, HO, WO;
    uint32_t curNCstart, curNCend;

    GM_ADDR out_gm_addr;
};


extern "C" __global__ __aicore__ void fractional_max_pool3_d_grad(GM_ADDR grad_output, GM_ADDR input, GM_ADDR indices,
                                                                  GM_ADDR random_sample, GM_ADDR out, GM_ADDR workspace,
                                                                  GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    if (TILING_KEY_IS(0)) {
        KernelFractionalMaxPool3DGrad<DTYPE_INPUT, 0> op;
        op.Init(grad_output, input, indices, random_sample, out, tiling_data);
        op.Process();
    }
}