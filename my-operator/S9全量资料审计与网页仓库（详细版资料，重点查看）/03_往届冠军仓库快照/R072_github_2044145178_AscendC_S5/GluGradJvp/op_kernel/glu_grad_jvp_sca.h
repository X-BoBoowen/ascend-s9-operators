#include "kernel_operator.h"
#include <type_traits>

using namespace AscendC;
class KernelGluGradJvpSca {
public:
    __aicore__ inline KernelGluGradJvpSca() {}
    __aicore__ inline void Init(GM_ADDR x_grad, GM_ADDR y_grad, GM_ADDR x, GM_ADDR v_y, GM_ADDR v_x, GM_ADDR jvp_out, int HI, int J, int KS, TPipe *pipeIn) {
        this->totalSize = HI * J * KS;

        x_gradGm.SetGlobalBuffer((__gm__ DTYPE_X *)x_grad, totalSize);
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, totalSize);
        v_xGm.SetGlobalBuffer((__gm__ DTYPE_X *)v_x, totalSize);
        v_yGm.SetGlobalBuffer((__gm__ DTYPE_X *)v_y, totalSize / 2);

        jvp_outGm.SetGlobalBuffer((__gm__ DTYPE_X *)jvp_out, totalSize / 2);
        pipe = pipeIn;
        pipe->InitBuffer(inBuf, 512);
        pipe->InitBuffer(outBuf, 512);
        this->HI = HI;
        this->J = J;
        this->KS = KS;
    }
    __aicore__ inline void Process() {
        int index = 0;

        for (int hi = 0; hi < HI; hi++) {
            for (int j = J / 2; j < J; j++) {
                for (int ks = 0; ks < KS; ks++) {
                    int input1_idx = hi * J * KS + (j - J / 2) * KS + ks;
                    int input2_idx = hi * J * KS + j * KS + ks;
                    if constexpr (std::is_same<DTYPE_X, bfloat16_t>::value) {
                        float x1_grad_val = ToFloat(x_gradGm.GetValue(input1_idx));
                        float x1_val = ToFloat(xGm.GetValue(input1_idx));
                        float x2_val = ToFloat(xGm.GetValue(input2_idx));
                        float v_x1_val = ToFloat(v_xGm.GetValue(input1_idx));
                        float v_x2_val = ToFloat(v_xGm.GetValue(input2_idx));
                        float v_y_val = ToFloat(v_yGm.GetValue(index));

                        LocalTensor<float> inLocal = inBuf.Get<float>();
                        inLocal.SetValue(0, x2_val);
                        LocalTensor<float> outLocal = outBuf.Get<float>();
                        Sigmoid(outLocal, inLocal, 32);
                        float sigmoid_b = outLocal.GetValue(0);

                        float glu_out_val = x1_val * sigmoid_b;
                        float db_neg_sigmoid_b = v_x2_val - v_x2_val * sigmoid_b;
                        float dgrad_x_a = v_y_val * sigmoid_b + x1_grad_val * db_neg_sigmoid_b;
                        float dgrad_x_b = dgrad_x_a * (x1_val - glu_out_val) + x1_grad_val * (v_x1_val - v_x1_val * sigmoid_b - glu_out_val * db_neg_sigmoid_b);

                        LocalTensor<bfloat16_t> outLocal_bf16 = outBuf.Get<bfloat16_t>();
                        inLocal.SetValue(0, dgrad_x_a);
                        inLocal.SetValue(1, dgrad_x_b);
                        Cast(outLocal_bf16, inLocal, RoundMode::CAST_RINT, 32);
                        bfloat16_t out_1_bf16 = outLocal_bf16.GetValue(0);
                        bfloat16_t out_2_bf16 = outLocal_bf16.GetValue(1);

                        jvp_outGm.SetValue(input1_idx, out_1_bf16);
                        jvp_outGm.SetValue(input2_idx, out_2_bf16);

                    } else {
                        float x1_grad_val = x_gradGm.GetValue(input1_idx);
                        float x1_val = xGm.GetValue(input1_idx);
                        float x2_val = xGm.GetValue(input2_idx);
                        float v_x1_val = v_xGm.GetValue(input1_idx);
                        float v_x2_val = v_xGm.GetValue(input2_idx);
                        float v_y_val = v_yGm.GetValue(index);

                        LocalTensor<float> inLocal = inBuf.Get<float>();
                        inLocal.SetValue(0, x2_val);
                        LocalTensor<float> outLocal = outBuf.Get<float>();
                        Sigmoid(outLocal, inLocal, 32);
                        float sigmoid_b = outLocal.GetValue(0);

                        float glu_out_val = x1_val * sigmoid_b;
                        float db_neg_sigmoid_b = v_x2_val - v_x2_val * sigmoid_b;
                        float dgrad_x_a = v_y_val * sigmoid_b + x1_grad_val * db_neg_sigmoid_b;
                        float dgrad_x_b = dgrad_x_a * (x1_val - glu_out_val) + x1_grad_val * (v_x1_val - v_x1_val * sigmoid_b - glu_out_val * db_neg_sigmoid_b);

                        jvp_outGm.SetValue(input1_idx, dgrad_x_a);
                        jvp_outGm.SetValue(input2_idx, dgrad_x_b);
                    }
                    index++;
                }
            }
        }
    }

private:
    TPipe *pipe;
    GlobalTensor<DTYPE_X> x_gradGm;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_X> v_xGm;
    GlobalTensor<DTYPE_X> v_yGm;

    GlobalTensor<DTYPE_X> jvp_outGm;

    TBuf<QuePosition::VECCALC> inBuf;
    TBuf<QuePosition::VECCALC> outBuf;
    int KS;
    int J;
    int HI;
    int totalSize;
};