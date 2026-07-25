#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class BruteForce
{
public:
    __aicore__ inline BruteForce() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR random_sample, GM_ADDR out, uint32_t kernelSize[3], uint32_t inputSize[3], 
                                uint32_t outputSize[3], uint32_t n, uint32_t c, uint32_t bigCoreNum, uint64_t bigCoreProcessNum, uint64_t smallCoreProcessNum, TPipe *pipe)
    {
        uint32_t coreNum = GetBlockIdx();
        this->n = n;
        this->c = c;
        for(int i = 0; i < 3; i ++)
        {
            this->kernelSize[i] = kernelSize[i];
            this->inputSize[i] = inputSize[i];
            this->outputSize[i] = outputSize[i];
        }
        this->globalBufferIndex = coreNum * bigCoreProcessNum;
        if(coreNum < bigCoreNum)
            this->processNum = bigCoreProcessNum;
        else
        {
            this->processNum = smallCoreProcessNum;
            this->globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (coreNum - bigCoreNum);
        }
        int align = 32 / sizeof(T);
        uint32_t sz = kernelSize[0] * kernelSize[1] * kernelSize[2];
        uint32_t inputSz = kernelSize[0] * ((inputSize[2] * kernelSize[1] + align - 1) / align * align);
        x1Gm.SetGlobalBuffer((__gm__ T *)input, (uint64_t)n * c * inputSize[0] * inputSize[1] * inputSize[2]);
        x2Gm.SetGlobalBuffer((__gm__ T *)random_sample, (uint64_t)n * c * 3);
        yGm.SetGlobalBuffer((__gm__ T *)out, (uint64_t)n * c * outputSize[0] * outputSize[1] * outputSize[2]);
        pipe->InitBuffer(Qin, 1, inputSz * sizeof(T));
        pipe->InitBuffer(Qout, 1, 32);
        pipe->InitBuffer(B1, 32);
        pipe->InitBuffer(B2, 32);
        pipe->InitBuffer(B3, 32);
        pipe->InitBuffer(BT, outputSize[0] * 4);
        pipe->InitBuffer(BH, outputSize[1] * 4);
        pipe->InitBuffer(BW, outputSize[2] * 4);
        pipe->InitBuffer(Bin, sz * sizeof(T));
        pipe->InitBuffer(Boff, sz * 4);
        if constexpr (std::is_same_v<T, bfloat16_t>)
            pipe->InitBuffer(BCast, sz * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>)
        {
            int align = 32 / sizeof(T);
            uint32_t stride = (inputSize[2] * kernelSize[1] + align - 1) / align * align;
            LocalTensor<uint32_t> off = Boff.Get<uint32_t>();
            for(uint32_t i = 0, p = 0; i < kernelSize[0]; i ++)
                for(uint32_t j = 0; j < kernelSize[1]; j ++)
                    for(uint32_t k = 0; k < kernelSize[2]; k ++, p ++)
                        off.SetValue(p, (i * stride + j * inputSize[2] + k) * sizeof(T));
            uint32_t last_batch = -1, last_plane = -1;
            LocalTensor<uint32_t> poolT = BT.Get<uint32_t>();
            LocalTensor<uint32_t> poolH = BH.Get<uint32_t>();
            LocalTensor<uint32_t> poolW = BW.Get<uint32_t>();
            LocalTensor<T> in = Qin.AllocTensor<T>();
            LocalTensor<T> out = Qout.AllocTensor<T>();
            LocalTensor<T> bin = Bin.AllocTensor<T>();
            DataCopyExtParams copyParams{1, (uint32_t)(sizeof(T)), 0, 0, 0};
            DataCopyExtParams copyParams1{(uint16_t)kernelSize[0], (uint32_t)(inputSize[2] * kernelSize[1] * sizeof(T)), (uint32_t)((inputSize[1] - kernelSize[1]) * inputSize[2] * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            int count = kernelSize[0] * kernelSize[1] * kernelSize[2];
            uint32_t batch, plane, outputT, outputH, outputW = -1;
            for(uint64_t i = 0; i < processNum; i ++)
            {
                bool st = false;
                if((outputW + 1) % outputSize[2] == 0)
                {
                    uint64_t idx = i + globalBufferIndex;
                    uint64_t outputIdx = idx % ((uint64_t)outputSize[0] * outputSize[1] * outputSize[2]);
                    idx /= ((uint64_t)outputSize[0] * outputSize[1] * outputSize[2]);
                    plane = idx % c;
                    idx /= c;
                    batch = idx % n;
                    outputT = outputIdx / (outputSize[1] * outputSize[2]);
                    outputH = (outputIdx / outputSize[2]) % outputSize[1];
                    outputW = outputIdx % outputSize[2];
                    st = true;
                }
                else outputW ++;
                if(last_batch != batch || last_plane != plane)
                {
                    last_batch = batch, last_plane = plane;
                    for(uint32_t j = 0; j < outputSize[0]; j ++)
                        poolT.SetValue(j, get_intervals(x2Gm.GetValue(batch * c * 3 + plane * 3), j, inputSize[0], outputSize[0], kernelSize[0]));
                    for(uint32_t j = 0; j < outputSize[1]; j ++)
                        poolH.SetValue(j, get_intervals(x2Gm.GetValue(batch * c * 3 + plane * 3 + 1), j, inputSize[1], outputSize[1], kernelSize[1]));
                    for(uint32_t j = 0; j < outputSize[2]; j ++)
                        poolW.SetValue(j, get_intervals(x2Gm.GetValue(batch * c * 3 + plane * 3 + 2), j, inputSize[2], outputSize[2], kernelSize[2]));
                }
                if(st)
                {
                    uint64_t inputIdx = get_idx(batch, plane, poolT.GetValue(outputT), poolH.GetValue(outputH), 0, inputSize);
                    DataCopyPad(in, x1Gm[inputIdx], copyParams1, padParams);
                    Qin.EnQue(in);
                    in = Qin.DeQue<T>();
                }
                Gather(bin, in, off, poolW.GetValue(outputW) * sizeof(T), count);
                ReduceMax<T>(bin, bin, bin, count, false);
                out.SetValue(0, bin.GetValue(0));
                Qout.EnQue(out);
                out = Qout.DeQue<T>();
                DataCopyPad(yGm[i + globalBufferIndex], out, copyParams);
            }
            Qin.FreeTensor(in);
            Qout.FreeTensor(out);
        }
        else
        {
            int align = 32 / sizeof(T);
            uint32_t stride = (inputSize[2] * kernelSize[1] + align - 1) / align * align;
            LocalTensor<uint32_t> off = Boff.Get<uint32_t>();
            for(uint32_t i = 0, p = 0; i < kernelSize[0]; i ++)
                for(uint32_t j = 0; j < kernelSize[1]; j ++)
                    for(uint32_t k = 0; k < kernelSize[2]; k ++, p ++)
                        off.SetValue(p, (i * stride + j * inputSize[2] + k) * sizeof(T));
            uint32_t last_batch = -1, last_plane = -1;
            LocalTensor<uint32_t> poolT = BT.Get<uint32_t>();
            LocalTensor<uint32_t> poolH = BH.Get<uint32_t>();
            LocalTensor<uint32_t> poolW = BW.Get<uint32_t>();
            LocalTensor<T> in = Qin.AllocTensor<T>();
            LocalTensor<T> out = Qout.AllocTensor<T>();
            LocalTensor<T> bin = Bin.Get<T>();
            LocalTensor<float> in_f = BCast.Get<float>();
            DataCopyExtParams copyParams{1, (uint32_t)(sizeof(T)), 0, 0, 0};
            DataCopyExtParams copyParams1{(uint16_t)kernelSize[0], (uint32_t)(inputSize[2] * kernelSize[1] * sizeof(T)), (uint32_t)((inputSize[1] - kernelSize[1]) * inputSize[2] * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            int count = kernelSize[0] * kernelSize[1] * kernelSize[2];
            uint32_t batch, plane, outputT, outputH, outputW = -1;
            for(uint64_t i = 0; i < processNum; i ++)
            {
                bool st = false;
                if((outputW + 1) % outputSize[2] == 0)
                {
                    uint64_t idx = i + globalBufferIndex;
                    uint64_t outputIdx = idx % ((uint64_t)outputSize[0] * outputSize[1] * outputSize[2]);
                    idx /= ((uint64_t)outputSize[0] * outputSize[1] * outputSize[2]);
                    plane = idx % c;
                    idx /= c;
                    batch = idx % n;
                    outputT = outputIdx / (outputSize[1] * outputSize[2]);
                    outputH = (outputIdx / outputSize[2]) % outputSize[1];
                    outputW = outputIdx % outputSize[2];
                    st = true;
                }
                else outputW ++;
                if(last_batch != batch || last_plane != plane)
                {
                    last_batch = batch, last_plane = plane;
                    for(uint32_t j = 0; j < outputSize[0]; j ++)
                        poolT.SetValue(j, get_intervals(ToFloat(x2Gm.GetValue(batch * c * 3 + plane * 3)), j, inputSize[0], outputSize[0], kernelSize[0]));
                    for(uint32_t j = 0; j < outputSize[1]; j ++)
                        poolH.SetValue(j, get_intervals(ToFloat(x2Gm.GetValue(batch * c * 3 + plane * 3 + 1)), j, inputSize[1], outputSize[1], kernelSize[1]));
                    for(uint32_t j = 0; j < outputSize[2]; j ++)
                        poolW.SetValue(j, get_intervals(ToFloat(x2Gm.GetValue(batch * c * 3 + plane * 3 + 2)), j, inputSize[2], outputSize[2], kernelSize[2]));
                }
                if(st)
                {
                    uint64_t inputIdx = get_idx(batch, plane, poolT.GetValue(outputT), poolH.GetValue(outputH), 0, inputSize);
                    DataCopyPad(in, x1Gm[inputIdx], copyParams1, padParams);
                    Qin.EnQue(in);
                    in = Qin.DeQue<T>();
                }
                Gather(bin, in, off, poolW.GetValue(outputW) * sizeof(T), count);
                Cast(in_f, bin, RoundMode::CAST_NONE, count);
                ReduceMax<float>(in_f, in_f, in_f, count, false);
                out.SetValue(0, ToBfloat16(in_f.GetValue(0)));
                Qout.EnQue(out);
                out = Qout.DeQue<T>();
                DataCopyPad(yGm[i + globalBufferIndex], out, copyParams);
            }
            Qin.FreeTensor(in);
            Qout.FreeTensor(out);
        }
    }
    __aicore__ inline uint32_t get_intervals(float sample, int32_t index, int32_t inputSize, int32_t outputSize, int32_t kernelSize)
    {
        if(index == outputSize - 1) return inputSize - kernelSize;
        float index_f = index, inputSize_f = inputSize, outputSize_f = outputSize, kernelSize_f = kernelSize;
        if constexpr (std::is_same_v<T, float>)
        {
            float alpha = (inputSize_f - kernelSize_f) / (outputSize_f - 1);
            return ScalarCast<float, int32_t, RoundMode::CAST_FLOOR>((index_f + sample) * alpha) - ScalarCast<float, int32_t, RoundMode::CAST_FLOOR>(sample * alpha);
        }
        else if constexpr (std::is_same_v<T, half>)
        {
            LocalTensor<half> buf1 = B1.Get<half>();
            LocalTensor<half> buf2 = B2.Get<half>();
            LocalTensor<half> buf3 = B3.Get<half>();
            buf1.SetValue(0, inputSize_f - kernelSize_f);
            buf2.SetValue(0, outputSize_f - 1);
            Div(buf1, buf1, buf2, 1);
            buf2.SetValue(0, index_f);
            buf3.SetValue(0, sample);
            Add(buf2, buf2, buf3, 1);
            Mul(buf2, buf1, buf2, 1);
            Mul(buf1, buf1, buf3, 1);
            float r = buf2.GetValue(0);
            float l = buf1.GetValue(0);
            return ScalarCast<float, int32_t, RoundMode::CAST_FLOOR>(r) - ScalarCast<float, int32_t, RoundMode::CAST_FLOOR>(l);
        }
        else
        {
            float alpha = get(get(inputSize_f - kernelSize_f) / get(outputSize_f - 1));
            return ScalarCast<float, int32_t, RoundMode::CAST_FLOOR>(get(get(get(index_f) + sample) * alpha)) - ScalarCast<float, int32_t, RoundMode::CAST_FLOOR>(get(sample * alpha));
        }
        return 0;
    }
    __aicore__ inline uint64_t get_idx(uint32_t x1, uint32_t x2, uint32_t x3, uint32_t x4, uint32_t x5, uint32_t size[3])
    {
        uint32_t x[5] = {x1, x2, x3, x4, x5};
        uint32_t y[5] = {n, c, size[0], size[1], size[2]};
        uint64_t res = 0, s = 1;
        for(int i = 4; i >= 0; i --)
        {
            res += x[i] * s;
            s *= y[i];
        }
        return res;
    }
    __aicore__ inline float get(float f)
    {
        LocalTensor<float> buf1 = B1.Get<float>();
        LocalTensor<bfloat16_t> buf2 = B2.Get<bfloat16_t>();
        buf1.SetValue(0, f);
        Cast(buf2, buf1, RoundMode::CAST_RINT, 1);
        return ToFloat(buf2.GetValue(0)); 
    }
private:
    GlobalTensor<T> x1Gm, x2Gm, yGm;
    TQue<QuePosition::VECIN, 1> Qin;
    TQue<QuePosition::VECOUT, 1> Qout;
    TBuf<QuePosition::VECCALC> B1, B2, B3, BT, BH, BW, BCast, Bin, Boff;
    uint32_t n, c, kernelSize[3], inputSize[3], outputSize[3];
    uint64_t processNum, globalBufferIndex;
};

template <typename T>
class BruteForce_With_Indices
{
public:
    __aicore__ inline BruteForce_With_Indices() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR random_sample, GM_ADDR out, GM_ADDR indices, uint32_t kernelSize[3], uint32_t inputSize[3], 
                                uint32_t outputSize[3], uint32_t n, uint32_t c, uint32_t bigCoreNum, uint64_t bigCoreProcessNum, uint64_t smallCoreProcessNum, TPipe *pipe)
    {
        uint32_t coreNum = GetBlockIdx();
        this->n = n;
        this->c = c;
        for(int i = 0; i < 3; i ++)
        {
            this->kernelSize[i] = kernelSize[i];
            this->inputSize[i] = inputSize[i];
            this->outputSize[i] = outputSize[i];
        }
        this->globalBufferIndex = coreNum * bigCoreProcessNum;
        if(coreNum < bigCoreNum)
            this->processNum = bigCoreProcessNum;
        else
        {
            this->processNum = smallCoreProcessNum;
            this->globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (coreNum - bigCoreNum);
        }
        int align = 32 / sizeof(T);
        uint32_t sz = kernelSize[0] * kernelSize[1] * ((kernelSize[2] + align - 1) / align * align);
        uint32_t inputSz = kernelSize[0] * ((inputSize[2] * kernelSize[1] + align - 1) / align * align);
        x1Gm.SetGlobalBuffer((__gm__ T *)input, (uint64_t)n * c * inputSize[0] * inputSize[1] * inputSize[2]);
        x2Gm.SetGlobalBuffer((__gm__ T *)random_sample, (uint64_t)n * c * 3);
        yGm.SetGlobalBuffer((__gm__ T *)out, (uint64_t)n * c * outputSize[0] * outputSize[1] * outputSize[2]);
        indicesGm.SetGlobalBuffer((__gm__ int32_t *)indices, (uint64_t)n * c * outputSize[0] * outputSize[1] * outputSize[2]);
        pipe->InitBuffer(Qin, 1, inputSz * sizeof(T));
        pipe->InitBuffer(Qout, 1, 32);
        pipe->InitBuffer(Qout_indices, 1, 32);
        pipe->InitBuffer(B1, 32);
        pipe->InitBuffer(B2, 32);
        pipe->InitBuffer(B3, 32);
        pipe->InitBuffer(BT, outputSize[0] * 4);
        pipe->InitBuffer(BH, outputSize[1] * 4);
        pipe->InitBuffer(BW, outputSize[2] * 4);
        pipe->InitBuffer(Bin, sz * sizeof(T));
        pipe->InitBuffer(Boff, sz * 4);
        if constexpr (std::is_same_v<T, bfloat16_t>)
            pipe->InitBuffer(BCast, sz * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>)
        {
            int align = 32 / sizeof(T);
            uint32_t stride = (inputSize[2] * kernelSize[1] + align - 1) / align * align;
            LocalTensor<uint32_t> off = Boff.Get<uint32_t>();
            for(uint32_t i = 0, p = 0; i < kernelSize[0]; i ++)
                for(uint32_t j = 0; j < kernelSize[1]; j ++)
                    for(uint32_t k = 0; k < kernelSize[2]; k ++, p ++)
                        off.SetValue(p, (i * stride + j * inputSize[2] + k) * sizeof(T));
            uint32_t last_batch = -1, last_plane = -1;
            LocalTensor<uint32_t> poolT = BT.Get<uint32_t>();
            LocalTensor<uint32_t> poolH = BH.Get<uint32_t>();
            LocalTensor<uint32_t> poolW = BW.Get<uint32_t>();
            LocalTensor<T> in = Qin.AllocTensor<T>();
            LocalTensor<T> out = Qout.AllocTensor<T>();
            LocalTensor<int32_t> indices = Qout_indices.AllocTensor<int32_t>();
            LocalTensor<T> bin = Bin.AllocTensor<T>();
            DataCopyExtParams copyParams1{(uint16_t)kernelSize[0], (uint32_t)(inputSize[2] * kernelSize[1] * sizeof(T)), (uint32_t)((inputSize[1] - kernelSize[1]) * inputSize[2] * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            int count = kernelSize[0] * kernelSize[1] * kernelSize[2];
            uint32_t batch, plane, outputT, outputH, outputW = -1;
            for(uint64_t i = 0; i < processNum; i ++)
            {
                bool st = false;
                if((outputW + 1) % outputSize[2] == 0)
                {
                    uint64_t idx = i + globalBufferIndex;
                    uint64_t outputIdx = idx % ((uint64_t)outputSize[0] * outputSize[1] * outputSize[2]);
                    idx /= ((uint64_t)outputSize[0] * outputSize[1] * outputSize[2]);
                    plane = idx % c;
                    idx /= c;
                    batch = idx % n;
                    outputT = outputIdx / (outputSize[1] * outputSize[2]);
                    outputH = (outputIdx / outputSize[2]) % outputSize[1];
                    outputW = outputIdx % outputSize[2];
                    st = true;
                }
                else outputW ++;
                if(last_batch != batch || last_plane != plane)
                {
                    last_batch = batch, last_plane = plane;
                    for(uint32_t j = 0; j < outputSize[0]; j ++)
                        poolT.SetValue(j, get_intervals(x2Gm.GetValue(batch * c * 3 + plane * 3), j, inputSize[0], outputSize[0], kernelSize[0]));
                    for(uint32_t j = 0; j < outputSize[1]; j ++)
                        poolH.SetValue(j, get_intervals(x2Gm.GetValue(batch * c * 3 + plane * 3 + 1), j, inputSize[1], outputSize[1], kernelSize[1]));
                    for(uint32_t j = 0; j < outputSize[2]; j ++)
                        poolW.SetValue(j, get_intervals(x2Gm.GetValue(batch * c * 3 + plane * 3 + 2), j, inputSize[2], outputSize[2], kernelSize[2]));
                }
                if(st)
                {
                    uint64_t inputIdx = get_idx(batch, plane, poolT.GetValue(outputT), poolH.GetValue(outputH), 0, inputSize);
                    DataCopyPad(in, x1Gm[inputIdx], copyParams1, padParams);
                    Qin.EnQue(in);
                    in = Qin.DeQue<T>();
                }
                Gather(bin, in, off, poolW.GetValue(outputW) * sizeof(T), count);
                ReduceMax<T>(bin, bin, bin, count, true);
                T maxIndex = bin.GetValue(1);
                if(sizeof(T) == 2)
                {
                    uint16_t realIndex = *reinterpret_cast<uint16_t*>(&maxIndex);
                    uint32_t t, h, w;
                    w = (realIndex % kernelSize[2]) + poolW.GetValue(outputW);
                    realIndex /= kernelSize[2];
                    h = (realIndex % kernelSize[1]) + poolH.GetValue(outputH);
                    realIndex /= kernelSize[1];
                    t = (realIndex % kernelSize[0]) + poolT.GetValue(outputT);
                    indices.SetValue(0, t * inputSize[1] * inputSize[2] + h * inputSize[2] + w);
                }
                else
                {
                    uint32_t realIndex = *reinterpret_cast<uint32_t*>(&maxIndex);
                    uint32_t t, h, w;
                    w = (realIndex % kernelSize[2]) + poolW.GetValue(outputW);
                    realIndex /= kernelSize[2];
                    h = (realIndex % kernelSize[1]) + poolH.GetValue(outputH);
                    realIndex /= kernelSize[1];
                    t = (realIndex % kernelSize[0]) + poolT.GetValue(outputT);
                    indices.SetValue(0, t * inputSize[1] * inputSize[2] + h * inputSize[2] + w);
                }
                out.SetValue(0, bin.GetValue(0));
                Qout.EnQue(out);
                Qout_indices.EnQue(indices);
                out = Qout.DeQue<T>();
                indices = Qout_indices.DeQue<int32_t>();
                DataCopyExtParams copyParams{1, (uint32_t)(sizeof(T)), 0, 0, 0};
                DataCopyPad(yGm[i + globalBufferIndex], out, copyParams);
                copyParams.blockLen = (uint32_t)(sizeof(int32_t));
                DataCopyPad(indicesGm[i + globalBufferIndex], indices, copyParams);
            }
            Qin.FreeTensor(in);
            Qout.FreeTensor(out);
            Qout_indices.FreeTensor(indices);
        }
        else
        {
            int align = 32 / sizeof(T);
            uint32_t stride = (inputSize[2] * kernelSize[1] + align - 1) / align * align;
            LocalTensor<uint32_t> off = Boff.Get<uint32_t>();
            for(uint32_t i = 0, p = 0; i < kernelSize[0]; i ++)
                for(uint32_t j = 0; j < kernelSize[1]; j ++)
                    for(uint32_t k = 0; k < kernelSize[2]; k ++, p ++)
                        off.SetValue(p, (i * stride + j * inputSize[2] + k) * sizeof(T));
            uint32_t last_batch = -1, last_plane = -1;
            LocalTensor<uint32_t> poolT = BT.Get<uint32_t>();
            LocalTensor<uint32_t> poolH = BH.Get<uint32_t>();
            LocalTensor<uint32_t> poolW = BW.Get<uint32_t>();
            LocalTensor<T> in = Qin.AllocTensor<T>();
            LocalTensor<T> out = Qout.AllocTensor<T>();
            LocalTensor<int32_t> indices = Qout_indices.AllocTensor<int32_t>();
            LocalTensor<T> bin = Bin.Get<T>();
            LocalTensor<float> in_f = BCast.Get<float>();
            DataCopyExtParams copyParams1{(uint16_t)kernelSize[0], (uint32_t)(inputSize[2] * kernelSize[1] * sizeof(T)), (uint32_t)((inputSize[1] - kernelSize[1]) * inputSize[2] * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            int count = kernelSize[0] * kernelSize[1] * kernelSize[2];
            uint32_t batch, plane, outputT, outputH, outputW = -1;
            for(uint64_t i = 0; i < processNum; i ++)
            {
                bool st = false;
                if((outputW + 1) % outputSize[2] == 0)
                {
                    uint64_t idx = i + globalBufferIndex;
                    uint64_t outputIdx = idx % ((uint64_t)outputSize[0] * outputSize[1] * outputSize[2]);
                    idx /= ((uint64_t)outputSize[0] * outputSize[1] * outputSize[2]);
                    plane = idx % c;
                    idx /= c;
                    batch = idx % n;
                    outputT = outputIdx / (outputSize[1] * outputSize[2]);
                    outputH = (outputIdx / outputSize[2]) % outputSize[1];
                    outputW = outputIdx % outputSize[2];
                    st = true;
                }
                else outputW ++;
                if(last_batch != batch || last_plane != plane)
                {
                    last_batch = batch, last_plane = plane;
                    for(uint32_t j = 0; j < outputSize[0]; j ++)
                        poolT.SetValue(j, get_intervals(ToFloat(x2Gm.GetValue(batch * c * 3 + plane * 3)), j, inputSize[0], outputSize[0], kernelSize[0]));
                    for(uint32_t j = 0; j < outputSize[1]; j ++)
                        poolH.SetValue(j, get_intervals(ToFloat(x2Gm.GetValue(batch * c * 3 + plane * 3 + 1)), j, inputSize[1], outputSize[1], kernelSize[1]));
                    for(uint32_t j = 0; j < outputSize[2]; j ++)
                        poolW.SetValue(j, get_intervals(ToFloat(x2Gm.GetValue(batch * c * 3 + plane * 3 + 2)), j, inputSize[2], outputSize[2], kernelSize[2]));
                }
                if(st)
                {
                    uint64_t inputIdx = get_idx(batch, plane, poolT.GetValue(outputT), poolH.GetValue(outputH), 0, inputSize);
                    DataCopyPad(in, x1Gm[inputIdx], copyParams1, padParams);
                    Qin.EnQue(in);
                    in = Qin.DeQue<T>();
                }
                Gather(bin, in, off, poolW.GetValue(outputW) * sizeof(T), count);
                Cast(in_f, bin, RoundMode::CAST_NONE, count);
                ReduceMax<float>(in_f, in_f, in_f, count, true);
                float maxIndex = in_f.GetValue(1);
                uint32_t realIndex = *reinterpret_cast<uint32_t*>(&maxIndex);
                uint32_t t, h, w;
                w = (realIndex % kernelSize[2]) + poolW.GetValue(outputW);
                realIndex /= kernelSize[2];
                h = (realIndex % kernelSize[1]) + poolH.GetValue(outputH);
                realIndex /= kernelSize[1];
                t = (realIndex % kernelSize[0]) + poolT.GetValue(outputT);
                indices.SetValue(0, t * inputSize[1] * inputSize[2] + h * inputSize[2] + w);
                out.SetValue(0, ToBfloat16(in_f.GetValue(0)));
                Qout.EnQue(out);
                Qout_indices.EnQue(indices);
                out = Qout.DeQue<T>();
                indices = Qout_indices.DeQue<int32_t>();
                DataCopyExtParams copyParams{1, (uint32_t)(sizeof(T)), 0, 0, 0};
                DataCopyPad(yGm[i + globalBufferIndex], out, copyParams);
                copyParams.blockLen = (uint32_t)(sizeof(int32_t));
                DataCopyPad(indicesGm[i + globalBufferIndex], indices, copyParams);
            }
            Qin.FreeTensor(in);
            Qout.FreeTensor(out);
            Qout_indices.FreeTensor(indices);
        }
    }
    __aicore__ inline uint32_t get_intervals(float sample, int32_t index, int32_t inputSize, int32_t outputSize, int32_t kernelSize)
    {
        float index_f = index, inputSize_f = inputSize, outputSize_f = outputSize, kernelSize_f = kernelSize;
        if constexpr (std::is_same_v<T, float>)
        {
            float alpha = (inputSize_f - kernelSize_f) / (outputSize_f - 1);
            if(index == outputSize - 1) return inputSize - kernelSize;
            else return ScalarCast<float, int32_t, RoundMode::CAST_FLOOR>((index_f + sample) * alpha) - ScalarCast<float, int32_t, RoundMode::CAST_FLOOR>(sample * alpha);
        }
        else if constexpr (std::is_same_v<T, half>)
        {
            LocalTensor<half> buf1 = B1.Get<half>();
            LocalTensor<half> buf2 = B2.Get<half>();
            LocalTensor<half> buf3 = B3.Get<half>();
            buf1.SetValue(0, inputSize_f - kernelSize_f);
            buf2.SetValue(0, outputSize_f - 1);
            Div(buf1, buf1, buf2, 1);
            if(index == outputSize - 1) return inputSize - kernelSize;
            else 
            {
                buf2.SetValue(0, index_f);
                buf3.SetValue(0, sample);
                Add(buf2, buf2, buf3, 1);
                Mul(buf2, buf1, buf2, 1);
                Mul(buf1, buf1, buf3, 1);
                float r = buf2.GetValue(0);
                float l = buf1.GetValue(0);
                return ScalarCast<float, int32_t, RoundMode::CAST_FLOOR>(r) - ScalarCast<float, int32_t, RoundMode::CAST_FLOOR>(l);
            }
        }
        else
        {
            float alpha = get(get(inputSize_f - kernelSize_f) / get(outputSize_f - 1));
            if(index == outputSize - 1) return inputSize - kernelSize;
            else return ScalarCast<float, int32_t, RoundMode::CAST_FLOOR>(get(get(get(index_f) + sample) * alpha)) - ScalarCast<float, int32_t, RoundMode::CAST_FLOOR>(get(sample * alpha));
        }
    }
    __aicore__ inline uint64_t get_idx(uint32_t x1, uint32_t x2, uint32_t x3, uint32_t x4, uint32_t x5, uint32_t size[3])
    {
        uint32_t x[5] = {x1, x2, x3, x4, x5};
        uint32_t y[5] = {n, c, size[0], size[1], size[2]};
        uint64_t res = 0, s = 1;
        for(int i = 4; i >= 0; i --)
        {
            res += x[i] * s;
            s *= y[i];
        }
        return res;
    }
    __aicore__ inline float get(float f)
    {
        LocalTensor<float> buf1 = B1.Get<float>();
        LocalTensor<bfloat16_t> buf2 = B2.Get<bfloat16_t>();
        buf1.SetValue(0, f);
        Cast(buf2, buf1, RoundMode::CAST_RINT, 1);
        return ToFloat(buf2.GetValue(0)); 
    }
private:
    GlobalTensor<T> x1Gm, x2Gm, yGm;
    GlobalTensor<int32_t> indicesGm;
    TQue<QuePosition::VECIN, 1> Qin;
    TQue<QuePosition::VECOUT, 1> Qout, Qout_indices;
    TBuf<QuePosition::VECCALC> B1, B2, B3, BT, BH, BW, BCast, Bin, Boff;
    uint32_t n, c, kernelSize[3], inputSize[3], outputSize[3];
    uint64_t processNum, globalBufferIndex;
};

extern "C" __global__ __aicore__ void fractional_max_pool3_d(GM_ADDR input, GM_ADDR random_sample, GM_ADDR out, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if(TILING_KEY_IS(1))
    {
        BruteForce<DTYPE_INPUT> op;
        op.Init(input, random_sample, out, tiling_data.kernelSize, tiling_data.inputSize, tiling_data.outputSize, tiling_data.n,
                tiling_data.c, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
    else if(TILING_KEY_IS(2))
    {
        BruteForce_With_Indices<DTYPE_INPUT> op;
        op.Init(input, random_sample, out, indices, tiling_data.kernelSize, tiling_data.inputSize, tiling_data.outputSize, tiling_data.n,
                tiling_data.c, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
}