#include "kernel_operator.h"

using namespace AscendC;
const uint32_t BUFFER_NUM = 1;

class KernelRmsNorm0 { // float32
    public:
        __aicore__ inline KernelRmsNorm0() {}
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR rstd, 
                                    float epsilon, uint32_t x1TotalLength, 
                                    uint32_t x2TotalLength, uint32_t batchNum, 
                                    uint32_t batchLength, uint32_t tileNum, 
                                    uint32_t tileLength, uint32_t lastTileLength,
                                    uint32_t rstdTileNum, uint32_t rstdTileLength, 
                                    uint32_t rstdLastTileLength, uint32_t resLength) 
        {   
            this->epsilon = epsilon;
            this->x1TotalLength = x1TotalLength;
            this->x2TotalLength = x2TotalLength;
            this->batchNum = batchNum;
            this->batchLength = batchLength;
            this->tileNum = tileNum;
            this->tileLength = tileLength;
            this->lastTileLength = lastTileLength;
            this->rstdTileNum = rstdTileNum;
            this->rstdTileLength = rstdTileLength;
            this->rstdLastTileLength = rstdLastTileLength;
            this->resLength = resLength;
    
            this->x1Gm.SetGlobalBuffer((__gm__ float *)x1, this->x1TotalLength);
            this->x2Gm.SetGlobalBuffer((__gm__ float *)x2, this->x2TotalLength);
            this->yGm.SetGlobalBuffer((__gm__ float *)y, this->x1TotalLength);
            this->rstdGm.SetGlobalBuffer((__gm__ float *)rstd, this->batchNum);

            pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileLength * sizeof(float));
            pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileLength * sizeof(float));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
            pipe.InitBuffer(outQueueRstd, BUFFER_NUM, this->rstdTileLength * sizeof(float));
            pipe.InitBuffer(buffer1, this->rstdTileLength * sizeof(float));
            this->OneTensor = buffer1.Get<float>();
            Duplicate(this->OneTensor, 1.0f, this->rstdTileLength);
        }

        __aicore__ inline void Process() {
            ComputeRstd();
            ComputeY();
        }
    
    private:
        __aicore__ inline void ComputeRstd() {
            LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
            uint32_t loopCount = this->tileNum * BUFFER_NUM;
            uint32_t offset = 0, rstdOffset = 0;
            float temp = 1.0f;
            temp = temp / this->batchLength;
            for(uint32_t i = 0; i < this->rstdTileNum - 1; ++i){
                // compute rstd
                for(uint32_t j = 0; j < this->rstdTileLength; ++j){
                    // compute sum
                    this->batchSumFloat = 0;
                    for (uint32_t k = 0; k < loopCount; ++k){
                        if(k == loopCount - 1){
                            CopyX1In(k, this->lastTileLength, offset);
                            ComputeLastSum(k, this->lastTileLength);
                        }else{
                            CopyX1In(k, this->tileLength, offset);
                            ComputeSum(k, this->tileLength);
                        }
                    }
                    rstdLocal.SetValue(j, this->batchSumFloat);
                    offset += this->batchLength;
                }
                Muls(rstdLocal, rstdLocal, temp, this->rstdTileLength);
                Adds(rstdLocal, rstdLocal, this->epsilon, this->rstdTileLength);
                Div(rstdLocal, OneTensor, rstdLocal, this->rstdTileLength);
                Sqrt(rstdLocal, rstdLocal, this->rstdTileLength);
                outQueueRstd.EnQue<float>(rstdLocal);
                rstdLocal = outQueueRstd.DeQue<float>();
                DataCopy(rstdGm[rstdOffset], rstdLocal[0], this->rstdTileLength);
                rstdOffset += this->rstdTileLength;
            }
            // compute tail rstd
            for(uint32_t j = 0; j < this->rstdLastTileLength; ++j){
                // compute sum
                this->batchSumFloat = 0;
                for (uint32_t k = 0; k < loopCount; ++k){
                    if(k == loopCount - 1){
                        CopyX1In(k, this->lastTileLength, offset);
                        ComputeLastSum(k, this->lastTileLength);
                    }else{
                        CopyX1In(k, this->tileLength, offset);
                        ComputeSum(k, this->tileLength);
                    }
                }
                rstdLocal.SetValue(j, this->batchSumFloat);
                offset += this->batchLength;
            }
            Muls(rstdLocal, rstdLocal, temp, this->rstdTileLength);
            Adds(rstdLocal, rstdLocal, this->epsilon, this->rstdTileLength);
            Div(rstdLocal, OneTensor, rstdLocal, this->rstdTileLength);
            Sqrt(rstdLocal, rstdLocal, this->rstdTileLength);
            outQueueRstd.EnQue<float>(rstdLocal);
            rstdLocal = outQueueRstd.DeQue<float>();
            DataCopy(rstdGm[rstdOffset], rstdLocal[0], this->rstdTileLength);
            rstdOffset += this->rstdTileLength;
        }

        __aicore__ inline void CopyX1In(uint32_t progress, uint32_t length, uint32_t offset) {
            LocalTensor<float> x1Local = inQueueX1.AllocTensor<float>();

            DataCopy(x1Local[0], x1Gm[offset + progress * this->tileLength], length);

            inQueueX1.EnQue(x1Local);
        }

        __aicore__ inline void ComputeSum(uint32_t progress, uint32_t length) {
            LocalTensor<float> x1Local = inQueueX1.DeQue<float>();
            LocalTensor<float> tempLocal = outQueueY.AllocTensor<float>();

            Mul(x1Local, x1Local, x1Local, length);
            ReduceSum(x1Local, x1Local, tempLocal, length);
            this->batchSumFloat += (float)x1Local.GetValue(0);

            inQueueX1.FreeTensor(x1Local);
            outQueueY.FreeTensor(tempLocal);
        }

        __aicore__ inline void ComputeLastSum(uint32_t progress, uint32_t length) {
            LocalTensor<float> x1Local = inQueueX1.DeQue<float>();
            LocalTensor<float> tempLocal = outQueueY.AllocTensor<float>();

            const float ZERO = 0.0f;
            for(uint32_t k = 1; k <= this->resLength; ++k){
                x1Local.SetValue(length - k, ZERO);
            }
            Mul(x1Local, x1Local, x1Local, length);
            ReduceSum(x1Local, x1Local, tempLocal, length);
            this->batchSumFloat += (float)x1Local.GetValue(0);

            inQueueX1.FreeTensor(x1Local);
            outQueueY.FreeTensor(tempLocal);
        }

        __aicore__ inline void ComputeY() {
            uint32_t loopCount = this->tileNum * BUFFER_NUM;
            uint32_t copyOffset = 0;

            for(uint32_t i = 0; i < this->batchNum; ++i){
                this->rstd = rstdGm.GetValue(i);
                for (uint32_t j = 0; j < loopCount; ++j){
                    if(j == loopCount - 1){
                        CopyX1X2In(j, this->lastTileLength, copyOffset);
                        ComputeMul(j, this->lastTileLength);
                        CopyOut(j, this->lastTileLength, copyOffset);
                    }else{
                        CopyX1X2In(j, this->tileLength, copyOffset);
                        ComputeMul(j, this->tileLength);
                        CopyOut(j, this->tileLength, copyOffset);
                    }
                }
                copyOffset += this->batchLength;
            }
        }

        __aicore__ inline void CopyX1X2In(uint32_t progress, uint32_t length, uint32_t offset) {
            LocalTensor<float> x1Local = inQueueX1.AllocTensor<float>();
            LocalTensor<float> x2Local = inQueueX2.AllocTensor<float>();

            DataCopy(x1Local[0], x1Gm[offset + progress * this->tileLength], length);
            DataCopy(x2Local[0], x2Gm[progress * this->tileLength], length);

            inQueueX1.EnQue(x1Local);
            inQueueX2.EnQue(x2Local);
        }

        __aicore__ inline void ComputeMul(uint32_t progress, uint32_t length) {
            LocalTensor<float> x1Local = inQueueX1.DeQue<float>();
            LocalTensor<float> x2Local = inQueueX2.DeQue<float>();
            LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

            Muls(x1Local, x1Local, this->rstd, length);
            Mul(yLocal, x1Local, x2Local, length);

            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
            outQueueY.EnQue(yLocal);
        }
    
        __aicore__ inline void CopyOut(uint32_t progress, uint32_t length, uint32_t offset) {
            LocalTensor<float> yLocal = outQueueY.DeQue<float>();
            
            if(this->resLength != 0){
                float t;
                for(uint32_t i = 0; i < length; ++i){
                    t = yLocal.GetValue(i);
                    yGm.SetValue(offset + progress * this->tileLength + i, t);
                }
            }else{
                DataCopy(yGm[offset + progress * this->tileLength], yLocal[0], length);
            }
    
            outQueueY.FreeTensor(yLocal);
        }
    
    private:
        TPipe pipe;
        TBuf<TPosition::VECCALC> buffer1;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY, outQueueRstd;
        LocalTensor<float> OneTensor;
        GlobalTensor<float> x1Gm, x2Gm, yGm, rstdGm;
        float rstd;
        float epsilon, batchSumFloat;
        uint32_t batchNum, batchLength, x1TotalLength, x2TotalLength;
        uint32_t tileNum, tileLength, lastTileLength, resLength;
        uint32_t rstdTileNum, rstdTileLength, rstdLastTileLength;
};

class KernelRmsNorm1 { // float16
    public:
        __aicore__ inline KernelRmsNorm1() {}
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR rstd, 
                                    float epsilon, uint32_t x1TotalLength, uint32_t x2TotalLength,
                                    uint32_t batchNum, uint32_t batchLength, 
                                    uint32_t tileNum, uint32_t tileLength, 
                                    uint32_t lastTileLength, uint32_t rstdTileNum, 
                                    uint32_t rstdTileLength, uint32_t rstdLastTileLength, uint32_t resLength, uint32_t mulLastTileLength) 
        {   
            this->epsilon = epsilon;
            this->x1TotalLength = x1TotalLength;
            this->x2TotalLength = x2TotalLength;
            this->batchNum = batchNum;
            this->batchLength = batchLength;
            this->tileNum = tileNum;
            this->tileLength = tileLength;
            this->lastTileLength = lastTileLength;
            this->rstdTileNum = rstdTileNum;
            this->rstdTileLength = rstdTileLength;
            this->rstdLastTileLength = rstdLastTileLength;
            this->resLength = resLength;
            this->mulLastTileLength = mulLastTileLength;
            this->factor = 1.0f / this->batchLength;

            this->x1Gm.SetGlobalBuffer((__gm__ half *)x1, this->x1TotalLength);
            this->x2Gm.SetGlobalBuffer((__gm__ half *)x2, this->x2TotalLength);
            this->yGm.SetGlobalBuffer((__gm__ half *)y, this->x1TotalLength);
            this->rstdGm.SetGlobalBuffer((__gm__ half *)rstd, this->batchNum);

            pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileLength * sizeof(half));
            pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileLength * sizeof(half));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
            pipe.InitBuffer(outQueueRstd, BUFFER_NUM, this->rstdTileLength * sizeof(half));

            pipe.InitBuffer(buffer1, this->rstdTileLength * sizeof(half));
            pipe.InitBuffer(buffer2, this->tileLength * sizeof(float));
            pipe.InitBuffer(buffer3, this->tileLength * sizeof(float));
            pipe.InitBuffer(buffer4, this->rstdTileLength * sizeof(float));

            this->OneTensor = buffer1.Get<half>();
            Duplicate(this->OneTensor, (half)1.0f, this->rstdTileLength);
            this->tempTensor1 = buffer2.Get<float>();
            this->tempTensor2 = buffer3.Get<float>();
            this->rstdFloatTensor = buffer4.Get<float>();
        }

        __aicore__ inline void Process() {
            ComputeRstd();
            ComputeY();
        }
    
    private:
        __aicore__ inline void ComputeRstd() {
            LocalTensor<half> rstdLocal = outQueueRstd.AllocTensor<half>();
            uint32_t loopCount = this->tileNum * BUFFER_NUM;
            uint32_t offset = 0, rstdOffset = 0, copyOffset = 0;
            
            for(uint32_t i = 0; i < this->rstdTileNum - 1; ++i){
                // compute rstd
                for(uint32_t j = 0; j < this->rstdTileLength; ++j){
                    // compute sum
                    this->batchSumFloat = 0.0f;
                    this->batchSumHalf = 0.0f;
                    for (uint32_t k = 0; k < loopCount; ++k){
                        if(k == loopCount - 1){
                            CopyX1In(k, this->lastTileLength, offset);
                            ComputeLastSum(k, this->lastTileLength);
                        }else{
                            CopyX1In(k, this->tileLength, offset);
                            ComputeSum(k, this->tileLength);
                        }
                    }
                    this->rstdFloatTensor.SetValue(j, this->batchSumFloat);
                    // this->rstdFloatTensor.SetValue(j, (float)this->batchSumHalf);
                    offset += this->batchLength;
                }
                Muls(this->rstdFloatTensor, this->rstdFloatTensor, this->factor, this->rstdTileLength);
                Cast(rstdLocal, this->rstdFloatTensor, RoundMode::CAST_ROUND, this->rstdTileLength);
                Adds(rstdLocal, rstdLocal, (half)this->epsilon, this->rstdTileLength);
                Sqrt(rstdLocal, rstdLocal, this->rstdTileLength);
                Div(rstdLocal, this->OneTensor, rstdLocal, this->rstdTileLength);
                outQueueRstd.EnQue<half>(rstdLocal);
                rstdLocal = outQueueRstd.DeQue<half>();
                DataCopy(rstdGm[rstdOffset], rstdLocal[0], this->rstdTileLength);
                rstdOffset += this->rstdTileLength;
            }

            // compute tail rstd
            for(uint32_t j = 0; j < this->rstdLastTileLength; ++j){
                // compute sum
                this->batchSumFloat = 0.0f;
                this->batchSumHalf = 0.0f;
                for (uint32_t k = 0; k < loopCount; ++k){
                    if(k == loopCount - 1){
                        CopyX1In(k, this->lastTileLength, offset);
                        ComputeLastSum(k, this->lastTileLength);
                    }else{
                        CopyX1In(k, this->tileLength, offset);
                        ComputeSum(k, this->tileLength);
                    }
                }
                // this->rstdFloatTensor.SetValue(j, this->batchSumFloat);
                this->rstdFloatTensor.SetValue(j, this->batchSumFloat);
                offset += this->batchLength;
            }
            Muls(this->rstdFloatTensor, this->rstdFloatTensor, this->factor, this->rstdTileLength);
            Cast(rstdLocal, this->rstdFloatTensor, RoundMode::CAST_ROUND, this->rstdTileLength);
            Adds(rstdLocal, rstdLocal, (half)this->epsilon, this->rstdTileLength);
            Sqrt(rstdLocal, rstdLocal, this->rstdTileLength);
            Div(rstdLocal, this->OneTensor, rstdLocal, this->rstdTileLength);
            outQueueRstd.EnQue<half>(rstdLocal);
            rstdLocal = outQueueRstd.DeQue<half>();
            DataCopy(rstdGm[rstdOffset], rstdLocal[0], this->rstdTileLength);

            outQueueRstd.FreeTensor(rstdLocal);
        }

        __aicore__ inline void CopyX1In(uint32_t progress, uint32_t length, uint32_t offset) {
            LocalTensor<half> x1Local = inQueueX1.AllocTensor<half>();

            DataCopy(x1Local[0], x1Gm[offset + progress * this->tileLength], length);

            inQueueX1.EnQue(x1Local);
        }

        __aicore__ inline void ComputeSum(uint32_t progress, uint32_t length) {
            LocalTensor<half> x1Local = inQueueX1.DeQue<half>();

            Cast(this->tempTensor1, x1Local, RoundMode::CAST_NONE, length);
            Mul(this->tempTensor1, this->tempTensor1, this->tempTensor1, length);
            Cast(x1Local, this->tempTensor1, RoundMode::CAST_NONE, length);

            // Cast(this->tempTensor1, x1Local, RoundMode::CAST_NONE, length);
            // ReduceSum(this->tempTensor1, this->tempTensor1, this->tempTensor2, length);
            // this->batchSumFloat += this->tempTensor1.GetValue(0);

            for(uint32_t i = 0; i < length; ++i){
                this->batchSumFloat += (float)x1Local.GetValue(i);
            }

            inQueueX1.FreeTensor(x1Local);
        }

        __aicore__ inline void ComputeLastSum(uint32_t progress, uint32_t length) {
            LocalTensor<half> x1Local = inQueueX1.DeQue<half>();

            const half ZERO = 0.0f;
            for(uint32_t ki = 1; ki <= this->resLength; ++ki){
                x1Local.SetValue(length - ki, ZERO);
            }
            
            Cast(this->tempTensor1, x1Local, RoundMode::CAST_NONE, length);
            Mul(this->tempTensor1, this->tempTensor1, this->tempTensor1, length);
            Cast(x1Local, this->tempTensor1, RoundMode::CAST_NONE, length);

            // Cast(this->tempTensor1, x1Local, RoundMode::CAST_NONE, length);
            // ReduceSum(this->tempTensor1, this->tempTensor1, this->tempTensor2, length);
            // this->batchSumFloat += this->tempTensor1.GetValue(0);

            for(uint32_t i = 0; i < length - this->resLength; ++i){
                this->batchSumFloat += (float)x1Local.GetValue(i);
            }

            inQueueX1.FreeTensor(x1Local);
        }

        __aicore__ inline void ComputeY() {
            uint32_t loopCount = this->tileNum * BUFFER_NUM;
            uint32_t copyOffset = 0;

            for(uint32_t i = 0; i < this->batchNum; ++i){
                this->rstd = rstdGm.GetValue(i);
                for (uint32_t j = 0; j < loopCount; ++j){
                    if(j == loopCount - 1){
                        CopyX1X2In(j, this->lastTileLength, copyOffset);
                        ComputeMul(j, this->lastTileLength);
                        CopyOut(j, this->lastTileLength, copyOffset);
                    }else{
                        CopyX1X2In(j, this->tileLength, copyOffset);
                        ComputeMul(j, this->tileLength);
                        CopyOut(j, this->tileLength, copyOffset);
                    }
                }
                copyOffset += this->batchLength;
            }
        }

        __aicore__ inline void CopyX1X2In(uint32_t progress, uint32_t length, uint32_t offset) {
            LocalTensor<half> x1Local = inQueueX1.AllocTensor<half>();
            LocalTensor<half> x2Local = inQueueX2.AllocTensor<half>();

            DataCopy(x1Local[0], x1Gm[offset + progress * this->tileLength], length);
            DataCopy(x2Local[0], x2Gm[progress * this->tileLength], length);

            inQueueX1.EnQue(x1Local);
            inQueueX2.EnQue(x2Local);
        }

        __aicore__ inline void ComputeMul(uint32_t progress, uint32_t length) {
            LocalTensor<half> x1Local = inQueueX1.DeQue<half>();
            LocalTensor<half> x2Local = inQueueX2.DeQue<half>();
            LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();

            Muls(x1Local[0], x1Local[0], this->rstd, length);
            Mul(yLocal[0], x1Local[0], x2Local[0], length);
            
            outQueueY.EnQue(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }
    
        __aicore__ inline void CopyOut(uint32_t progress, uint32_t length, uint32_t offset) {
            LocalTensor<half> yLocal = outQueueY.DeQue<half>();

            if(this->resLength != 0){
                half t;
                for(uint32_t i = 0; i < length; ++i){
                    t = yLocal.GetValue(i);
                    yGm.SetValue(offset + progress * this->tileLength + i, t);
                }
            }else{
                DataCopy(yGm[offset + progress * this->tileLength], yLocal[0], length);
            }

            outQueueY.FreeTensor(yLocal);
        }
    
    private:
        TPipe pipe;
        TBuf<TPosition::VECCALC> buffer1, buffer2, buffer3, buffer4;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY, outQueueRstd;
        LocalTensor<half> OneTensor;
        LocalTensor<float> tempTensor1, tempTensor2, rstdFloatTensor;
        GlobalTensor<half> x1Gm, x2Gm, yGm, rstdGm;
        half rstd, batchSumHalf;
        float epsilon, batchSumFloat, factor;
        uint32_t batchNum, batchLength, x1TotalLength, x2TotalLength;
        uint32_t tileNum, tileLength, lastTileLength, resLength, mulLastTileLength;
        uint32_t rstdTileNum, rstdTileLength, rstdLastTileLength;
};

class KernelRmsNorm2 { // bfloat16
    public:
        __aicore__ inline KernelRmsNorm2() {}
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR rstd, 
                                    float epsilon, 
                                    uint32_t x1TotalLength, uint32_t x2TotalLength,
                                    uint32_t batchNum, uint32_t batchLength, 
                                    uint32_t tileNum, uint32_t tileLength, uint32_t lastTileLength,
                                    uint32_t rstdTileNum, uint32_t rstdTileLength, 
                                    uint32_t rstdLastTileLength, uint32_t resLength) 
        {   
            this->epsilon = epsilon;
            this->x1TotalLength = x1TotalLength;
            this->x2TotalLength = x2TotalLength;
            this->batchNum = batchNum;
            this->batchLength = batchLength;
            this->tileNum = tileNum;
            this->tileLength = tileLength;
            this->lastTileLength = lastTileLength;
            this->rstdTileNum = rstdTileNum;
            this->rstdTileLength = rstdTileLength;
            this->rstdLastTileLength = rstdLastTileLength;
            this->resLength = resLength;
            this->factor = 1.0f / batchLength;
    
            this->x1Gm.SetGlobalBuffer((__gm__ bfloat16_t *)x1, this->x1TotalLength);
            this->x2Gm.SetGlobalBuffer((__gm__ bfloat16_t *)x2, this->x2TotalLength);
            this->yGm.SetGlobalBuffer((__gm__ bfloat16_t *)y, this->x1TotalLength);
            this->rstdGm.SetGlobalBuffer((__gm__ bfloat16_t *)rstd, this->batchNum);

            pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileLength * sizeof(bfloat16_t));
            pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileLength * sizeof(bfloat16_t));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(bfloat16_t));
            pipe.InitBuffer(outQueueRstd, BUFFER_NUM, this->rstdTileLength * sizeof(bfloat16_t));

            pipe.InitBuffer(buffer1, this->rstdTileLength * sizeof(float));
            pipe.InitBuffer(buffer2, this->tileLength * sizeof(float));
            pipe.InitBuffer(buffer3, this->tileLength * sizeof(float));
            pipe.InitBuffer(buffer4, this->rstdTileLength * sizeof(float));
            pipe.InitBuffer(buffer5, 16 * sizeof(bfloat16_t));

            this->OneTensor = buffer1.Get<float>();
            Duplicate(this->OneTensor, 1.0f, this->rstdTileLength);
            this->tempTensor1 = buffer2.Get<float>();
            this->tempTensor2 = buffer3.Get<float>();
            this->floatRstdTensor = buffer4.Get<float>();
            this->tempBFTensor = buffer5.Get<bfloat16_t>();
        }

        __aicore__ inline void Process() {
            ComputeRstd();
            ComputeY();
        }
    
    private:
        __aicore__ inline void ComputeRstd() {
            LocalTensor<bfloat16_t> rstdLocal = outQueueRstd.AllocTensor<bfloat16_t>();
            uint32_t loopCount = this->tileNum * BUFFER_NUM;
            uint32_t offset = 0, rstdOffset = 0, copyOffset = 0;
            for(uint32_t i = 0; i < this->rstdTileNum - 1; ++i){
                for(uint32_t j = 0; j < this->rstdTileLength; ++j){
                    // compute sum
                    this->batchSumFloat = 0;
                    for (uint32_t k = 0; k < loopCount; ++k){
                        if(k == loopCount - 1){
                            CopyX1In(k, this->lastTileLength, offset);
                            ComputeLastSum(k, this->lastTileLength);
                        }else{
                            CopyX1In(k, this->tileLength, offset);
                            ComputeSum(k, this->tileLength);
                        }
                    }
                    this->floatRstdTensor.SetValue(j, this->batchSumFloat);
                    offset += this->batchLength;
                }
                Muls(this->floatRstdTensor, this->floatRstdTensor, this->factor, this->rstdTileLength);
                Cast(rstdLocal, this->floatRstdTensor, RoundMode::CAST_RINT, this->rstdTileLength);
                
                Cast(this->floatRstdTensor, rstdLocal, RoundMode::CAST_NONE, this->rstdTileLength);
                Adds(this->floatRstdTensor, this->floatRstdTensor, this->epsilon, this->rstdTileLength);
                Cast(rstdLocal, this->floatRstdTensor, RoundMode::CAST_RINT, this->rstdTileLength);

                Cast(this->floatRstdTensor, rstdLocal, RoundMode::CAST_NONE, this->rstdTileLength);
                Sqrt(this->floatRstdTensor, this->floatRstdTensor, this->rstdTileLength);
                Cast(rstdLocal, this->floatRstdTensor, RoundMode::CAST_RINT, this->rstdTileLength);

                Cast(this->floatRstdTensor, rstdLocal, RoundMode::CAST_NONE, this->rstdTileLength);
                Div(this->floatRstdTensor, this->OneTensor, this->floatRstdTensor, this->rstdTileLength);
                Cast(rstdLocal, this->floatRstdTensor, RoundMode::CAST_RINT, this->rstdTileLength);
                
                outQueueRstd.EnQue<bfloat16_t>(rstdLocal);
                rstdLocal = outQueueRstd.DeQue<bfloat16_t>();
                DataCopy(rstdGm[rstdOffset], rstdLocal[0], this->rstdTileLength);
                rstdOffset += this->rstdTileLength;
            }
            
            for(uint32_t j = 0; j < this->rstdLastTileLength; ++j){
                // compute sum
                this->batchSumFloat = 0;
                for (uint32_t k = 0; k < loopCount; ++k){
                    if(k == loopCount - 1){
                        CopyX1In(k, this->lastTileLength, offset);
                        ComputeLastSum(k, this->lastTileLength);
                    }else{
                        CopyX1In(k, this->tileLength, offset);
                        ComputeSum(k, this->tileLength);
                    }
                }
                this->floatRstdTensor.SetValue(j, this->batchSumFloat);
                offset += this->batchLength;
            }
            Muls(this->floatRstdTensor, this->floatRstdTensor, this->factor, this->rstdTileLength);
            Cast(rstdLocal, this->floatRstdTensor, RoundMode::CAST_RINT, this->rstdTileLength);
            
            Cast(this->floatRstdTensor, rstdLocal, RoundMode::CAST_NONE, this->rstdTileLength);
            Adds(this->floatRstdTensor, this->floatRstdTensor, this->epsilon, this->rstdTileLength);
            Cast(rstdLocal, this->floatRstdTensor, RoundMode::CAST_RINT, this->rstdTileLength);

            Cast(this->floatRstdTensor, rstdLocal, RoundMode::CAST_NONE, this->rstdTileLength);
            Sqrt(this->floatRstdTensor, this->floatRstdTensor, this->rstdTileLength);
            Cast(rstdLocal, this->floatRstdTensor, RoundMode::CAST_RINT, this->rstdTileLength);

            Cast(this->floatRstdTensor, rstdLocal, RoundMode::CAST_NONE, this->rstdTileLength);
            Div(this->floatRstdTensor, this->OneTensor, this->floatRstdTensor, this->rstdTileLength);
            Cast(rstdLocal, this->floatRstdTensor, RoundMode::CAST_RINT, this->rstdTileLength);
           
            outQueueRstd.EnQue<bfloat16_t>(rstdLocal);
            rstdLocal = outQueueRstd.DeQue<bfloat16_t>();
            DataCopy(rstdGm[rstdOffset], rstdLocal[0], this->rstdTileLength);
            rstdOffset += this->rstdTileLength;

            Cast(this->floatRstdTensor, rstdLocal, RoundMode::CAST_NONE, this->rstdTileLength);

            outQueueRstd.FreeTensor(rstdLocal);
        }

        __aicore__ inline void CopyX1In(uint32_t progress, uint32_t length, uint32_t offset) {
            LocalTensor<bfloat16_t> x1Local = inQueueX1.AllocTensor<bfloat16_t>();

            DataCopy(x1Local[0], x1Gm[offset + progress * this->tileLength], length);

            inQueueX1.EnQue(x1Local);
        }

        __aicore__ inline void ComputeSum(uint32_t progress, uint32_t length) {
            LocalTensor<bfloat16_t> x1Local = inQueueX1.DeQue<bfloat16_t>();

            Cast(this->tempTensor1, x1Local, RoundMode::CAST_NONE, length);
            Mul(this->tempTensor1, this->tempTensor1, this->tempTensor1, length);
            Cast(x1Local, this->tempTensor1, RoundMode::CAST_RINT, length);
            Cast(this->tempTensor1, x1Local, RoundMode::CAST_NONE, length);

            // ReduceSum(this->tempTensor1, this->tempTensor1, this->tempTensor2, length);
            // this->batchSumFloat += this->tempTensor1.GetValue(0);

            for(uint32_t i = 0; i < length; ++i){
                this->batchSumFloat += this->tempTensor1.GetValue(i);
            }

            inQueueX1.FreeTensor(x1Local);
        }

        __aicore__ inline void ComputeLastSum(uint32_t progress, uint32_t length) {
            LocalTensor<bfloat16_t> x1Local = inQueueX1.DeQue<bfloat16_t>();

            Cast(this->tempTensor1, x1Local, RoundMode::CAST_NONE, length);
            Mul(this->tempTensor1, this->tempTensor1, this->tempTensor1, length);
            Cast(x1Local, this->tempTensor1, RoundMode::CAST_RINT, length);
            Cast(this->tempTensor1, x1Local, RoundMode::CAST_NONE, length);

            const float ZERO = 0.0f;
            for(uint32_t k = 1; k <= this->resLength; ++k){
                this->tempTensor1.SetValue(length - k, ZERO);
            }
            
            // ReduceSum(this->tempTensor1, this->tempTensor1, this->tempTensor2, length);
            // this->batchSumFloat += this->tempTensor1.GetValue(0);

            for(uint32_t i = 0; i < length - this->resLength; ++i){
                this->batchSumFloat += this->tempTensor1.GetValue(i);
            }

            inQueueX1.FreeTensor(x1Local);
        }

        __aicore__ inline void ComputeY() {
            uint32_t loopCount = this->tileNum * BUFFER_NUM;
            uint32_t copyOffset = 0;

            for(uint32_t i = 0; i < this->batchNum; ++i){
                this->rstd = rstdGm.GetValue(i);
                for (uint32_t j = 0; j < loopCount; ++j){
                    if(j == loopCount - 1){
                        CopyX1X2In(j, this->lastTileLength, copyOffset);
                        ComputeMul(j, this->lastTileLength);
                        CopyOut(j, this->lastTileLength, copyOffset);
                    }else{
                        CopyX1X2In(j, this->tileLength, copyOffset);
                        ComputeMul(j, this->tileLength);
                        CopyOut(j, this->tileLength, copyOffset);
                    }
                }
                copyOffset += this->batchLength;
            }
        }

        __aicore__ inline void CopyX1X2In(uint32_t progress, uint32_t length, uint32_t offset) {
            LocalTensor<bfloat16_t> x1Local = inQueueX1.AllocTensor<bfloat16_t>();
            LocalTensor<bfloat16_t> x2Local = inQueueX2.AllocTensor<bfloat16_t>();

            DataCopy(x1Local[0], x1Gm[offset + progress * this->tileLength], length);
            DataCopy(x2Local[0], x2Gm[progress * this->tileLength], length);

            inQueueX1.EnQue(x1Local);
            inQueueX2.EnQue(x2Local);
        }

        __aicore__ inline void ComputeMul(uint32_t progress, uint32_t length) {
            LocalTensor<bfloat16_t> x1Local = inQueueX1.DeQue<bfloat16_t>();
            LocalTensor<bfloat16_t> x2Local = inQueueX2.DeQue<bfloat16_t>();
            LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();

            this->tempBFTensor.SetValue(0, this->rstd);
            Cast(this->floatRstdTensor, this->tempBFTensor, RoundMode::CAST_NONE, 16);
            this->floatRstd = this->floatRstdTensor.GetValue(0);
            Cast(this->tempTensor1, x1Local, RoundMode::CAST_NONE, length);
            Muls(this->tempTensor1, this->tempTensor1, this->floatRstd, length);
            Cast(x1Local, this->tempTensor1, RoundMode::CAST_RINT, length);
            Cast(this->tempTensor1, x1Local, RoundMode::CAST_NONE, length);
            Cast(this->tempTensor2, x2Local, RoundMode::CAST_NONE, length);
            Mul(this->tempTensor1, this->tempTensor1, this->tempTensor2, length);
            
            if(this->resLength != 0){
                Cast(x1Local, this->tempTensor1, RoundMode::CAST_RINT, length);
                bfloat16_t t;
                for(uint32_t i = 0; i < length; ++i){
                    t = x1Local.GetValue(i);
                    yLocal.SetValue(i, t);
                }
            }else{
                Cast(yLocal, this->tempTensor1, RoundMode::CAST_RINT, length);
            }
            
            outQueueY.EnQue(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }
    
        __aicore__ inline void CopyOut(uint32_t progress, uint32_t length, uint32_t offset) {
            LocalTensor<bfloat16_t> yLocal = outQueueY.DeQue<bfloat16_t>();

            if(this->resLength != 0){
                bfloat16_t t;
                for(uint32_t i = 0; i < length; ++i){
                    t = yLocal.GetValue(i);
                    yGm.SetValue(offset + progress * this->tileLength + i, t);
                }
            }else{
                DataCopy(yGm[offset + progress * this->tileLength], yLocal[0], length);
            }
    
            outQueueY.FreeTensor(yLocal);
        }
    
    private:
        TPipe pipe;
        TBuf<TPosition::VECCALC> buffer1, buffer2, buffer3, buffer4, buffer5;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY, outQueueRstd;
        LocalTensor<float> OneTensor;
        LocalTensor<bfloat16_t> tempBFTensor;
        LocalTensor<float> tempTensor1, tempTensor2, floatRstdTensor;
        GlobalTensor<bfloat16_t> x1Gm, x2Gm, yGm, rstdGm;
        bfloat16_t rstd;
        float epsilon, batchSumFloat, factor, floatRstd;;
        uint32_t batchNum, batchLength, x1TotalLength, x2TotalLength;
        uint32_t tileNum, tileLength, lastTileLength, resLength;
        uint32_t rstdTileNum, rstdTileLength, rstdLastTileLength;
};

extern "C" __global__ __aicore__ void rms_norm(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
    if(TILING_KEY_IS(0)){
        GET_TILING_DATA(tiling_data, tiling);
        KernelRmsNorm0 op;
        op.Init(x1, x2, y, rstd, tiling_data.epsilon, tiling_data.x1TotalLength, 
                tiling_data.x2TotalLength, tiling_data.batchNum, tiling_data.batchLength, 
                tiling_data.tileNum, tiling_data.tileLength, tiling_data.lastTileLength,
                tiling_data.rstdTileNum, tiling_data.rstdTileLength, tiling_data.rstdLastTileLength, tiling_data.resLength);
        op.Process();
    }else if(TILING_KEY_IS(1)){
        GET_TILING_DATA(tiling_data, tiling);
        KernelRmsNorm1 op;
        op.Init(x1, x2, y, rstd, tiling_data.epsilon, tiling_data.x1TotalLength, 
                tiling_data.x2TotalLength, tiling_data.batchNum, tiling_data.batchLength, 
                tiling_data.tileNum, tiling_data.tileLength, tiling_data.lastTileLength,
                tiling_data.rstdTileNum, tiling_data.rstdTileLength, tiling_data.rstdLastTileLength, tiling_data.resLength, tiling_data.mulLastTileLength);
        op.Process();
    }else if(TILING_KEY_IS(2)){
        GET_TILING_DATA(tiling_data, tiling);
        KernelRmsNorm2 op;
        op.Init(x1, x2, y, rstd, tiling_data.epsilon, tiling_data.x1TotalLength, 
                tiling_data.x2TotalLength, tiling_data.batchNum, tiling_data.batchLength, 
                tiling_data.tileNum, tiling_data.tileLength, tiling_data.lastTileLength,
                tiling_data.rstdTileNum, tiling_data.rstdTileLength, tiling_data.rstdLastTileLength, tiling_data.resLength);
        op.Process();
    }
}