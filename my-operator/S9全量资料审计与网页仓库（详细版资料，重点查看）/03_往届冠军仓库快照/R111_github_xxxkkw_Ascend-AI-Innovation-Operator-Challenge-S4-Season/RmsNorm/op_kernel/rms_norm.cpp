#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;  

class KernelRmsNorm{
public:
    __aicore__ inline KernelRmsNorm(){}
    __aicore__ inline void InitTiling(GM_ADDR tiling) {
        GET_TILING_DATA(tiling_data, tiling); 
        rstdGmLength = tiling_data.rstdGmLength;
        rstdLength = tiling_data.rstdLength;   
        totalLength = tiling_data.totalLength;      // 总数据量
        tileLoop = tiling_data.tileLoop;            // 核内单次计行数极限
        maxPerTime = tiling_data.maxPerTime;        // 核内单次计算数据量,用于datacopy时长度
        loopCount = tiling_data.loopCount;          // 核内计算总需循环数
        leftNum = tiling_data.leftNum;              // 如果有余行，单独计算一下
        leftPerTime = tiling_data.leftPerTime;      // 余块计算长度 
        tileLength = tiling_data.tileLength;        // 矩阵每行长度
        tileNum = tiling_data.tileNum;              // 矩阵总行数
        factor = tiling_data.factor;                // 1 / D  求均值系数 行长倒数
        eps = tiling_data.eps;                      // 缩放
    }
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd,GM_ADDR tiling){
        
        InitTiling(tiling);
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
       
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, this->totalLength);                  // 无多核 索性不偏移 矩阵总数据为totalLength
        gammaGm.SetGlobalBuffer((__gm__ DTYPE_X*)gamma, this->tileLength);           // gamma的长度为 tileLength 也就是D
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, this->totalLength);
        rstdGm.SetGlobalBuffer((__gm__ DTYPE_Y*)rstd, this->rstdGmLength);           // 为了datacopy 
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->maxPerTime * sizeof(DTYPE_X));   // 队列宽度为maxPerTime
        pipe.InitBuffer(inQueueG, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));   // gamma仅拷一次，一次性全放核心内
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->maxPerTime * sizeof(DTYPE_Y));   
        pipe.InitBuffer(outQueueR, BUFFER_NUM, this->rstdLength * sizeof(DTYPE_Y));    // 队列宽度为tileLoop，存中间的rstd分块
        
        if constexpr (std::is_same_v<DTYPE_X, bfloat16_t> || std::is_same_v<DTYPE_X, half>){
            pipe.InitBuffer(tmpBuffer1, this->maxPerTime * sizeof(float));
            pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(float));
            pipe.InitBuffer(tmpBuffer3, this->rstdLength * sizeof(float));
            pipe.InitBuffer(tmpBuffer4, this->rstdLength * sizeof(float));
            pipe.InitBuffer(tmpBuffer5, this->tileLength * sizeof(float));
            pipe.InitBuffer(tmpBuffer6, this->maxPerTime * sizeof(float));
            pipe.InitBuffer(tmpBuffer7, this->tileLength * sizeof(float));
        }else{
            pipe.InitBuffer(tmpBuffer1, this->maxPerTime * sizeof(float));
            pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(float));
            pipe.InitBuffer(tmpBuffer3, this->rstdLength * sizeof(float));
            pipe.InitBuffer(tmpBuffer4, this->rstdLength * sizeof(float));
            pipe.InitBuffer(tmpBuffer7, this->tileLength * sizeof(float));
        }
        
    }
    __aicore__ inline void Process() {
        LocalTensor<DTYPE_X> gammaLocal = inQueueG.AllocTensor<DTYPE_X>();     // gamma只用拷一次，长期放在核心内
        DataCopy(gammaLocal, gammaGm,this->tileLength);                        // 一次性做完拷贝
        inQueueG.EnQue(gammaLocal);
        LocalTensor<DTYPE_X> GammaLocal = inQueueG.DeQue<DTYPE_X>();
        
        for (int32_t i = 0; i < this->loopCount; i++) {
            CopyIn(i, this->maxPerTime);                                       // progress totalDataLength
            Compute(i, this->maxPerTime,this->tileLoop,GammaLocal);            // progress totalDataLength dataWideN gammaLocal
            CopyOut(i, this->maxPerTime, this->tileLoop);
        }
        if (this->leftNum > 0) {
            CopyIn(this->loopCount, this->leftPerTime);
            Compute(this->loopCount, this->leftPerTime,this->leftNum, GammaLocal);
            CopyOut(this->loopCount, this->leftPerTime,this->leftNum);
        }
        inQueueG.FreeTensor(gammaLocal);
    }
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t totalDataLength) {
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm[progress * this->maxPerTime], totalDataLength);    // 地址偏移 数据总长
        inQueueX.EnQue(xLocal);
    }

     __aicore__ inline void Compute(int32_t progress, uint32_t totalDataLength,uint32_t dataWideN ,LocalTensor<DTYPE_X> gammaLocal) {
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        LocalTensor<DTYPE_Y> rstdLocal = outQueueR.AllocTensor<DTYPE_Y>();

        if constexpr (std::is_same_v<DTYPE_X, bfloat16_t> || std::is_same_v<DTYPE_X, half>){
            LocalTensor<float> xTmpLocal = tmpBuffer1.Get<float>();            // tmpBuffer 1 2 3 分别用于暂存
            LocalTensor<float> gammaTmpLocal = tmpBuffer2.Get<float>();        // x  [N,D] gamma [D]
            LocalTensor<float> rstdTmpLocal = tmpBuffer3.Get<float>();         // rstd中间态
            LocalTensor<float> meanTmpLocal = tmpBuffer6.Get<float>();
            Cast(xTmpLocal,xLocal,RoundMode::CAST_NONE,totalDataLength);       // fp16 bf16均填充到fp32
            Cast(gammaTmpLocal,gammaLocal,RoundMode::CAST_NONE,this->tileLength);
            Mul(meanTmpLocal,xTmpLocal,xTmpLocal,totalDataLength);
            LocalTensor<float> TmpLocal = tmpBuffer7.Get<float>();
            for (uint32_t j = 0; j < dataWideN; ++j) {                        
                uint32_t buffIndex = j * this->tileLength;
                ReduceSum<float>(rstdTmpLocal[j], meanTmpLocal[buffIndex], TmpLocal, this->tileLength);
            }
            Muls(rstdTmpLocal, rstdTmpLocal, (float)this->factor, dataWideN);     
            Adds(rstdTmpLocal, rstdTmpLocal, (float)this->eps, dataWideN);
            LocalTensor<float> rstdAnsLocal = tmpBuffer4.Get<float>();         
            Sqrt(rstdTmpLocal, rstdTmpLocal, dataWideN);
            LocalTensor<float> tmpTensor3 = tmpBuffer5.Get<float>();          // tmpBuffer 5 存rstd的一个值 [D]
            for (uint32_t j = 0; j < dataWideN; ++j) {
                uint32_t buffIndex = j * this->tileLength;
                Duplicate(tmpTensor3,rstdTmpLocal.GetValue(j),this->tileLength);  
                Div(xTmpLocal[buffIndex],xTmpLocal[buffIndex],tmpTensor3,this->tileLength);
                Mul(xTmpLocal[buffIndex],xTmpLocal[buffIndex],gammaTmpLocal,this->tileLength);
            }
            Duplicate(rstdAnsLocal,(float)1.0f,dataWideN);
            Div(rstdAnsLocal,rstdAnsLocal,rstdTmpLocal,dataWideN);
            Cast(yLocal,xTmpLocal,RoundMode::CAST_RINT,totalDataLength);
            Cast(rstdLocal,rstdAnsLocal,RoundMode::CAST_RINT,dataWideN);
        }else{
            LocalTensor<DTYPE_X> mean = tmpBuffer1.Get<DTYPE_X>();                // 存求和        [N,D]
            LocalTensor<DTYPE_X> rstdTmpLocal = tmpBuffer4.Get<DTYPE_X>();
            Mul(mean,xLocal,xLocal,totalDataLength);                              // x^2          [N,D]
            LocalTensor<DTYPE_X> TmpLocal = tmpBuffer7.Get<DTYPE_X>();
            for (uint32_t j = 0; j < dataWideN; ++j) {                            // sum(x^2)     [N,1]
                uint32_t buffIndex = j * this->tileLength;
                ReduceSum<DTYPE_X>(rstdTmpLocal[j], mean[buffIndex], TmpLocal, this->tileLength);
            }
            Muls(rstdTmpLocal,rstdTmpLocal,(DTYPE_X)this->factor, dataWideN);     // sum(x^2)/D   [N,1]
            Adds(rstdTmpLocal,rstdTmpLocal,(DTYPE_X)this->eps, dataWideN);        // + epsilon    [N,1]
            Sqrt(rstdTmpLocal,rstdTmpLocal,dataWideN);                            // sqrt(mean)   [N,1]
            LocalTensor<DTYPE_X> tmpTensor3 = tmpBuffer2.Get<DTYPE_X>();          // 存rstd的一个值 [D]
            for (uint32_t j = 0; j < dataWideN; ++j) {
                uint32_t buffIndex = j * this->tileLength;
                Duplicate(tmpTensor3,rstdTmpLocal.GetValue(j),this->tileLength);  // 没有标量除，只能填充一下再计算
                Div(yLocal[buffIndex],xLocal[buffIndex],tmpTensor3,this->tileLength);
                Mul(yLocal[buffIndex],yLocal[buffIndex],gammaLocal,this->tileLength);
            }
            LocalTensor<DTYPE_X> rstdAnsLocal = tmpBuffer3.Get<DTYPE_X>();
            Duplicate(rstdAnsLocal,(DTYPE_X)1.0f,dataWideN);
            Div(rstdLocal,rstdAnsLocal,rstdTmpLocal,dataWideN);
        }

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        outQueueR.EnQue<DTYPE_Y>(rstdLocal);
        inQueueX.FreeTensor(xLocal);
    }

   __aicore__ inline void CopyOut(int32_t progress, uint32_t totalDataLength,uint32_t dataWideN) {
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        LocalTensor<DTYPE_Y> rstdLocal = outQueueR.DeQue<DTYPE_Y>();
        DataCopy(rstdGm[progress * this->tileLoop], rstdLocal, this->rstdLength);
        DataCopy(yGm[progress * this->maxPerTime], yLocal, totalDataLength);
        outQueueY.FreeTensor(yLocal);
        outQueueR.FreeTensor(rstdLocal);
    }

private:
    // 类对象的初始化
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueG;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueR;

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_X> gammaGm;
    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<DTYPE_Y> rstdGm;
    TBuf<QuePosition::VECCALC> tmpBuffer1;
    TBuf<QuePosition::VECCALC> tmpBuffer2;
    TBuf<QuePosition::VECCALC> tmpBuffer3;
    TBuf<QuePosition::VECCALC> tmpBuffer4;
    TBuf<QuePosition::VECCALC> tmpBuffer5;
    TBuf<QuePosition::VECCALC> tmpBuffer6;
    TBuf<QuePosition::VECCALC> tmpBuffer7;

    uint64_t totalLength;
    uint64_t tileLoop; 
    uint64_t maxPerTime;
    uint64_t loopCount;
    uint64_t leftNum; 
    uint64_t leftPerTime; 
    uint64_t tileLength; 
    uint64_t tileNum; 
    uint64_t rstdLength;
    uint64_t rstdGmLength;
    float factor;
    float eps;
};


extern "C" __global__ __aicore__ void rms_norm(GM_ADDR x,GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
    KernelRmsNorm op;
    op.Init(x,gamma,y,rstd,tiling);
    op.Process();
}