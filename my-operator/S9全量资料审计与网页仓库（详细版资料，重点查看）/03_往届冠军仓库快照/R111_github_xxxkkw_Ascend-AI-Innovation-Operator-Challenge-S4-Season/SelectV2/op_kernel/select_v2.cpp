#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;  

class KernelSelectV2{ // tiling key 1 正常情况不用广播
public:
    __aicore__ inline KernelSelectV2(){}
    __aicore__ inline void InitTiling(GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    totalLength = tiling_data.totalLength;
        tileLength = tiling_data.tileLength;
        loopCount = tiling_data.loopCount;
        leftNum = tiling_data.leftNum;
    }
    __aicore__ inline void Init(GM_ADDR c,GM_ADDR x1,GM_ADDR x2, GM_ADDR y,GM_ADDR tiling,TPipe* pipeIn){
        InitTiling(tiling);
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        
        cGm.SetGlobalBuffer((__gm__ bool*)c, this->totalLength);
        x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1*)x1,this->totalLength);
        x2Gm.SetGlobalBuffer((__gm__ DTYPE_X1*)x2,this->totalLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y,this->totalLength);
        pipe = pipeIn;
        pipe->InitBuffer(inQueueC, BUFFER_NUM, this->tileLength * sizeof(bool));
        pipe->InitBuffer(inQueueX1, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X1));
        pipe->InitBuffer(inQueueX2, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X2));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe->InitBuffer(zeroBuffer,this->tileLength * sizeof(half));
        pipe->InitBuffer(cBuffer,this->tileLength * sizeof(half));
        if constexpr (std::is_same_v<DTYPE_X1, int32_t>){
            pipe->InitBuffer(tmpBuffer1, this->tileLength * sizeof(float));
            pipe->InitBuffer(tmpBuffer2, this->tileLength * sizeof(float));
            pipe->InitBuffer(tmpBuffer3, this->tileLength * sizeof(float));
        }
        else if constexpr (std::is_same_v<DTYPE_X1, int8_t>){
            pipe->InitBuffer(tmpBuffer1, this->tileLength * sizeof(half));
            pipe->InitBuffer(tmpBuffer2, this->tileLength * sizeof(half));
            pipe->InitBuffer(tmpBuffer3, this->tileLength * sizeof(half));
        }
    }
    __aicore__ inline void Process() {
        for (int32_t i = 0; i < this->loopCount; ++ i) {
            CopyIn(i,  this->tileLength);       // 32B对齐后的长度                                
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        if (this->leftNum > 0) {
            CopyIn(this->loopCount, this->leftNum);
            Compute(this->loopCount, this->leftNum);
            CopyOut(this->loopCount, this->leftNum);
        }
    }
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<DTYPE_X1> x1Local = inQueueX1.AllocTensor<DTYPE_X1>();
        LocalTensor<DTYPE_X2> x2Local = inQueueX2.AllocTensor<DTYPE_X2>();
        LocalTensor<bool> cLocal = inQueueC.AllocTensor<bool>();
        DataCopy(cLocal, cGm[progress * this->tileLength], length);
        DataCopy(x1Local, x1Gm[progress * this->tileLength], length);
        DataCopy(x2Local, x2Gm[progress * this->tileLength], length);
        inQueueC.EnQue(cLocal);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

     __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<bool> cLocal = inQueueC.DeQue<bool>();
        LocalTensor<DTYPE_X1> x1Local = inQueueX1.DeQue<DTYPE_X1>();
        LocalTensor<DTYPE_X2> x2Local = inQueueX2.DeQue<DTYPE_X2>();
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        LocalTensor<half> cTmpLocal = cBuffer.Get<half>();
        auto intc = cLocal.template ReinterpretCast<uint8_t>();
        LocalTensor<half> zeroLocal = zeroBuffer.Get<half>();
        Cast(cTmpLocal,intc,RoundMode::CAST_NONE,length);
        Duplicate(zeroLocal, half(0), length);
        Compare(zeroLocal,cTmpLocal, zeroLocal, CMPMODE::NE, length);
        if constexpr (std::is_same_v<DTYPE_X1, float> || std::is_same_v<DTYPE_X1, half>){
            Select(yLocal,zeroLocal,x1Local,x2Local,SELMODE::VSEL_TENSOR_TENSOR_MODE,length);
        }
        if constexpr (std::is_same_v<DTYPE_X1, int8_t>){
            LocalTensor<half> x1TmpLocal = tmpBuffer1.Get<half>();
            LocalTensor<half> x2TmpLocal = tmpBuffer2.Get<half>();
            LocalTensor<half> yTmpLocal = tmpBuffer3.Get<half>();
            Cast(x1TmpLocal,x1Local,RoundMode::CAST_NONE,length);
            Cast(x2TmpLocal,x2Local,RoundMode::CAST_NONE,length);
            Select(yTmpLocal,zeroLocal,x1TmpLocal,x2TmpLocal,SELMODE::VSEL_TENSOR_TENSOR_MODE,length);
            Cast(yLocal,yTmpLocal,RoundMode::CAST_RINT,length);
        }else if constexpr (std::is_same_v<DTYPE_X1, int32_t>){
            LocalTensor<float> x1TmpLocal = tmpBuffer1.Get<float>();
            LocalTensor<float> x2TmpLocal = tmpBuffer2.Get<float>();
            LocalTensor<float> yTmpLocal = tmpBuffer3.Get<float>();
            Cast(x1TmpLocal,x1Local,RoundMode::CAST_NONE,length);
            Cast(x2TmpLocal,x2Local,RoundMode::CAST_NONE,length);
            
            Select(yTmpLocal,zeroLocal,x1TmpLocal,x2TmpLocal,SELMODE::VSEL_TENSOR_TENSOR_MODE,length);
            Cast(yLocal,yTmpLocal,RoundMode::CAST_RINT,length);
        }

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueC.FreeTensor(cLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }

   __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        DataCopy(yGm[progress * this->tileLength], yLocal, length);
        outQueueY.FreeTensor(yLocal);
    }

private:
    //一些类对象的初始化
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueC;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    GlobalTensor<bool> cGm;
    GlobalTensor<DTYPE_X1> x1Gm;
    GlobalTensor<DTYPE_X2> x2Gm;
    GlobalTensor<DTYPE_Y> yGm;
    TBuf<QuePosition::VECCALC> tmpBuffer1;
    TBuf<QuePosition::VECCALC> tmpBuffer2;
    TBuf<QuePosition::VECCALC> tmpBuffer3;
    TBuf<QuePosition::VECCALC> zeroBuffer;
    TBuf<QuePosition::VECCALC> cBuffer;    

    TPipe* pipe;
    uint64_t totalLength;
    uint64_t tileLength;
    uint64_t loopCount;
    uint64_t leftNum;
};

class KernelSelectV2BroadCast{  // 非对齐 广播case
    public:
        __aicore__ inline KernelSelectV2BroadCast(){}
        __aicore__ inline void InitTiling(GM_ADDR tiling) {
        GET_TILING_DATA(tiling_data, tiling); 
        y_dimensional = tiling_data.y_dimensional;
        y_ndarray = tiling_data.y_ndarray;
        c_ndarray = tiling_data.c_ndarray;
        x1_ndarray = tiling_data.x1_ndarray;
        x2_ndarray = tiling_data.x2_ndarray;
        y_sumndarray = tiling_data.y_sumndarray;
        c_sumndarray = tiling_data.c_sumndarray;
        x1_sumndarray = tiling_data.x1_sumndarray;
        x2_sumndarray = tiling_data.x2_sumndarray;
        cSize = tiling_data.cSize;
        x1Size = tiling_data.x1Size;
        x2Size = tiling_data.x2Size;
        totalLength = tiling_data.totalLength;
        tileLength = tiling_data.tileLength;
        loopCount = tiling_data.loopCount;
        leftNum = tiling_data.leftNum;

    }
    __aicore__ inline void Init(GM_ADDR c,GM_ADDR x1,GM_ADDR x2, GM_ADDR y,GM_ADDR tiling,TPipe* pipeIn){
        InitTiling(tiling);
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        cGm.SetGlobalBuffer((__gm__ bool*)c, this->cSize);
        x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1*)x1,this->x1Size);
        x2Gm.SetGlobalBuffer((__gm__ DTYPE_X1*)x2,this->x2Size);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y,this->totalLength);
        pipe = pipeIn;
        pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe->InitBuffer(zeroBuffer,this->tileLength * sizeof(half));
        pipe->InitBuffer(cBuffer,this->tileLength * sizeof(half));
        pipe->InitBuffer(tmpBufferC,this->tileLength * sizeof(DTYPE_Y));
        pipe->InitBuffer(tmpBufferX1,this->tileLength * sizeof(DTYPE_Y));
        pipe->InitBuffer(tmpBufferX2,this->tileLength * sizeof(DTYPE_Y));
        if constexpr (std::is_same_v<DTYPE_X1, int32_t>){
            pipe->InitBuffer(tmpBuffer1, this->tileLength * sizeof(float));
            pipe->InitBuffer(tmpBuffer2, this->tileLength * sizeof(float));
            pipe->InitBuffer(tmpBuffer3, this->tileLength * sizeof(float));
        }
        else if constexpr (std::is_same_v<DTYPE_X1, int8_t>){
            pipe->InitBuffer(tmpBuffer1, this->tileLength * sizeof(half));
            pipe->InitBuffer(tmpBuffer2, this->tileLength * sizeof(half));
            pipe->InitBuffer(tmpBuffer3, this->tileLength * sizeof(half));
        }
    }

     __aicore__ inline void Process() {
        LocalTensor<bool> cLocal = tmpBufferC.Get<bool>();
        LocalTensor<DTYPE_X1> x1Local = tmpBufferX1.Get<DTYPE_X1>();
        LocalTensor<DTYPE_X2> x2Local = tmpBufferX2.Get<DTYPE_X2>();
        for (uint32_t i = 0; i < this->loopCount; i ++) {               
            Compute(i, this->tileLength,cLocal,x1Local,x2Local);           
        CopyOut(i, this->tileLength);
    }
    if (this->leftNum > 0) {
            Compute(this->loopCount, this->leftNum,cLocal,x1Local,x2Local);
        CopyOut(this->loopCount, this->leftNum);
    }
}
private:
    __aicore__ inline void Compute(uint32_t progress, uint32_t length,LocalTensor<bool> cLocal,LocalTensor<DTYPE_X1> x1Local,LocalTensor<DTYPE_X2> x2Local) { 
    LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        uint32_t offset = progress * this->tileLength;
        for(uint32_t j = 0;j < length;j ++){
        uint32_t c_start = 0;
        uint32_t x1_start = 0;
        uint32_t x2_start = 0;
            uint32_t index = j + offset;
            for (uint32_t k = 0; k < this->y_dimensional;k ++){
            if (this->c_ndarray[k] != 1){
                    c_start += this->c_sumndarray[k] * (index / this->y_sumndarray[k] % this->y_ndarray[k]);
            }
            if (this->x1_ndarray[k] != 1){
                    x1_start += this->x1_sumndarray[k] * (index / this->y_sumndarray[k] % this->y_ndarray[k]);
            }
            if(this->x2_ndarray[k] != 1){
                    x2_start += this->x2_sumndarray[k] * (index / this->y_sumndarray[k] % this->y_ndarray[k]);  
            }
        }
        auto c = cGm.GetValue(c_start);
        auto x1 = x1Gm.GetValue(x1_start); 
        auto x2 = x2Gm.GetValue(x2_start);
        cLocal.SetValue(j,c);
        x1Local.SetValue(j,x1);
        x2Local.SetValue(j,x2);
    }
    LocalTensor<half> cTmpLocal = cBuffer.Get<half>();
    auto intc = cLocal.template ReinterpretCast<uint8_t>();
    LocalTensor<half> zeroLocal = zeroBuffer.Get<half>();
    Cast(cTmpLocal,intc,RoundMode::CAST_NONE,length);
        Duplicate(zeroLocal, half(0), length);
    Compare(zeroLocal,cTmpLocal, zeroLocal, CMPMODE::NE, length);
    if constexpr (std::is_same_v<DTYPE_X1, float> || std::is_same_v<DTYPE_X1, half>){
        Select(yLocal,zeroLocal,x1Local,x2Local,SELMODE::VSEL_TENSOR_TENSOR_MODE,length);
    } 
    if constexpr (std::is_same_v<DTYPE_X1, int8_t>){
        LocalTensor<half> x1TmpLocal = tmpBuffer1.Get<half>();
        LocalTensor<half> x2TmpLocal = tmpBuffer2.Get<half>();
        LocalTensor<half> yTmpLocal = tmpBuffer3.Get<half>();
        Cast(x1TmpLocal,x1Local,RoundMode::CAST_NONE,length);
        Cast(x2TmpLocal,x2Local,RoundMode::CAST_NONE,length);
        Select(yTmpLocal,zeroLocal,x1TmpLocal,x2TmpLocal,SELMODE::VSEL_TENSOR_TENSOR_MODE,length);
        Cast(yLocal,yTmpLocal,RoundMode::CAST_RINT,length);
    }else if constexpr (std::is_same_v<DTYPE_X1, int32_t>){
        LocalTensor<float> x1TmpLocal = tmpBuffer1.Get<float>();
        LocalTensor<float> x2TmpLocal = tmpBuffer2.Get<float>();
        LocalTensor<float> yTmpLocal = tmpBuffer3.Get<float>();
        Cast(x1TmpLocal,x1Local,RoundMode::CAST_NONE,length);
        Cast(x2TmpLocal,x2Local,RoundMode::CAST_NONE,length);
        Select(yTmpLocal,zeroLocal,x1TmpLocal,x2TmpLocal,SELMODE::VSEL_TENSOR_TENSOR_MODE,length);
        Cast(yLocal,yTmpLocal,RoundMode::CAST_RINT,length);
    }

    outQueueY.EnQue<DTYPE_Y>(yLocal);
}

__aicore__ inline void CopyOut(int32_t progress, uint32_t length) {  //原始地址 输出
    LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
    DataCopy(yGm[progress * this->tileLength], yLocal, length);
    outQueueY.FreeTensor(yLocal);
}

private:
TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

GlobalTensor<bool> cGm;
GlobalTensor<DTYPE_X1> x1Gm;
GlobalTensor<DTYPE_X2> x2Gm;
GlobalTensor<DTYPE_Y> yGm;
TBuf<QuePosition::VECCALC> tmpBuffer1;
TBuf<QuePosition::VECCALC> tmpBuffer2;
TBuf<QuePosition::VECCALC> tmpBuffer3;
TBuf<QuePosition::VECCALC> tmpBufferC;
TBuf<QuePosition::VECCALC> tmpBufferX1;
TBuf<QuePosition::VECCALC> tmpBufferX2;
TBuf<QuePosition::VECCALC> zeroBuffer;
TBuf<QuePosition::VECCALC> cBuffer;


TPipe* pipe;
uint64_t totalLength;
uint64_t tileLength;
uint64_t loopCount;
uint64_t leftNum;
uint32_t y_dimensional;
uint32_t *c_sumndarray;
uint32_t *x1_sumndarray;
uint32_t *x2_sumndarray;
uint32_t *y_ndarray;
uint32_t *c_ndarray;
uint32_t *x1_ndarray;
uint32_t *x2_ndarray;
uint32_t *y_sumndarray;
uint32_t cSize;
uint32_t x1Size;
uint32_t x2Size;
};
extern "C" __global__ __aicore__ void select_v2(GM_ADDR c, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    // TODO: user kernel impl
    if(TILING_KEY_IS(1)){
        KernelSelectV2 op;
        op.Init(c,x1,x2, y, tiling,&pipe);
        op.Process();
    }
    else if(TILING_KEY_IS(2)){
        KernelSelectV2BroadCast op;
        op.Init(c,x1,x2, y, tiling,&pipe);
        op.Process();
    }
}
