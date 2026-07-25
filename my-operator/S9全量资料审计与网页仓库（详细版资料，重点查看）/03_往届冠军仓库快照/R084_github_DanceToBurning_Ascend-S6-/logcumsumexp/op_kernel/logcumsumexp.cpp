#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template<typename T> class KernelLogcumsumexp {
public:
    __aicore__ inline KernelLogcumsumexp() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, 
                            int32_t size_in, int32_t x_ndarray_in[], int32_t x_dimensional_in, int32_t tileDataMaxNum_in,int32_t dim_in) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        
        this->x_dimensional = x_dimensional_in;
        for(int i = 0; i < x_dimensional_in; i++) {
            this->x_ndarray[i] = x_ndarray_in[i];
        }
        this->size = size_in;
        this->dim = dim_in;
        if(this->dim < 0) {
            this->dim += this->x_dimensional;
        }

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)(x));
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)(y));

        int32_t cycles = 1;
        int32_t interval = 1;
        int32_t loopCount = 1;

        for(int32_t i = 0; i < this->dim; i++) {
            loopCount *= this->x_ndarray[i];
        }
        cycles = this->x_ndarray[this->dim];
        for(int32_t i = this->dim + 1; i < this->x_dimensional; i++) {
            interval *= this->x_ndarray[i];
        }

        this->cycles = cycles;
        this->interval = interval;
        this->loopCount = loopCount;

        
        
        const int32_t DMA_BYTE_SIZE = 32;
        int32_t logicalTileNum = ((tileDataMaxNum_in * sizeof(T) + DMA_BYTE_SIZE - 1) / DMA_BYTE_SIZE) * DMA_BYTE_SIZE / sizeof(T);
        
        this->tileDataMaxNum = logicalTileNum; 
        this->circulate = interval / this->tileDataMaxNum;
        this->lastHoleData = interval % this->tileDataMaxNum;
        this->SingleData = this->tileDataMaxNum;

        
        const int32_t VEC_BYTE_SIZE = 256;

        
        int32_t bufferBytesT = 
            ((this->tileDataMaxNum * sizeof(T) + VEC_BYTE_SIZE - 1) / VEC_BYTE_SIZE) * VEC_BYTE_SIZE;

        
        int32_t bufferBytesFloat = 
            ((this->tileDataMaxNum * sizeof(float) + VEC_BYTE_SIZE - 1) / VEC_BYTE_SIZE) * VEC_BYTE_SIZE;

        
        int32_t bufferBytesU8 = 
            ((this->tileDataMaxNum * sizeof(uint8_t) + VEC_BYTE_SIZE - 1) / VEC_BYTE_SIZE) * VEC_BYTE_SIZE;

        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, bufferBytesT);
        pipe.InitBuffer(outQueueY, BUFFER_NUM, bufferBytesT);
        pipe.InitBuffer(QueueTemp1, bufferBytesFloat);
        pipe.InitBuffer(QueueTemp2, bufferBytesFloat);
        pipe.InitBuffer(tmpBuffer1, bufferBytesFloat);
        pipe.InitBuffer(tmpBuffer4, bufferBytesFloat);
        
        if constexpr(!std::is_same_v<T, float>){
            pipe.InitBuffer(tmpBuffer2, bufferBytesFloat);
            pipe.InitBuffer(tmpBuffer3, bufferBytesFloat);
        }
    
    }

    __aicore__ inline void Process() {
    int32_t blockNum = GetBlockNum();
    int32_t blockId = GetBlockIdx();
    
    
    int32_t total_k_tasks = this->circulate + (this->lastHoleData > 0 ? 1 : 0);

    
    if (this->loopCount >= total_k_tasks) {
        
        
        int32_t totalTasks = this->loopCount;
        if (totalTasks == 0) return;

        
        int32_t tasksPerBlock = totalTasks / blockNum;
        int32_t remainderTasks = totalTasks % blockNum;
        
        int32_t start_z = tasksPerBlock * blockId + (blockId < remainderTasks ? blockId : remainderTasks);
        int32_t end_z = start_z + (tasksPerBlock + (blockId < remainderTasks ? 1 : 0));

        
        for (int32_t z = start_z; z < end_z; z++) {
            
            LocalTensor<float> presum = QueueTemp1.Get<float>();
            LocalTensor<float> premax = QueueTemp2.Get<float>();
            
            
            for (int32_t k = 0; k < circulate; ++k) {
                for (int32_t i = 0; i < cycles; i++) {
                    CopyIn(z, i, k, tileDataMaxNum);
                    Compute(z, i, k, presum, premax, i == 0, tileDataMaxNum);
                    CopyOut(z, i, k, tileDataMaxNum);
                }
            }
            
            if (lastHoleData > 0) {
                for (int32_t i = 0; i < cycles; i++) {
                    CopyIn(z, i, circulate, lastHoleData);
                    Compute(z, i, circulate, presum, premax, i == 0, lastHoleData);
                    CopyOut(z, i, circulate, lastHoleData);
                }
            }
        }
    } else {
        
        
        if (this->loopCount == 0) return;

        
        for (int32_t z = 0; z < this->loopCount; ++z) {
            
            if (total_k_tasks == 0) continue;

            int32_t tasksPerBlock = total_k_tasks / blockNum;
            int32_t remainderTasks = total_k_tasks % blockNum;
            
            int32_t start_k = tasksPerBlock * blockId + (blockId < remainderTasks ? blockId : remainderTasks);
            int32_t end_k = start_k + (tasksPerBlock + (blockId < remainderTasks ? 1 : 0));

            
            for (int32_t k = start_k; k < end_k; ++k) {
                
                LocalTensor<float> presum = QueueTemp1.Get<float>();
                LocalTensor<float> premax = QueueTemp2.Get<float>();

                
                int32_t currentCopySize = (k < this->circulate) ? this->tileDataMaxNum : this->lastHoleData;

                
                for (int32_t i = 0; i < this->cycles; ++i) {
                    CopyIn(z, i, k, currentCopySize);
                    Compute(z, i, k, presum, premax, i == 0, currentCopySize);
                    CopyOut(z, i, k, currentCopySize);
                }
            }
        }
    }
}
private:
__aicore__ inline void LogcumsumexpComputeCore(
        LocalTensor<float>& y,         
        LocalTensor<float>& x,        
        LocalTensor<float>& presum,   
        LocalTensor<float>& premax,   
        LocalTensor<float>& temp,    
        LocalTensor<float>& tempc,
        const bool isFirst,            
        const int32_t count)
    {
        if (isFirst) {
            Duplicate(presum,1.0f,count);
            Duplicate(tempc,0.0f,count);
            Adds(premax, x, 0.0f, count);
            Adds(y,x,0.0f,count);
            
        }else{
            Adds(temp, premax, 0.0f, count);
        
            Max(premax, premax, x, count);


            
            Sub(temp, temp, premax, count);
            


            Exp(temp, temp, count);

            Mul(presum, presum, temp, count);

            Sub(temp, x, premax, count);

            
            

            Exp(temp, temp, count);
            
            
            Sub(y,temp,tempc,count);
            Add(x,presum,y,count);
            Sub(tempc,x,presum,count);
            Sub(tempc,tempc,y,count);
            Adds(presum,x,0.f,count);
            
            

            Ln(y, presum, count);
            Add(y, y, premax, count);
        }
    }
     __aicore__ inline void CopyIn(int32_t loop_idx, int32_t cycle_idx, int32_t tile_idx, int32_t copySize) {
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        
        uint32_t offset = loop_idx * cycles * interval + cycle_idx * interval + tile_idx * tileDataMaxNum;
        copySize=(copySize*sizeof(T)+31)/32*32/sizeof(T);
        DataCopy(xLocal, xGm[offset], copySize);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t loop_idx, int32_t cycle_idx, int32_t tile_idx,
                                LocalTensor<float>& presum,
                                LocalTensor<float>& premax,
                                bool isFirst, 
                                int32_t count) { 
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        
        auto tempc=tmpBuffer4.Get<float>();
        
        int32_t currentBlockLen = 32 / sizeof(T); 
        int align_cur = ((count + currentBlockLen - 1) / currentBlockLen) * currentBlockLen;
        auto bits=tmpBuffer5.Get<uint8_t>();

        if constexpr (std::is_same_v<T, float>) {
            LocalTensor<float> temp_f = tmpBuffer1.Get<float>();
            LogcumsumexpComputeCore(yLocal, xLocal, presum, premax, temp_f,tempc, isFirst, align_cur);
        } else if constexpr(std::is_same_v<T,half>){
            LocalTensor<float> temp_x_f = tmpBuffer1.Get<float>();
            LocalTensor<float> temp_y_f = tmpBuffer2.Get<float>();
            LocalTensor<float> temp_work_f = tmpBuffer3.Get<float>();
            Cast(temp_x_f, xLocal, RoundMode::CAST_NONE, align_cur);
            LogcumsumexpComputeCore(temp_y_f, temp_x_f, presum, premax, temp_work_f,tempc,isFirst, align_cur);
            constexpr auto rMode =RoundMode::CAST_NONE;

            
            Cast(yLocal, temp_y_f, rMode, align_cur); 
            float val=65504.f;
            CompareScalar(bits,temp_y_f,val,CMPMODE::LE,align_cur);
            half val2=(half)(1.f/0.f);
            Select(yLocal,bits,yLocal,val2,SELMODE::VSEL_TENSOR_SCALAR_MODE,align_cur);

            val=-val;
            val2=(half)(-1.f/0.f);
            CompareScalar(bits,temp_y_f,val,CMPMODE::GE,align_cur);
            Select(yLocal,bits,yLocal,val2,SELMODE::VSEL_TENSOR_SCALAR_MODE,align_cur);

            val2=(half)(0.f/0.f);
            Compare(bits,temp_y_f,temp_y_f,CMPMODE::EQ,align_cur);
            Select(yLocal,bits,yLocal,val2,SELMODE::VSEL_TENSOR_SCALAR_MODE,align_cur);
            
        }
        else{
            LocalTensor<float> temp_x_f = tmpBuffer1.Get<float>();
            LocalTensor<float> temp_y_f = tmpBuffer2.Get<float>();
            LocalTensor<float> temp_work_f = tmpBuffer3.Get<float>();

            Cast(temp_x_f, xLocal, RoundMode::CAST_NONE, align_cur);
            LogcumsumexpComputeCore(temp_y_f, temp_x_f, presum, premax, temp_work_f,tempc, isFirst, align_cur);
            constexpr auto rMode = RoundMode::CAST_RINT;

            
            Cast(yLocal, temp_y_f, rMode, align_cur); 

            float val=1.0f/0.f;
            bfloat16_t val2=ToBfloat16(1.f/0.f);

            CompareScalar(bits,temp_y_f,val,CMPMODE::NE,align_cur);
            Select(yLocal,bits,yLocal,val2,SELMODE::VSEL_TENSOR_SCALAR_MODE,align_cur);

            val=-val;
            val2=ToBfloat16(-1.f/0.f);
            CompareScalar(bits,temp_y_f,val,CMPMODE::NE,align_cur);
            Select(yLocal,bits,yLocal,val2,SELMODE::VSEL_TENSOR_SCALAR_MODE,align_cur);
        }

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }


    __aicore__ inline void CopyOut(int32_t loop_idx, int32_t cycle_idx, int32_t tile_idx, int32_t copySize) {
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        uint32_t offset = loop_idx * cycles * interval + cycle_idx * interval + tile_idx * tileDataMaxNum;

        DataCopyExtParams copyParams={1, static_cast<uint32_t>(copySize * sizeof(T)), 0, 0, 0};

        DataCopyPad(yGm[offset], yLocal, copyParams);
        
        outQueueY.FreeTensor(yLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> QueueTemp1,QueueTemp2,tmpBuffer1,tmpBuffer2,tmpBuffer3,tmpBuffer4,tmpBuffer5;

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;

    int32_t x_ndarray[10];
    int32_t x_dimensional;
    int32_t dim;
    int32_t size;

    int32_t cycles;
    int32_t interval;
    int32_t loopCount;

    int32_t tileDataMaxNum;
    
    int32_t circulate;
    int32_t SingleData;
    int32_t lastHoleData;
    int32_t blocklen=256/sizeof(T);
};

extern "C" __global__ __aicore__ void logcumsumexp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    KernelLogcumsumexp<DTYPE_X> op;
    op.Init(x,  y, 
            tiling_data.size, tiling_data.x_ndarray, tiling_data.x_dimensional,tiling_data.tileDataMaxNum,tiling_data.axis);
    op.Process();
}