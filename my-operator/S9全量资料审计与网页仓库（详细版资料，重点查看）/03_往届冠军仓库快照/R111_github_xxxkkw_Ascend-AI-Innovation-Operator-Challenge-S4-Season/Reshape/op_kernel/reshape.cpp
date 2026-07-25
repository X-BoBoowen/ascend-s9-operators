#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;  

class KernelReshape{ 
    public:
        __aicore__ inline KernelReshape(){}
        __aicore__ inline void InitTiling(GM_ADDR tiling) {
            GET_TILING_DATA(tiling_data, tiling); 
            totalLength = tiling_data.totalLength;
            tileLength = tiling_data.tileLength;
            loopCount = tiling_data.loopCount;
            leftNum = tiling_data.leftNum;
            shapeLength = tiling_data.shapeLength;
            axis = tiling_data.axis;
            num_axes = tiling_data.num_axes;
        }
        __aicore__ inline void Init(GM_ADDR x,GM_ADDR shape, GM_ADDR y,GM_ADDR tiling,TPipe* pipeIn){
            InitTiling(tiling);
           
            xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x,this->totalLength);
            sGm.SetGlobalBuffer((__gm__ DTYPE_SHAPE*)shape,this->shapeLength);
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y,this->totalLength);
            pipe = pipeIn;
            pipe->InitBuffer(queBind, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        }
        __aicore__ inline void Process() {
            for(uint32_t i = 0; i < loopCount; i ++) {
                Solve(i,this->tileLength);
            }
            if(this->leftNum > 0){
                Solve(this->loopCount,this->leftNum);
            }
        }  
        __aicore__ inline void Solve(uint32_t progress,uint32_t length){
            auto bindLocal = queBind.AllocTensor<DTYPE_X>();
            DataCopy(bindLocal,xGm[progress * this->tileLength],length);
            queBind.EnQue(bindLocal);
            bindLocal = queBind.DeQue<DTYPE_X>();
            DataCopy(yGm[progress * this->tileLength],bindLocal,length);
            queBind.FreeTensor(bindLocal);
        }
    private:
        TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> queBind;
        GlobalTensor<DTYPE_X> xGm;
        GlobalTensor<DTYPE_SHAPE> sGm;
        GlobalTensor<DTYPE_Y> yGm;
        TPipe* pipe;
        uint32_t totalLength;   
        uint32_t tileLength;
        uint32_t loopCount;
        uint32_t leftNum;
        uint32_t shapeLength;
        int32_t axis;
        int32_t num_axes;
    
    };
extern "C" __global__ __aicore__ void reshape(GM_ADDR x, GM_ADDR shape, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    KernelReshape op;
    op.Init(x,shape,y,tiling,&pipe);
    op.Process();
}
