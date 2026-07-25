#include "kernel_inc.h"
#include "kernel_operator.h"
// #include "gather_tiling.h"  // 包含host侧的tiling定义

/* ******************************************************************************************************* */
template<class DATA, int tilingKey, class tilingStruct> class DataCopyHelper { };
/* ******************************************************************************************************* */
template<class DATA, class tilingStruct>
class DataCopyHelper<DATA, 2, tilingStruct> {
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> q_local;
    GlobalTensor<DATA> gm_x, gm_y;
    tilingStruct ti;
    __aicore__ inline void CopyIn(uint32_t prefix, uint32_t length) {
        ALLOC(DATA, local);
        DataCopy(local, gm_x[prefix], length);
        ENQUE(DATA, local);
    }
    __aicore__ inline void CopyOut(uint32_t prefix, uint32_t length) {
        DEQUE(DATA, local);
        DataCopy(gm_y[prefix], local, length);
        FREE(local);
    }
public:
    __aicore__ inline DataCopyHelper(GM_ADDR x, GM_ADDR y, const tilingStruct &ti, TPipe *p):ti(ti) {
        this->set_ptr(x, y);
        INIT_QUEUE_2(DATA, q_local, ti.tileLength);
    }
    __aicore__ inline DataCopyHelper(const tilingStruct &ti, TPipe *p):ti(ti) {
        INIT_QUEUE_2(DATA, q_local, ti.tileLength);
    }
    __aicore__ inline void set_ptr(GM_ADDR x, GM_ADDR y) {
        SET_GLOBAL(DATA, x, 0);
        SET_GLOBAL(DATA, y, 0);
    }
    __aicore__ inline void exec(GM_ADDR x, GM_ADDR y) {
        this->set_ptr(x, y);
        this->exec();
    }
    __aicore__ inline void exec() {
        uint32_t i;
        for(i = 0; i < ti.maxLength; i += ti.tileLength) {
            CopyIn(i, ti.tileLength);
            CopyOut(i, ti.tileLength);
        }
        CopyIn(i, ti.reminder);
        CopyOut(i, ti.reminder);
    }
};
/* ******************************************************************************************************* */
template<class DATA, class INDICES, int tilingKey, class tilingStruct> class MyGatherV3;


template<class DATA, class INDICES, class tilingStruct> class MyGatherV3<DATA, INDICES, 1, tilingStruct> {
    GM_ADDR x;
    GM_ADDR indices;
    GM_ADDR y;
    tilingStruct ti;
    
    __aicore__ inline void CopyScalar(__gm__ DATA *src, __gm__ DATA *dst) {
        // 使用host侧tiling中的slice长度
        for(uint32_t i = 0; i < ti.outerDimLength; ++i) {
            *(dst + i) = *(src + i);
        }
    }
    
public:
    __aicore__ inline MyGatherV3( 
        GM_ADDR x, 
        GM_ADDR indices,
        GM_ADDR y,
        const tilingStruct &ti
    ) : x(x), indices(indices), y(y), ti(ti) {}
    
    __aicore__ inline void exec() {
        INDICES index;
        
        // 基址计算
        uint32_t currentBatchOffset = 0;   // 输入基地址
        uint32_t currentBatchIndex = 0;    // 索引基地址
        uint32_t outBaseOffset = 0;
        
        // 直接从host tiling获取维度
        const uint32_t outputSliceStride = ti.outerDimLength;  // sliceLength
        const uint32_t outputInnerStride = ti.indicesLength * outputSliceStride;
        const uint32_t batchLength = ti.batchLength;


        for (uint32_t batch_idx = 0; batch_idx < ti.batchNumber; ++batch_idx) {
            for (uint32_t inner_idx = 0; inner_idx < ti.innerDimLength; ++inner_idx) {
                // 输入内块偏移
                const uint32_t inputInnerOffset = inner_idx * ti.axisDim * outputSliceStride;
                
                for (uint32_t idx = 0; idx < ti.indicesLength; ++idx) {
                    // 读取索引
                    index = *(CAST(__gm__ INDICES *, indices) + currentBatchIndex + idx);
                    
                    // 输入偏移
                    __gm__ DATA* inputPtr = CAST(__gm__ DATA *, this->x);
                    uint32_t inputOffset = currentBatchOffset + 
                                         inputInnerOffset + 
                                         index * outputSliceStride;
                    
                    // 输出偏移
                    __gm__ DATA* outputPtr = CAST(__gm__ DATA *, this->y);
                    uint32_t outputOffset = outBaseOffset + 
                                          (inner_idx * outputInnerStride) + 
                                          (idx * outputSliceStride);
                    
                    // 直接拷贝（使用host提供的维度信息）
                    CopyScalar(
                        inputPtr + inputOffset,
                        outputPtr + outputOffset
                    );
                }
            }
            
            // 更新批次偏移（使用host计算的长度）
            currentBatchOffset += batchLength; 
            currentBatchIndex += ti.indicesLength;
            outBaseOffset += ti.innerDimLength * outputInnerStride;
        }
    }
};

template<class DATA, class INDICES, class tilingStruct>
class MyGatherV3<DATA, INDICES, 0, tilingStruct> {
    GM_ADDR x;
    GM_ADDR indices;
    GM_ADDR y;
    tilingStruct ti;
    DataCopyHelper<DATA, 2, tilingStruct> helper;
    
public:
    __aicore__ inline MyGatherV3( 
        GM_ADDR x, 
        GM_ADDR indices,
        GM_ADDR y,
        const tilingStruct &ti,
        TPipe *p
    ) : x(x), indices(indices), y(y), ti(ti), helper(x,y,ti, p) {} 
    
    __aicore__ inline void exec() {
        INDICES index;
        
        // 使用host侧tiling中的维度字段
        uint32_t currentBatchOffset = 0;   // 当前batch的输入基地址
        uint32_t currentBatchIndex = 0;    // 当前batch的开销索引地址
        uint32_t outBaseOffset = 0; 
        
        // 直接从tilingStruct获取维度信息
        const uint32_t outputSliceStride = ti.outerDimLength;  // 等同于sliceLength
        const uint32_t outputInnerStride = ti.indicesLength * outputSliceStride;
        const uint32_t batchLength = ti.batchLength;
        for (uint32_t batch_idx = 0; batch_idx < ti.batchNumber; ++batch_idx) {
            for (uint32_t inner_idx = 0; inner_idx < ti.innerDimLength; ++inner_idx) {
                // 输入内块偏移 (inner_idx * axisDim * sliceLength)
                const uint32_t inputInnerOffset = inner_idx * ti.axisDim * outputSliceStride;
                
                for (uint32_t idx = 0; idx < ti.indicesLength; ++idx) {
                    // 从索引中读取当前index
                    index = *(CAST(__gm__ INDICES *, indices) + currentBatchIndex + idx);
                    
                    // 输入偏移: batch_base + inner_block + index * inner_block
                    uint32_t inputOffset = currentBatchOffset + 
                                         inputInnerOffset + 
                                         index * outputSliceStride;
                    
                    // 输出偏移: batch_inner_output + inner_pos * indices_count + idx
                    uint32_t outputOffset = outBaseOffset + 
                                          (inner_idx * outputInnerStride) + 
                                          (idx * outputSliceStride);
                    
                    // 执行DMA拷贝（使用host侧tiling中的分块参数）
                    helper.exec(
                        this->x + inputOffset * sizeof(DATA),
                        this->y + outputOffset * sizeof(DATA)
                    );
                    
                    // 保持原有内存屏障
                    PipeBarrier<PIPE_MTE3>();
                }
            }
            
            // 移动到下一批次（使用host侧计算的分批长度）
            currentBatchOffset += batchLength;
            currentBatchIndex += ti.indicesLength;
            outBaseOffset += ti.innerDimLength * outputInnerStride;
        }
    }
};
/* ******************************************************************************************************* */
extern "C" __global__ __aicore__ void gather_v3(
    GM_ADDR x1, 
    GM_ADDR indices, 
    GM_ADDR axis,  // 注意：这里增加axis参数
    GM_ADDR y, 
    GM_ADDR workspace, 
    GM_ADDR tiling) 
{
    // 直接从host侧tiling结构体获取参数
    if(TILING_KEY_IS(0)) {      // DataCopy Ver
        // 使用host侧的GatherTilingDataWithDataCopy结构体
        GET_TILING_DATA_WITH_STRUCT(GatherTilingDataWithDataCopy, ti, tiling);
        TPipe p;
        // 传入axis参数（虽然内核代码未直接使用，但保持接口一致）
        MyGatherV3<DTYPE_Y, DTYPE_INDICES, 0, GatherTilingDataWithDataCopy>(x1, indices, y, ti, &p).exec();
    }
    else if(TILING_KEY_IS(1)) { // ScalarCopy Ver
        // 使用host侧的GatherTilingDataScalarCopy结构体
        GET_TILING_DATA_WITH_STRUCT(GatherTilingDataScalarCopy, ti, tiling);
        MyGatherV3<DTYPE_Y, DTYPE_INDICES, 1, GatherTilingDataScalarCopy>(x1, indices, y, ti).exec();
    }
}