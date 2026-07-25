#include "kernel_operator.h"

using namespace AscendC; 

constexpr int32_t MAX_DIM = 8;

__aicore__ inline uint32_t min_u32(uint32_t a, uint32_t b) {
    return (a < b) ? a : b;
}

__aicore__ inline void CalculateElementStrides(
    const uint32_t* shape_info, // 输入: [ndim, d0, d1, ...]
    uint32_t ndim,              // 输入: 维度数
    uint32_t* strides)          // 输出: 存储计算出的元素步长 (uint32_t)
{
    if (ndim == 0) return;
    strides[ndim - 1] = 1; // 最内维 stride 为 1
    for (int32_t i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape_info[i + 1 + 1]; // stride[i] = stride[i+1] * shape[i+1]
    }
}

__aicore__ inline void CalculateGmOffsetsAndStrides(
    uint32_t iter,                  // 输入: 当前 1D 切片的线性索引 (0-based)
    uint32_t scatter_dim,           // 输入: scatter 操作的维度
    const uint32_t* self_shape_info, // 输入: self/y 的形状信息 [ndim, d0, ...]
    const uint32_t* src_shape_info,  // 输入: src/index 的形状信息 [ndim, d0, ...]
    uint32_t self_dtype_size,       // 输入: self/y 数据类型大小 (字节)
    uint32_t src_dtype_size,        // 输入: src 数据类型大小 (字节)
    uint32_t index_dtype_size,      // 输入: index 数据类型大小 (字节)
    // 输出参数 (指针传递，用于修改外部变量)
    uint32_t* self_offset_bytes,
    uint32_t* index_offset_bytes,
    uint32_t* src_offset_bytes,
    uint32_t* y_offset_bytes,
    uint32_t* src_stride_bytes,      // src 在 scatter_dim 上的字节步长
    uint32_t* index_stride_bytes,    // index 在 scatter_dim 上的字节步长
    uint32_t* self_y_stride_bytes)   // self/y 在 scatter_dim 上的字节步长
{
    uint32_t ndim = self_shape_info[0];
    uint32_t src_ndim = src_shape_info[0]; // 获取 src 的维度数

    // 基本检查
    // if (ndim == 0 || src_ndim == 0 || scatter_dim >= ndim || scatter_dim >= src_ndim) {
    //     *self_offset_bytes = *index_offset_bytes = *src_offset_bytes = *y_offset_bytes = 0;
    //     *src_stride_bytes = *index_stride_bytes = *self_y_stride_bytes = 0;
    //     return;
    // }

    uint32_t self_y_elem_strides[MAX_DIM];
    uint32_t src_index_elem_strides[MAX_DIM];
    CalculateElementStrides(self_shape_info, ndim, self_y_elem_strides);
    CalculateElementStrides(src_shape_info, src_ndim, src_index_elem_strides); 

    // 2. 计算沿 scatter_dim 的字节步长
    uint32_t elem_stride_along_scatter_dim_self_y = self_y_elem_strides[scatter_dim];
    uint32_t elem_stride_along_scatter_dim_src_index = src_index_elem_strides[scatter_dim];

    *self_y_stride_bytes = elem_stride_along_scatter_dim_self_y * self_dtype_size;
    *src_stride_bytes = elem_stride_along_scatter_dim_src_index * src_dtype_size;
    *index_stride_bytes = elem_stride_along_scatter_dim_src_index * index_dtype_size;

    // 3. 将线性 iter 映射回多维坐标 (只考虑非 scatter 维度)
    uint32_t current_iter = iter;
    uint32_t base_offset_self_y_elem = 0;
    uint32_t base_offset_src_index_elem = 0;

    // 从最高维开始分解 iter (不包括 scatter_dim)
    for (uint32_t d = 0; d < ndim; ++d) {
        if (d == scatter_dim) continue; // 跳过 scatter 维度

        // 计算当前维度 'd' 之前的非 scatter 维度的总大小 (逻辑步长)
        uint32_t logical_stride = 1;
        for (uint32_t inner_d = d + 1; inner_d < ndim; ++inner_d) {
            if (inner_d == scatter_dim) continue;
             // 检查溢出（简化检查）
             uint32_t dim_size_inner = self_shape_info[inner_d + 1];
             if (dim_size_inner == 0) { logical_stride = 0; break; } 
            logical_stride *= dim_size_inner;
        }

        uint32_t coord = 0;
        if (logical_stride > 0) {
            coord = current_iter / logical_stride;
            current_iter %= logical_stride;
        } else if (current_iter == 0) {
            coord = 0; // If stride is 0, coord is 0 if current_iter is 0
        } else {
            base_offset_self_y_elem = 0;
            base_offset_src_index_elem = 0;
            break;
        }


        base_offset_self_y_elem += coord * self_y_elem_strides[d];
        base_offset_src_index_elem += coord * src_index_elem_strides[d];
    }

    *self_offset_bytes = base_offset_self_y_elem * self_dtype_size;
    *y_offset_bytes = base_offset_self_y_elem * self_dtype_size; // y 和 self 偏移相同
    *src_offset_bytes = base_offset_src_index_elem * src_dtype_size;
    *index_offset_bytes = base_offset_src_index_elem * index_dtype_size;
}

template<typename T>
class scatter_reduce_1d {
public:
    __aicore__ inline scatter_reduce_1d() {}
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR src, GM_ADDR y, uint32_t self_M, uint32_t src_M, uint32_t include_self, uint32_t reduce_code)
    {
        self_Gm.SetGlobalBuffer((__gm__ T *)self);
        index_Gm.SetGlobalBuffer((__gm__ int32_t *)index);
        src_Gm.SetGlobalBuffer((__gm__ T *)src);
        y_Gm.SetGlobalBuffer((__gm__ T *)y);
        pipe.InitBuffer(selfQueue, 1, 8192 * 4);
        pipe.InitBuffer(indexQueue, 1, 8192 * 4);
        pipe.InitBuffer(srcQueue, 1, 8192 * 4);
        pipe.InitBuffer(yQueue, 1, 8192 * 4);
        pipe.InitBuffer(workQueue, 8192 * 4); 
        this->self_M = self_M;
        this->src_M = src_M;
        this->include_self = include_self;
        this->reduce_code = reduce_code;
    }
    __aicore__ inline void Process(uint32_t selfoffset, uint32_t indexoffset, uint32_t srcoffset, uint32_t selfstride, uint32_t indexstride, uint32_t srcstride)
    {
        // AscendC::PRINTF("Debug Offsets/Strides: self_off=%u, idx_off=%u, src_off=%u | self_stride=%u, idx_stride=%u, src_stride=%u\n",
        //     selfoffset,
        //     indexoffset,
        //     srcoffset,
        //     selfstride,
        //     indexstride,
        //     srcstride);
        {
            if(selfstride == sizeof(T))
            {
                LocalTensor<T> srcLocal = srcQueue.AllocTensor<T>();
                LocalTensor<int32_t> indexLocal = indexQueue.AllocTensor<int32_t>();
                LocalTensor<T> selfLocal = selfQueue.AllocTensor<T>();
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(self_M * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
                DataCopyPad(selfLocal, self_Gm[selfoffset/sizeof(T)], copyParams, padParams);
                copyParams.blockLen = src_M * sizeof(T);
                DataCopyPad(srcLocal, src_Gm[srcoffset/sizeof(T)], copyParams, padParams);
                DataCopyPadExtParams<int32_t> indexpadParams{true, 0, 0, 0};
                copyParams.blockLen = src_M * sizeof(int32_t);
                DataCopyPad(indexLocal, index_Gm[indexoffset/sizeof(int32_t)], copyParams, indexpadParams);
                srcQueue.EnQue(srcLocal);
                indexQueue.EnQue(indexLocal);
                selfQueue.EnQue(selfLocal);
            } else {
                LocalTensor<T> srcLocal = srcQueue.AllocTensor<T>();
                LocalTensor<int32_t> indexLocal = indexQueue.AllocTensor<int32_t>();
                LocalTensor<T> selfLocal = selfQueue.AllocTensor<T>();
                DataCopyExtParams copyParams{static_cast<uint16_t>(self_M), sizeof(T), selfstride - static_cast<uint32_t>(sizeof(T)), 0, 0};
                DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
                DataCopyPad(selfLocal, self_Gm[selfoffset/sizeof(T)], copyParams, padParams);
                copyParams.blockCount = src_M;
                copyParams.blockLen = sizeof(T);
                copyParams.srcStride = srcstride - static_cast<uint32_t>(sizeof(T));
                DataCopyPad(srcLocal, src_Gm[srcoffset/sizeof(T)], copyParams, padParams);
                DataCopyPadExtParams<int32_t> indexpadParams{true, 0, 0, 0};
                copyParams.blockLen = 4;
                copyParams.srcStride = indexstride - static_cast<uint32_t>(4);
                DataCopyPad(indexLocal, index_Gm[indexoffset/sizeof(int32_t)], copyParams, indexpadParams);
                srcQueue.EnQue(srcLocal);
                indexQueue.EnQue(indexLocal);
                selfQueue.EnQue(selfLocal);
            }
        }
        {
            {
                LocalTensor<T> srcLocal = srcQueue.DeQue<T>();
                LocalTensor<int32_t> indexLocal = indexQueue.DeQue<int32_t>();
                LocalTensor<T> selfLocal = selfQueue.DeQue<T>();
                LocalTensor<T> yLocal = yQueue.AllocTensor<T>();
                // DumpTensor(indexLocal,0,16);
                // DumpTensor(selfLocal,0,16);
                {
                    uint64_t mask = 256 / sizeof(T);
                    uint8_t repeatTimes; 
                    if(selfstride == sizeof(T)) {
                        repeatTimes = (self_M + mask - 1) / mask;
                    } else {
                        repeatTimes = (self_M*chunklen + mask - 1) / mask;
                    }
                    // PRINTF("mask %lld repeat %d\n", mask, repeatTimes);
                    Copy(yLocal, selfLocal, mask, repeatTimes, { 1, 1, 8, 8 });
                    // DumpTensor(self_Gm[selfoffset/sizeof(T)],0,32);
                    // DumpTensor(selfLocal,1,32);
                    // DumpTensor(yLocal,2,32);
                }
                if(reduce_code == 0) {
                    if(include_self == 1) {
                        // PRINTF("here self\n");
                        if(selfstride == sizeof(T)) {
                            for(int i = 0; i != src_M; i++) {
                                int index = indexLocal.GetValue(i);
                                T val = srcLocal.GetValue(i);
                                yLocal.SetValue(index, static_cast<T>(static_cast<float>(yLocal.GetValue(index)) + static_cast<float>(val)));
                            }
                        } else {
                            for(int i = 0; i != src_M; i++) {
                                int index = indexLocal.GetValue(i*8);
                                T val = srcLocal.GetValue(i*chunklen);
                                yLocal.SetValue(index*chunklen, static_cast<T>(static_cast<float>(yLocal.GetValue(index*chunklen)) + static_cast<float>(val)));
                            }
                        }
                    } else {
                        LocalTensor<T> workLocal = workQueue.Get<T>();
                        Duplicate(workLocal, static_cast<T>(0), 8192 * 4 / sizeof(T));
                        if(selfstride == sizeof(T)) {
                            for(int i = 0; i != src_M; i++) {
                                int index = indexLocal.GetValue(i);
                                T val = srcLocal.GetValue(i);
                                if(static_cast<float>(workLocal.GetValue(index)) == 0){
                                    yLocal.SetValue(index, val);
                                    workLocal.SetValue(index, 1);
                                    // PRINTF("%f \n", static_cast<float>(workLocal.GetValue(index)));
                                }
                                else {
                                    // PRINTF("here\n");
                                    yLocal.SetValue(index, static_cast<T>(static_cast<float>(yLocal.GetValue(index)) + static_cast<float>(val)));
                                }
                            }
                        } else {
                            for(int i = 0; i != src_M; i++) {
                                int index = indexLocal.GetValue(i*8);
                                T val = srcLocal.GetValue(i*chunklen);
                                if(static_cast<float>(workLocal.GetValue(index*chunklen)) == 0) {
                                    yLocal.SetValue(index*chunklen, val);
                                    workLocal.SetValue(index*chunklen, 1);
                                    // PRINTF("%f \n", static_cast<float>(workLocal.GetValue(index*chunklen)));
                                }
                                else {
                                    // PRINTF("here\n");
                                    yLocal.SetValue(index*chunklen, static_cast<T>(static_cast<float>(yLocal.GetValue(index*chunklen)) + static_cast<float>(val)));
                                }
                            }
                        }
                    }
                } else if(reduce_code == 1) {
                    if(include_self == 1) {
                        if(selfstride == sizeof(T)) {
                            for(int i = 0; i != src_M; i++) {
                                int index = indexLocal.GetValue(i);
                                T val = srcLocal.GetValue(i);
                                yLocal.SetValue(index, static_cast<T>(static_cast<float>(yLocal.GetValue(index)) * static_cast<float>(val)));
                            }
                        } else {
                            for(int i = 0; i != src_M; i++) {
                                int index = indexLocal.GetValue(i*8);
                                T val = srcLocal.GetValue(i*chunklen);
                                yLocal.SetValue(index*chunklen, static_cast<T>(static_cast<float>(yLocal.GetValue(index*chunklen)) * static_cast<float>(val)));
                            }
                        }
                    } else {
                        LocalTensor<T> workLocal = workQueue.Get<T>();
                        Duplicate(workLocal, static_cast<T>(0), 8192 * 4 / sizeof(T));
                        if(selfstride == sizeof(T)) {
                            for(int i = 0; i != src_M; i++) {
                                int index = indexLocal.GetValue(i);
                                T val = srcLocal.GetValue(i);
                                if(static_cast<float>(workLocal.GetValue(index)) == 0){
                                    yLocal.SetValue(index, val);
                                    workLocal.SetValue(index, 1);
                                }
                                else {
                                    yLocal.SetValue(index, static_cast<T>(static_cast<float>(yLocal.GetValue(index)) * static_cast<float>(val)));
                                }
                            }
                        } else {
                            for(int i = 0; i != src_M; i++) {
                                int index = indexLocal.GetValue(i*8);
                                T val = srcLocal.GetValue(i*chunklen);
                                if(static_cast<float>(workLocal.GetValue(index*chunklen)) == 0) {
                                    yLocal.SetValue(index*chunklen, val);
                                    workLocal.SetValue(index*chunklen, 1);
                                }
                                else {
                                    yLocal.SetValue(index*chunklen, static_cast<T>(static_cast<float>(yLocal.GetValue(index*chunklen)) * static_cast<float>(val)));
                                }
                            }
                        }
                    }
                } else if(reduce_code == 2) {
                    LocalTensor<T> workLocal = workQueue.Get<T>();
                    if(include_self == 0)
                        Duplicate(workLocal, static_cast<T>(0), 8192 * 4 / sizeof(T));
                    else 
                        Duplicate(workLocal, static_cast<T>(1), 8192 * 4 / sizeof(T));
                    if(include_self == 1) {
                        if(selfstride == sizeof(T)) {
                            for(int i = 0; i != src_M; i++) {
                                int index = indexLocal.GetValue(i);
                                T val = srcLocal.GetValue(i);
                                yLocal.SetValue(index, static_cast<T>(static_cast<float>(yLocal.GetValue(index)) + static_cast<float>(val)));
                                workLocal.SetValue(index, static_cast<T>(static_cast<float>(workLocal.GetValue(index)) + static_cast<float>(1)));
                            }
                        } else {
                            for(int i = 0; i != src_M; i++) {
                                int index = indexLocal.GetValue(i*8);
                                T val = srcLocal.GetValue(i*chunklen);
                                yLocal.SetValue(index*chunklen, static_cast<T>(static_cast<float>(yLocal.GetValue(index*chunklen)) + static_cast<float>(val)));
                                workLocal.SetValue(index*chunklen, static_cast<T>(static_cast<float>(workLocal.GetValue(index*chunklen)) + static_cast<float>(1)));
                            }
                        }
                    } else {
                        if(selfstride == sizeof(T)) {
                            for(int i = 0; i != src_M; i++) {
                                int index = indexLocal.GetValue(i);
                                T val = srcLocal.GetValue(i);
                                if(static_cast<float>(workLocal.GetValue(index)) == 0){
                                    yLocal.SetValue(index, val);
                                    workLocal.SetValue(index, 1);
                                }
                                else {
                                    yLocal.SetValue(index, static_cast<T>(static_cast<float>(yLocal.GetValue(index)) + static_cast<float>(val)));
                                    workLocal.SetValue(index, static_cast<T>(static_cast<float>(workLocal.GetValue(index)) + static_cast<float>(1)));
                                }
                            }
                        } else {
                            for(int i = 0; i != src_M; i++) {
                                int index = indexLocal.GetValue(i*8);
                                T val = srcLocal.GetValue(i*chunklen);
                                if(static_cast<float>(workLocal.GetValue(index*chunklen)) == 0) {
                                    yLocal.SetValue(index*chunklen, val);
                                    workLocal.SetValue(index*chunklen, 1);
                                }
                                else {
                                    yLocal.SetValue(index*chunklen, static_cast<T>(static_cast<float>(yLocal.GetValue(index*chunklen)) + static_cast<float>(val)));
                                    workLocal.SetValue(index*chunklen, static_cast<T>(static_cast<float>(workLocal.GetValue(index*chunklen)) + static_cast<float>(1)));
                                }
                            }
                        }
                    }
                    if(selfstride == sizeof(T)) {
                        for(int i = 0; i != self_M; i++) {
                            T index = workLocal.GetValue(i);
                            if(static_cast<float>(index) != 0) {
                                yLocal.SetValue(i, static_cast<T>(static_cast<float>(yLocal.GetValue(i)) / static_cast<float>(index)));
                            }
                        }
                    } else {
                        for(int i = 0; i != self_M; i++) {
                            T index = workLocal.GetValue(i*chunklen);
                            if(static_cast<float>(index) != 0) {
                                yLocal.SetValue(i*chunklen, static_cast<T>(static_cast<float>(yLocal.GetValue(i*chunklen)) / static_cast<float>(index)));
                            }
                        }
                    }
                } else if(reduce_code == 3) {
                    // PRINTF("here\n");
                    LocalTensor<T> workLocal = workQueue.Get<T>();
                    Duplicate(workLocal, static_cast<T>(0), 8192 * 4 / sizeof(T));
                    if(selfstride == sizeof(T)) {
                        for(int i = 0; i != src_M; i++) {
                            int index = indexLocal.GetValue(i);
                            T val = srcLocal.GetValue(i);
                            if(include_self == 0 && static_cast<float>(workLocal.GetValue(index)) == 0) {
                                yLocal.SetValue(index, val);
                                workLocal.SetValue(index, 1);
                                // PRINTF("%d %f %f\n",index, static_cast<float>(val), val);
                            } else {
                                T val1 = yLocal.GetValue(index);
                                // PRINTF("%d %f %f \n",index,static_cast<float>(val),static_cast<float>(val1));
                                yLocal.SetValue(index, static_cast<float>(val) < static_cast<float>(val1) ? val1 : val);
                                // PRINTF("%d %f\n",index, static_cast<float>(yLocal.GetValue(index)));
                            }
                        }
                    } else {
                        for(int i = 0; i != src_M; i++) {
                            int index = indexLocal.GetValue(i*8);
                            T val = srcLocal.GetValue(i*chunklen);
                            // PRINTF("offset %d index %d val %f yvalue %f\n",selfoffset/sizeof(T) + index, index, val, yLocal.GetValue(i*chunklen));
                            if(include_self == 0 && static_cast<float>(workLocal.GetValue(index*chunklen)) == 0) {
                                yLocal.SetValue(index*chunklen, val);
                                workLocal.SetValue(index*chunklen, 1);
                            } else {
                                T val1 = yLocal.GetValue(index*chunklen);
                                val1 = static_cast<float>(val) < static_cast<float>(val1) ? val1 : val;
                                yLocal.SetValue(index*chunklen, val1);
                                // PRINTF("val1 %f val %f index %d \n",val1,val,index*chunklen);
                            }
                            // DumpTensor(yLocal[64], 3, 32);
                        }
                    }
                } else {
                    LocalTensor<T> workLocal = workQueue.Get<T>();
                    Duplicate(workLocal, static_cast<T>(0), 8192 * 4 / sizeof(T));
                    if(selfstride == sizeof(T)) {
                        for(int i = 0; i != src_M; i++) {
                            int index = indexLocal.GetValue(i);
                            T val = srcLocal.GetValue(i);
                            if(include_self == 0 && static_cast<float>(workLocal.GetValue(index)) == 0) {
                                yLocal.SetValue(index, val);
                                workLocal.SetValue(index, 1);
                            } else {
                                T val1 = yLocal.GetValue(index);
                                yLocal.SetValue(index, static_cast<float>(val) > static_cast<float>(val1) ? val1 : val);
                            }
                        }
                    } else {
                        for(int i = 0; i != src_M; i++) {
                            int index = indexLocal.GetValue(i*8);
                            T val = srcLocal.GetValue(i*chunklen);
                            if(include_self == 0 && static_cast<float>(workLocal.GetValue(index*chunklen)) == 0) {
                                yLocal.SetValue(index*chunklen, val);
                                workLocal.SetValue(index*chunklen, 1);
                            } else {
                                T val1 = yLocal.GetValue(index*chunklen);
                                yLocal.SetValue(index*chunklen, static_cast<float>(val) > static_cast<float>(val1) ? val1 : val);
                            }
                        }
                    }
                }
                yQueue.EnQue(yLocal);
                srcQueue.FreeTensor(srcLocal);
                indexQueue.FreeTensor(indexLocal);
                selfQueue.FreeTensor(selfLocal);
            }
        }
        {
            // int32_t eventIDSToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
            // AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);
            // AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);
            LocalTensor<T> yLocal = yQueue.DeQue<T>();
            // DumpTensor(yLocal[64], 3, 16);
            if(selfstride == sizeof(T)) {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(self_M * sizeof(T)), 0, 0, 0};
                DataCopyPad(y_Gm[selfoffset/sizeof(T)], yLocal, copyParams);
                yQueue.FreeTensor(yLocal);
            } else {
                DataCopyExtParams copyParams{static_cast<uint16_t>(self_M), sizeof(T), 0, selfstride - static_cast<uint32_t>(sizeof(T)), 0};
                DataCopyPad(y_Gm[selfoffset/sizeof(T)], yLocal, copyParams);
                yQueue.FreeTensor(yLocal);
            }
            // DumpTensor(y_Gm, 0, 16);
        }
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> selfQueue;
    TQue<QuePosition::VECIN, 1> indexQueue;
    TQue<QuePosition::VECIN, 1> srcQueue;
    TQue<QuePosition::VECOUT, 1> yQueue;
    TBuf<QuePosition::VECCALC> workQueue;
    GlobalTensor<T> self_Gm;
    GlobalTensor<T> src_Gm;
    GlobalTensor<T> y_Gm;
    GlobalTensor<int32_t> index_Gm;

    uint32_t blocklen = 256 / sizeof(T);
    uint32_t chunklen = 32 / sizeof(T);
    uint32_t src_M;
    uint32_t self_M;
    uint32_t include_self;
    uint32_t reduce_code;
};

extern "C" __global__ __aicore__ void scatter_reduce(
    GM_ADDR self, GM_ADDR index, GM_ADDR src, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {

    GET_TILING_DATA(tiling_data, tiling);       // 获取 Tiling 数据
    int blockIdx = GetBlockIdx();         // 当前核 ID
    int blockNum = GetBlockNum();         // 总核数

    // 解析 Tiling 数据
    uint32_t scatter_dim = tiling_data.dim;
    uint32_t reduce_code = tiling_data.reduce_code;
    uint32_t include_self = tiling_data.include_self_flag;
    uint32_t dtype_flag = tiling_data.dtype_flag; // 0: float32, 1: float16
    const uint32_t* self_shape_info = tiling_data.self_shape_info;
    const uint32_t* src_shape_info = tiling_data.src_shape_info; // 获取 src shape
    uint32_t ndim = self_shape_info[0];
    uint32_t src_ndim = src_shape_info[0];

    // --- 2. 计算总迭代次数和当前核负责的范围 (使用 uint32_t) ---
    uint32_t total_iterations = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        if (i == scatter_dim) continue;
        uint32_t dim_size = self_shape_info[i + 1];
        total_iterations *= dim_size;
    }

    // 任务分配
    uint32_t tasks_per_core = total_iterations / blockNum;
    uint32_t tasks_remainder = total_iterations % blockNum;
    uint32_t start_iter = blockIdx * tasks_per_core + min_u32(blockIdx, tasks_remainder);
    uint32_t end_iter = start_iter + tasks_per_core + ((block_idx < tasks_remainder) ? 1 : 0);

    // 切片长度 M 和 输出维度大小
    uint32_t length_M = src_shape_info[scatter_dim + 1]; // M 由 src/index 决定
    uint32_t output_dim_size = self_shape_info[scatter_dim + 1]; // 输出大小由 self/y 决定

    // --- 3. 循环处理分配到的切片 ---
    uint32_t self_dtype_size = (dtype_flag == 1) ? sizeof(half) : sizeof(float);
    uint32_t src_dtype_size = self_dtype_size; // 假设 src 和 self/y 类型一致
    uint32_t index_dtype_size = sizeof(int32_t);

    if(dtype_flag == 0) {
        scatter_reduce_1d<float> op;
        op.Init(self, index, src, y, output_dim_size, length_M, include_self, reduce_code);
        for (uint32_t iter = start_iter; iter < end_iter; ++iter) {
            // --- 计算当前 iter 对应的 GM 偏移量和步长 ---
            uint32_t self_offset_bytes, index_offset_bytes, src_offset_bytes, y_offset_bytes;
            uint32_t src_stride_bytes, index_stride_bytes, self_y_stride_bytes;
            
            CalculateGmOffsetsAndStrides(
                iter, scatter_dim, self_shape_info, src_shape_info,
                self_dtype_size, src_dtype_size, index_dtype_size,
                &self_offset_bytes, &index_offset_bytes, &src_offset_bytes, &y_offset_bytes,
                &src_stride_bytes, &index_stride_bytes, &self_y_stride_bytes);
            op.Process(self_offset_bytes, index_offset_bytes, src_offset_bytes, self_y_stride_bytes,
                index_stride_bytes, src_stride_bytes);
        }
    } else {
        scatter_reduce_1d<half> op;
        op.Init(self, index, src, y, output_dim_size, length_M, include_self, reduce_code);
        for (uint32_t iter = start_iter; iter < end_iter; ++iter) {
            // --- 计算当前 iter 对应的 GM 偏移量和步长 ---
            uint32_t self_offset_bytes, index_offset_bytes, src_offset_bytes, y_offset_bytes;
            uint32_t src_stride_bytes, index_stride_bytes, self_y_stride_bytes;
    
            CalculateGmOffsetsAndStrides(
                iter, scatter_dim, self_shape_info, src_shape_info,
                self_dtype_size, src_dtype_size, index_dtype_size,
                &self_offset_bytes, &index_offset_bytes, &src_offset_bytes, &y_offset_bytes,
                &src_stride_bytes, &index_stride_bytes, &self_y_stride_bytes);
            op.Process(self_offset_bytes, index_offset_bytes, src_offset_bytes, self_y_stride_bytes,
                index_stride_bytes, src_stride_bytes);
        }
    }
}

