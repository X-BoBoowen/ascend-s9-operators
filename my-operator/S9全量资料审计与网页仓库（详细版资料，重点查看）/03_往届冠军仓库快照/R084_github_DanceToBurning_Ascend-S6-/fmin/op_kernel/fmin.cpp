#include <kernel_operator.h>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t chunksize = 4096;

__aicore__ inline void parse_shape(const uint32_t* info, int* shape, int& ndim) {
    ndim = (int)info[0];
    for(int i = 0; i < ndim; ++i) {
        shape[i] = (int)info[i + 1];
        if (shape[i] <= 0) shape[i] = 1;
    }
}

__aicore__ inline void calc_stride(const int* shape, int ndim, int64_t* stride) {
    int64_t current_stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        stride[i] = current_stride;
        if (shape[i] > 0) {
            current_stride *= shape[i];
        }
    }
}




template<typename Dtype, typename OP>
__aicore__ inline void launch_broadcast(
    const uint32_t* x_info,
    const uint32_t* y_info,
    const uint32_t* z_info,
    OP& op) 
{
    const int MAX_DIM = 8;
    int x_shape[MAX_DIM] = {1}, y_shape[MAX_DIM] = {1}, z_shape[MAX_DIM] = {1};
    int64_t x_stride[MAX_DIM] = {0}, y_stride[MAX_DIM] = {0}, z_stride[MAX_DIM] = {0};
    int64_t bcast_x_stride[MAX_DIM] = {0}, bcast_y_stride[MAX_DIM] = {0};
    int ndim = 0, xndim = 0, yndim = 0;

    parse_shape(z_info, z_shape, ndim);
    parse_shape(x_info, x_shape, xndim);
    parse_shape(y_info, y_shape, yndim);

    int64_t total_elements_x = 1;
    for (int d = 0; d < xndim; ++d) {
        total_elements_x *= x_shape[d];
    }
    int64_t total_elements_y = 1;
    for (int d = 0; d < yndim; ++d) {
        total_elements_y *= y_shape[d];
    }

    int aligned_x_shape[MAX_DIM], aligned_y_shape[MAX_DIM];
    int64_t total_elements = 1; 
    for(int i = 0; i < MAX_DIM; ++i) { aligned_x_shape[i] = aligned_y_shape[i] = 1; }
    if (ndim > 0) {
        calc_stride(z_shape, ndim, z_stride);
        for(int d = 0; d < ndim; ++d) {
            total_elements *= z_shape[d];
        }
    } else { total_elements = 1; }
    if (total_elements > 0 && (total_elements_x == 1 || total_elements_y == 1)) {
        int64_t xi = 0; 
        int64_t yi = 0; 
        int64_t zi = 0;
        int64_t blk_x = (total_elements_x == 1) ? 1 : total_elements;
        int64_t blk_y = (total_elements_y == 1) ? 1 : total_elements;
        op.Process(xi, yi, zi, blk_x, blk_y);
        return;
    }

    int x_offset = ndim - xndim;
    int y_offset = ndim - yndim;
    for(int i = 0; i < ndim; ++i) {
        if (i >= x_offset && (i - x_offset) < xndim) aligned_x_shape[i] = x_shape[i - x_offset]; else aligned_x_shape[i] = 1;
        if (i >= y_offset && (i - y_offset) < yndim) aligned_y_shape[i] = y_shape[i - y_offset]; else aligned_y_shape[i] = 1;
    }

    calc_stride(aligned_x_shape, ndim, x_stride);
    calc_stride(aligned_y_shape, ndim, y_stride);
    for(int i=0; i<ndim; ++i) {
        bcast_x_stride[i] = (aligned_x_shape[i] == 1 && z_shape[i] > 1) ? 0 : x_stride[i];
        bcast_y_stride[i] = (aligned_y_shape[i] == 1 && z_shape[i] > 1) ? 0 : y_stride[i];
    }


    bool is_elementwise = (xndim == ndim && yndim == ndim);
    if (is_elementwise) {
        for (int d = 0; d < ndim; ++d) {
            if (aligned_x_shape[d] != z_shape[d] || aligned_y_shape[d] != z_shape[d]) {
                is_elementwise = false;
                break;
            }
        }
        if (is_elementwise && total_elements != total_elements_x) is_elementwise = false;
    }

    if (is_elementwise && total_elements > 0) {
        op.Process(0, 0, 0, total_elements, total_elements); 
    } else if (total_elements >= 0) { 
        int effective_ndim = ndim;
        for (int d = ndim - 1; d >= 0; --d) {
            if (aligned_x_shape[d] == 1 && aligned_y_shape[d] == 1) {
                effective_ndim = d;
            } else {
                break;
            }
        }
        if (effective_ndim < 0) effective_ndim = 0;

        int d_mismatch = -1;
        for (int d = effective_ndim - 1; d >= 0; --d) {
            if (aligned_x_shape[d] != aligned_y_shape[d]) {
                d_mismatch = d;
                break;
            }
        }

        int outer_loop_end_dim; 
        int block_start_dim;    
        int64_t block_lenx = 1;
        int64_t block_leny = 1;
        bool use_special_1N_block = false;

        if (d_mismatch == -1) { 
            outer_loop_end_dim = -1; 
            block_start_dim = 0;  
        } else {
            bool is_innermost_eff_dim = (d_mismatch == effective_ndim - 1);
            bool is_1_vs_N = (aligned_x_shape[d_mismatch] == 1 && aligned_y_shape[d_mismatch] > 1) ||
                             (aligned_x_shape[d_mismatch] > 1 && aligned_y_shape[d_mismatch] == 1);

            if (is_innermost_eff_dim && is_1_vs_N) {
                use_special_1N_block = true;
                outer_loop_end_dim = d_mismatch - 1; 
                block_start_dim = effective_ndim;    
                block_lenx = aligned_x_shape[d_mismatch];
                block_leny = aligned_y_shape[d_mismatch];
            } else {
                outer_loop_end_dim = d_mismatch; 
                block_start_dim = d_mismatch + 1; 
            }
        }

        int64_t total_loop_num = 1;
        for (int d = 0; d <= outer_loop_end_dim; ++d) {
             if(z_shape[d] <= 0) { total_loop_num = 0; break; }
             if (total_loop_num > INT64_MAX / z_shape[d] && z_shape[d] > 1) {
                 total_loop_num = -1; break;
             }
             total_loop_num *= z_shape[d];
        }

        if (!use_special_1N_block) {
             block_lenx = 1;
             block_leny = 1;
             for (int d = block_start_dim; d < effective_ndim; ++d) {
                 if (aligned_x_shape[d] <= 0) { block_lenx = 0; break;}
                 if (block_lenx > INT64_MAX / aligned_x_shape[d] && aligned_x_shape[d] > 1) { block_lenx = -1; break; }
                 block_lenx *= aligned_x_shape[d];

                 if (aligned_y_shape[d] <= 0) { block_leny = 0; break;}
                 if (block_leny > INT64_MAX / aligned_y_shape[d] && aligned_y_shape[d] > 1) { block_leny = -1; break; }
                 block_leny *= aligned_y_shape[d];
             }
             if (block_lenx < 0 || block_leny < 0) return; 
             if (block_lenx <= 0 && effective_ndim >= block_start_dim) block_lenx = 1;
             if (block_leny <= 0 && effective_ndim >= block_start_dim) block_leny = 1;
        }

        if (total_loop_num < 0) { return; } 
        if (total_elements == 0) { total_loop_num = 0; } 
        else if (total_loop_num == 0 && outer_loop_end_dim < effective_ndim -1 && block_lenx > 0 && block_leny > 0) {
             total_loop_num = 1;
        }
         else if (total_loop_num == 0 && total_elements > 0) {
             total_loop_num = 1; 
        }
        int blockDim = GetBlockNum();
        int blockIdx = GetBlockIdx();

        
        int64_t block_len = max(block_lenx, block_leny);
        if (block_len <= 0) { return; } 

        
        int64_t chunks_per_block = (block_len + chunksize - 1) / chunksize;
        
        int64_t total_chunks = total_loop_num * chunks_per_block;
        
        
        for (int64_t flat_idx = blockIdx; flat_idx < total_chunks; flat_idx+=blockDim) {
            
            int64_t loop_idx = flat_idx / chunks_per_block;
            int64_t chunk_idx = flat_idx % chunks_per_block;
            
            int64_t current_idx_for_decomp = loop_idx;
            int64_t xi = 0, yi = 0, zi = 0;
            
            for (int d = outer_loop_end_dim; d >= 0; --d) {
                 if (z_shape[d] <= 0) continue; 
                 int64_t coord = current_idx_for_decomp % z_shape[d];
                 current_idx_for_decomp /= z_shape[d];
                 xi += coord * bcast_x_stride[d];
                 yi += coord * bcast_y_stride[d];
                 zi += coord * z_stride[d];
            }

            
            op.Process_SingleTask(xi, yi, zi, block_lenx, block_leny, chunk_idx);
        }
    }
}




template<typename T>
class KernelFmin {
public:
    __aicore__ inline KernelFmin() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c)
    {
        aGm.SetGlobalBuffer((__gm__ T *)a);
        bGm.SetGlobalBuffer((__gm__ T *)b);
        cGm.SetGlobalBuffer((__gm__ T *)c);
        pipe.InitBuffer(inQueueA, BUFFER_NUM, chunksize * sizeof(T));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, chunksize * sizeof(T));
        pipe.InitBuffer(outQueueC, BUFFER_NUM, chunksize * sizeof(T));
        pipe.InitBuffer(tmpBuffer1, chunksize * sizeof(float)); 
        pipe.InitBuffer(tmpBuffer2, chunksize * sizeof(float)); 
        pipe.InitBuffer(tmpBuffer3, chunksize * sizeof(float));
    }
    __aicore__ inline void Process(int offsetA, int offsetB, int offsetC, int asize, int bsize)
    {
        int blockDim = GetBlockNum();
        int blockIdx = GetBlockIdx();
        int size = max(asize, bsize);
        
        int chunks = (size + 15) / 16;
        int base_chunks = chunks / blockDim;
        int tile_chunks = chunks % blockDim;
        int start = (base_chunks * blockIdx + min(blockIdx, tile_chunks)) * 16;
        int len = min(int(size) - start, int ((base_chunks + (blockIdx < tile_chunks)) * 16));
        
        if (len <= 0) {
            return;
        }

        int loopCount = (len + chunksize - 1) / chunksize;
        
        if (asize == bsize) {
            
            for (int32_t i = 0; i < loopCount; i++) {
                int cur = min(len - i * chunksize, chunksize);
                if (cur <= 0) continue;
                CopyIn(start + i * chunksize + offsetA, start + i * chunksize + offsetB, cur);
                Compute(cur);
                CopyOut(start + i * chunksize + offsetC, cur);
            }
        } 
        else if (asize == 1) {
            
            T valA = aGm.GetValue(offsetA);
            
            for (int32_t i = 0; i < loopCount; i++) {
                int cur = min(len - i * chunksize, chunksize);
                if (cur <= 0) continue;
                CopyInB(start + i * chunksize + offsetB, cur);
                
                ComputeB_Scalar(cur, valA); 
                CopyOut(start + i * chunksize + offsetC, cur);
            }
        } 
        else { 
            
            T valB = bGm.GetValue(offsetB);
            for (int32_t i = 0; i < loopCount; i++) {
                int cur = min(len - i * chunksize, chunksize);
                if (cur <= 0) continue;
                CopyInA(start + i * chunksize + offsetA, cur);
                
                ComputeA(cur, valB);
                CopyOut(start + i * chunksize + offsetC, cur);
            }
        }
    }

    
    
    
     __aicore__ inline void Process_SingleTask(int offsetA, int offsetB, int offsetC, int asize, int bsize, int64_t chunk_idx)
    {
        int len = max(asize, bsize);
        int start = chunk_idx * chunksize;

        
        
        if (start >= len) {
            return;
        }

        int cur = min(len - start, chunksize);
        if (cur <= 0) {
            return;
        }
        
        if (asize == bsize) {
            CopyIn(start + offsetA, start + offsetB, cur);
            Compute(cur);
            CopyOut(start + offsetC, cur);
        } 
        else if (asize == 1) {
            T valA = aGm.GetValue(offsetA);
            CopyInB(start + offsetB, cur);
            ComputeB_Scalar(cur, valA); 
            CopyOut(start + offsetC, cur);
        } 
        else { 
            T valB = bGm.GetValue(offsetB);
            CopyInA(start + offsetA, cur);
            ComputeA(cur, valB);
            CopyOut(start + offsetC, cur);
        }
    }
    
    
    
private:

    __aicore__ inline void CopyIn(int startA, int startB, int cur)
    {
        LocalTensor<T> aLocal = inQueueA.AllocTensor<T>();
        LocalTensor<T> bLocal = inQueueB.AllocTensor<T>();
        cur = ((cur + 15) / 16) * 16;
        DataCopy(aLocal, aGm[startA], cur);
        DataCopy(bLocal, bGm[startB], cur);
        inQueueA.EnQue(aLocal);
        inQueueB.EnQue(bLocal);
    }
    __aicore__ inline void CopyInA(int startA, int cur)
    {
        LocalTensor<T> aLocal = inQueueA.AllocTensor<T>();
        cur = ((cur + 15) / 16) * 16;
        DataCopy(aLocal, aGm[startA], cur);
        inQueueA.EnQue(aLocal);
    }
    __aicore__ inline void CopyInB(int startB, int cur)
    {
        LocalTensor<T> bLocal = inQueueB.AllocTensor<T>();
        cur = ((cur + 15) / 16) * 16;
        DataCopy(bLocal, bGm[startB], cur);
        inQueueB.EnQue(bLocal);
    }
    __aicore__ inline void Compute(int cur)
    {
        uint32_t align_cur = ((cur + blocklen - 1) / blocklen) * blocklen;
        auto x1Local = inQueueA.DeQue<T>();
        auto x2Local = inQueueB.DeQue<T>();
        auto yLocal = outQueueC.AllocTensor<T>();

        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
            
            Min(yLocal, x1Local, x2Local, align_cur);
        } else {
            
            auto floatBuf1 = tmpBuffer1.Get<float>();
            auto floatBuf2 = tmpBuffer2.Get<float>();
            auto floatBuf3 = tmpBuffer3.Get<float>();
            Cast(floatBuf1, x1Local, RoundMode::CAST_NONE, align_cur);
            Cast(floatBuf2, x2Local, RoundMode::CAST_NONE, align_cur);
            Min(floatBuf3, floatBuf1, floatBuf2, align_cur);
            
            constexpr auto rMode = std::is_floating_point_v<T> ? RoundMode::CAST_NONE : RoundMode::CAST_RINT;
            Cast(yLocal, floatBuf3, rMode, align_cur);
        }
        
        outQueueC.EnQue(yLocal);
        inQueueA.FreeTensor(x1Local);
        inQueueB.FreeTensor(x2Local);
    }

    __aicore__ inline void ComputeA(int cur, T val)
    {
        LocalTensor<T> x1Local = inQueueA.DeQue<T>();
        LocalTensor<T> yLocal = outQueueC.AllocTensor<T>();
        uint32_t align_cur = ((cur + blocklen - 1) / blocklen) * blocklen;
        
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
            
            Mins(yLocal, x1Local, val, align_cur);
        } else {
            
            auto tmpBuf1 = tmpBuffer1.Get<float>();
            auto tmpBuf2 = tmpBuffer2.Get<float>();
            Cast(tmpBuf1, x1Local, RoundMode::CAST_NONE, align_cur);
            float value = ToFloat(val);
            
            Mins(tmpBuf2, tmpBuf1, value, align_cur);
            constexpr auto rMode = std::is_floating_point_v<T> ? RoundMode::CAST_NONE : RoundMode::CAST_RINT;
            Cast(yLocal, tmpBuf2, rMode, align_cur);
        }

        outQueueC.EnQue(yLocal);
        inQueueA.FreeTensor(x1Local);
    }
    __aicore__ inline void ComputeB_Scalar(int cur, T val)
    {
        uint32_t align_cur = ((cur + blocklen - 1) / blocklen) * blocklen;
        auto x2Local = inQueueB.DeQue<T>(); 
        auto yLocal = outQueueC.AllocTensor<T>();

        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
            
            Mins(yLocal, x2Local, val, align_cur);
        } else {
            
            auto floatBuf_in = tmpBuffer1.Get<float>();   
            auto floatBuf_out = tmpBuffer2.Get<float>();  
            Cast(floatBuf_in, x2Local, RoundMode::CAST_NONE, align_cur);
            float value = ToFloat(val);
            
            Mins(floatBuf_out, floatBuf_in, value, align_cur);
            
            constexpr auto rMode = std::is_floating_point_v<T> ? RoundMode::CAST_NONE : RoundMode::CAST_RINT;
            Cast(yLocal, floatBuf_out, rMode, align_cur);
        }
        
        outQueueC.EnQue(yLocal);
        inQueueB.FreeTensor(x2Local); 
    }
    
    __aicore__ inline void CopyOut(int start, int cur)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(cur * sizeof(T)), 0, 0, 0};
        LocalTensor<T> cLocal = outQueueC.DeQue<T>();
        DataCopyPad(cGm[start], cLocal, copyParams);
        outQueueC.FreeTensor(cLocal);
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueA;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueB;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueC;
    TBuf<QuePosition::VECCALC> tmpBuffer1,tmpBuffer2,tmpBuffer3;
    GlobalTensor<T> aGm;
    GlobalTensor<T> bGm;
    GlobalTensor<T> cGm;

    uint32_t blocklen = 256 / sizeof(T);
};
 
extern "C" __global__ __aicore__ void fmin(
    GM_ADDR x, GM_ADDR y, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if(tiling_data.xlength == 1 || tiling_data.vlength == 1) {
        if (tiling_data.dt == 0) {
            
            KernelFmin<float> op;
            op.Init(x, y, z);
            op.Process(0, 0, 0, tiling_data.xlength, tiling_data.vlength);
        } else if(tiling_data.dt==1) {
            
            KernelFmin<half> op;
            op.Init(x, y, z);
            op.Process(0, 0, 0, tiling_data.xlength, tiling_data.vlength);
        }else{
            KernelFmin<bfloat16_t> op;
            op.Init(x, y, z);
            op.Process(0, 0, 0, tiling_data.xlength, tiling_data.vlength);
        }
        return;
    }

    const uint32_t* x_info = tiling_data.input_shape_info;
    const uint32_t* y_info = tiling_data.values_shape_info;
    const uint32_t* z_info = tiling_data.output_shape_info;
    if (tiling_data.dt == 0) {
        
        KernelFmin<float> op;
        op.Init(x, y, z);
        launch_broadcast<float>(
            x_info, y_info, z_info, op
        );
    } else if(tiling_data.dt==1) {
        
        KernelFmin<half> op;
        op.Init(x, y, z);
        launch_broadcast<half>(
            x_info, y_info, z_info, op
        );
    }else{
        KernelFmin<bfloat16_t> op;
        op.Init(x, y, z);
        launch_broadcast<bfloat16_t>(
            x_info, y_info, z_info, op
        );
    }
}
 
 #ifndef ASCENDC_CPU_DEBUG
 
 void fmin_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *y, uint8_t *z,
                    uint8_t *workspace, uint8_t *tiling)
 {
     fmin<<<blockDim, l2ctrl, stream>>>(x, y, z, workspace, tiling);
 }
 #endif