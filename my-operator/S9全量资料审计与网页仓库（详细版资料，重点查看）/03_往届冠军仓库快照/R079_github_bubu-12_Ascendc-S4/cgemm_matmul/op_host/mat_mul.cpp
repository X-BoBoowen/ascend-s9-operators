
#include "mat_mul_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#ifndef SIZE_MAX
#define SIZE_MAX (~(size_t)0)
#endif

namespace optiling {

    const size_t COMPLEX64_SIZE = 8;
    const size_t WORKSPACE_ALIGNMENT = 128;
    const size_t EXTRA_FIXED_WORKSPACE = 16 * 1024 * 1024;
    const size_t FLOAT_SIZE = 4;  
    
    static bool EncodeShape(const gert::Shape *shape, uint32_t arr[8]) {
        if (shape == nullptr) {
            arr[0] = 0;
            for (size_t i = 1; i < 8; ++i) arr[i] = 1;
            return false;
        }
        size_t nd = shape->GetDimNum();
        // std::cout << "nd " << nd << std::endl;
        if(nd == 0) return false;
        arr[0] = static_cast<uint32_t>((nd < 7) ? nd : 7); 
        for (size_t i = 0; i < arr[0]; ++i) {
            int64_t dim_val = shape->GetDim(i);
            arr[i + 1] = (dim_val >= 0) ? static_cast<uint32_t>(dim_val) : 0; 
        }
        for (size_t i = arr[0] + 1; i < 8; ++i) {
            arr[i] = 1; 
        }
        return true;
    }
    
    static inline size_t AlignUp(size_t value, size_t alignment) {
            if (alignment == 0) return value; 
            return (value + alignment - 1) / alignment * alignment;
    }
    
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
        context->SetBlockDim(aicNum); 
        // std::cout << aicNum << std::endl;
        const auto &x_shape = context->GetInputShape(0)->GetOriginShape();
        const auto &y_shape = context->GetInputShape(1)->GetOriginShape();
        const auto &actual_bias_shape_to_encode = context->GetInputShape(2)->GetOriginShape();
        const auto &z_shape = context->GetOutputShape(0)->GetOriginShape();
    
        size_t x_orig_ndim = x_shape.GetDimNum();
        size_t y_orig_ndim = y_shape.GetDimNum();
    
        int64_t x_kernel_dims[8], y_kernel_dims[8];
        uint32_t x_kernel_ndim = 0, y_kernel_ndim = 0;
    
        if (x_orig_ndim == 1 && y_orig_ndim == 1) {
            int64_t dim0_x = x_shape.GetDim(0);
            int64_t dim0_y = y_shape.GetDim(0);
            if (dim0_x != dim0_y || dim0_x <= 0) return ge::GRAPH_FAILED;
            x_kernel_dims[0] = 1; x_kernel_dims[1] = dim0_x; x_kernel_ndim = 2;
            y_kernel_dims[0] = dim0_y; y_kernel_dims[1] = 1; y_kernel_ndim = 2;
        } else if (x_orig_ndim == 2 && y_orig_ndim == 2) {
            x_kernel_dims[0] = 1;
            x_kernel_dims[1] = x_shape.GetDim(0);
            x_kernel_dims[2] = x_shape.GetDim(1);
            x_kernel_ndim = 3;
    
            y_kernel_dims[0] = 1;
            y_kernel_dims[1] = y_shape.GetDim(0);
            y_kernel_dims[2] = y_shape.GetDim(1);
            y_kernel_ndim = 3;
        } else if (x_orig_ndim >= 2 && y_orig_ndim >= 2) {
            if (x_orig_ndim > 8 || y_orig_ndim > 8) return ge::GRAPH_FAILED; 
            x_kernel_ndim = x_orig_ndim;
            for(size_t i=0; i<x_kernel_ndim; ++i) x_kernel_dims[i] = x_shape.GetDim(i);
            y_kernel_ndim = y_orig_ndim;
            for(size_t i=0; i<y_kernel_ndim; ++i) y_kernel_dims[i] = y_shape.GetDim(i);
        } else {
            return ge::GRAPH_FAILED; 
        }
    
        for(uint32_t i=0; i<x_kernel_ndim; ++i) if(x_kernel_dims[i] < 0) return ge::GRAPH_FAILED;
        for(uint32_t i=0; i<y_kernel_ndim; ++i) if(y_kernel_dims[i] < 0) return ge::GRAPH_FAILED;
    
        TilingData tiling;
        uint32_t arrX[8], arrY[8], arrBias[8], arrZ[8];
    
        auto manual_encode = [](const int64_t dims[8], uint32_t ndim, uint32_t arr[8]) {
            arr[0] = (ndim < 7) ? ndim : 7;
            for (uint32_t i = 0; i < arr[0]; ++i) {
                arr[i+1] = (dims[i] >= 0) ? static_cast<uint32_t>(dims[i]) : 0;
            }
            for (uint32_t i = arr[0] + 1; i < 8; ++i) {
                arr[i] = 1;
            }
        };
    
        manual_encode(x_kernel_dims, x_kernel_ndim, arrX);
        manual_encode(y_kernel_dims, y_kernel_ndim, arrY);
        EncodeShape(&z_shape, arrZ);    
        if(EncodeShape(&actual_bias_shape_to_encode, arrBias)) {
            tiling.set_if_bias(1);
        } else {
            tiling.set_if_bias(0);
        }
        
        tiling.set_x_shape_info(arrX);
        tiling.set_y_shape_info(arrY);
        tiling.set_bias_shape_info(arrBias);
        tiling.set_z_shape_info(arrZ);
    
        size_t usrWorkspaceSize = 0; // 用户工作空间大小，初始化为 0
        {
            // --- 维度检查与提取 (与之前相同) ---
            if (x_kernel_ndim < 2 || y_kernel_ndim < 2) return ge::GRAPH_FAILED;
            int64_t M = x_kernel_dims[x_kernel_ndim - 2];
            int64_t Kx = x_kernel_dims[x_kernel_ndim - 1];
            int64_t Ky = y_kernel_dims[y_kernel_ndim - 2];
            int64_t N = y_kernel_dims[y_kernel_ndim - 1];

            // 验证 K 维度一致性
            if (Kx != Ky) {
                 // std::cerr << "Dimension mismatch: Kx=" << Kx << ", Ky=" << Ky << std::endl;
                 return ge::GRAPH_FAILED;
            }
            // 验证 M, K, N 非负
            if (Kx < 0 || M < 0 || N < 0) return ge::GRAPH_FAILED;
            int64_t K = Kx; // 使用 K

            // --- 计算对齐后的维度 ---
            size_t aligned_M = AlignUp(static_cast<size_t>(M), WORKSPACE_ALIGNMENT);
            size_t aligned_K = AlignUp(static_cast<size_t>(K), WORKSPACE_ALIGNMENT);
            size_t aligned_N = AlignUp(static_cast<size_t>(N), WORKSPACE_ALIGNMENT);

            // --- 计算所需总元素数 (MxK + KxN + MxN) ---
            uint64_t elements_mk = 0;
            // 检查 aligned_M * aligned_K 溢出 (size_t 范围)
            if (aligned_M > 0 && SIZE_MAX / aligned_M < aligned_K) return ge::GRAPH_FAILED;
            elements_mk = static_cast<uint64_t>(aligned_M) * aligned_K; // 转为 uint64_t

            uint64_t elements_kn = 0;
            // 检查 aligned_K * aligned_N 溢出 (size_t 范围)
            if (aligned_K > 0 && SIZE_MAX / aligned_K < aligned_N) return ge::GRAPH_FAILED;
            elements_kn = static_cast<uint64_t>(aligned_K) * aligned_N; // 转为 uint64_t

            uint64_t elements_mn = 0;
            // 检查 aligned_M * aligned_N 溢出 (size_t 范围)
            if (aligned_M > 0 && SIZE_MAX / aligned_M < aligned_N) return ge::GRAPH_FAILED;
            elements_mn = static_cast<uint64_t>(aligned_M) * aligned_N; // 转为 uint64_t

            // 累加总元素数，检查 uint64_t 溢出
            uint64_t total_elements = 0;
            if (total_elements > UINT64_MAX - elements_mk) return ge::GRAPH_FAILED;
            total_elements += elements_mk;
            if (total_elements > UINT64_MAX - elements_kn) return ge::GRAPH_FAILED;
            total_elements += elements_kn;
            if (total_elements > UINT64_MAX - elements_mn) return ge::GRAPH_FAILED;
            total_elements += elements_mn;

            // --- 计算总字节数 ---
            uint64_t total_bytes_u64 = 0;
            // 检查 total_elements * COMPLEX64_SIZE 溢出
            if (total_elements > 0 && UINT64_MAX / total_elements < COMPLEX64_SIZE) return ge::GRAPH_FAILED;
            total_bytes_u64 = total_elements * COMPLEX64_SIZE;

            // 检查结果是否超出 size_t 范围
            if (total_bytes_u64 > SIZE_MAX) return ge::GRAPH_FAILED;
            usrWorkspaceSize = 2 * static_cast<size_t>(total_bytes_u64);

            // --- (可选) 如果仍然需要固定的额外空间，在这里加上 ---
            // 例如，如果那 16MB 总是需要：
            // if (usrWorkspaceSize > SIZE_MAX - EXTRA_FIXED_WORKSPACE) return ge::GRAPH_FAILED;
            usrWorkspaceSize += EXTRA_FIXED_WORKSPACE;

        } // 工作空间计算块结束

        // --- 设置总工作空间大小 (用户 + 系统) ---
        uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace == nullptr) {
            return ge::GRAPH_FAILED;
        }

        size_t totalWorkspace = usrWorkspaceSize;
        // 检查加上系统空间后是否溢出
        if (totalWorkspace > SIZE_MAX - sysWorkspaceSize) {
             return ge::GRAPH_FAILED;
        }
        totalWorkspace += sysWorkspaceSize;

        currentWorkspace[0] = totalWorkspace; // 设置总需求大小
    
         if (tiling.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
              return ge::GRAPH_FAILED; 
         }
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
        return ge::GRAPH_SUCCESS;
    }
    } 


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class MatMul : public OpDef {
public:
    explicit MatMul(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
	this->AICore()
             .SetTiling(optiling::TilingFunc)
             .AddConfig("ascend910")
             .AddConfig("ascend310p")
             .AddConfig("ascend310b")
             .AddConfig("ascend910b");
        //this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(MatMul);
}
