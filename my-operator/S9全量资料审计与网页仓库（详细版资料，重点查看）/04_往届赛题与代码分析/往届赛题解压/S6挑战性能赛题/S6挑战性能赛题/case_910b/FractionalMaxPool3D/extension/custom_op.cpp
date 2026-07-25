#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include "../common/pytorch_npu_helper.hpp"
#include <cmath>  

using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

std::tuple<at::Tensor, at::Tensor> my_op_impl_npu(
    const at::Tensor& input, 
    const at::Tensor& _random_samples,
    const at::IntArrayRef& kernel_size, 
    const at::IntArrayRef& output_size, 
    double output_ratio_t, 
    double output_ratio_h,
    double output_ratio_w,
    bool return_indices,
   const at::IntArrayRef& golden_shape) {
    

    //创建输出shape
    std::vector<int64_t> outShape(golden_shape.begin(), golden_shape.end());
    
    std::nullptr_t temp = nullptr;
    //output_size是否为空
    at::Tensor result ;
    at::Tensor indices ;
    if (output_size.size() == 0)
    {
        size_t round = 50; 
        for (size_t i = 0; i < round; i++) {
        //创建输出tensor
        result = at::empty(outShape, input.options());
        // 创建indices tensor
        indices = at::empty_like(result, at::kInt);
        EXEC_NPU_CMD(aclnnFractionalMaxPool3D, 
                     input, 
                     _random_samples,
                     kernel_size, 
                     temp, 
                     output_ratio_t, 
                     output_ratio_h,
                     output_ratio_w,
                     return_indices, 
                     result, 
                     indices);
        }
    }
    else
    {
        size_t round = 50; 
        for (size_t i = 0; i < round; i++) {
        //创建输出tensor
        result = at::empty(outShape, input.options());
        // 创建indices tensor
        indices = at::empty_like(result, at::kInt);
        EXEC_NPU_CMD(aclnnFractionalMaxPool3D, 
                     input, 
                     _random_samples,
                     kernel_size, 
                     output_size, 
                     output_ratio_t, 
                     output_ratio_h,
                     output_ratio_w,
                     return_indices, 
                     result, 
                     indices);
    }
    }
    
    return {result, indices};
}

TORCH_LIBRARY(myops, m) {
		m.def("my_op(Tensor input,Tensor _random_samples, int[] kernel_size,int[] output_size,float output_ratio_t,float output_ratio_h,float output_ratio_w,bool return_indices,int[] golden_shape) -> (Tensor,Tensor)");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
		m.impl("my_op", &my_op_impl_npu);
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("custom_op", &my_op_impl_npu, "torch nn.FractionalMaxPool3d");
}