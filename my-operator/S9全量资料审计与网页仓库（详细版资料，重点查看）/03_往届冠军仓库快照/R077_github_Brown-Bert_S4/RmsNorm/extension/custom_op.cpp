/**
*
* Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include "../common/pytorch_npu_helper.hpp"
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

// 修改输入
std::tuple<at::Tensor, at::Tensor> my_op_impl_npu(const at::Tensor& self, const at::Tensor& gamma, double epsilon) {
    // 创建输出，根据实际需求判断根据第几个输入创建输出，确保输出类型正确
    at::Tensor result = at::Tensor(self);

    int64_t x_dimlenth = self.sizes().size();
    int64_t gamma_dimlenth = gamma.sizes().size();
    std::vector<int64_t> ratd_shape;
    for(int i = 0; i < x_dimlenth; i++){
        if(x_dimlenth-i > gamma_dimlenth)
        {
            ratd_shape.push_back(self.sizes().data()[i]);
        }
        else
        {
            ratd_shape.push_back(1);
        }
    }

    at::Tensor rstd = at::empty(ratd_shape, at::TensorOptions().dtype(self.dtype()).device(at::kCPU));

    // 调用aclnn接口计算
    EXEC_NPU_CMD(aclnnRmsNorm, self, gamma, epsilon,result,rstd);
    return {result, rstd};
}



// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
    m.def("my_op(Tensor self, Tensor self, float epsilo) -> (Tensor,Tensor)");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("my_op", &my_op_impl_npu);
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_op", &my_op_impl_npu, "tf.where");
}
