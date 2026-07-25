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


at::Tensor my_op_impl_npu(const at::Tensor& x, int64_t num_groups, int64_t num_channels, double eps, bool affine) {
    // at::Tensor result = at::empty_like(input);
    at::Tensor result;
    auto round = 50 ;
    for (size_t i = 0; i < round; i++)
    {
        auto tmp_x = x.clone();
        result = at::empty_like(x);
        EXEC_NPU_CMD(aclnnGroupNorm, tmp_x, num_groups, num_channels, eps, affine, result);
    }
    return result;
}



// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
		m.def("my_op(Tensor x, int num_groups, int num_channels, float eps, bool affine) -> Tensor");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
		m.impl("my_op", &my_op_impl_npu);
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("custom_op", &my_op_impl_npu, "torch group_norm");
}
