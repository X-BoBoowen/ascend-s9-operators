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


std::tuple<at::Tensor, at::Tensor> my_op_impl_npu(const at::Tensor& self, const at::Tensor& gamma,bool left,bool check_error) {
    int64_t viewDims[] = {self.sizes()[0]}; 
		at::Tensor second = at::empty(viewDims, at::TensorOptions().dtype(at::ScalarType::Int).device(self.options().device()));
    at::Tensor first = at::empty_like(gamma);
		EXEC_NPU_CMD(aclnnSolveEx, self, gamma,left,check_error,first,second);
   
   
		return std::make_tuple(first, second);
}



// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
		m.def("my_op(Tensor self, Tensor gamma,bool left,bool check_error) -> (Tensor,Tensor)");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
		m.impl("my_op", &my_op_impl_npu);
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("custom_op", &my_op_impl_npu, "tf.where");
}
