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


at::Tensor my_op_impl_npu(const at::Tensor& start, const at::Tensor& stop, const at::Tensor& num, int64_t size, int64_t caseNum) {
    std::vector<int64_t> outShape;
    outShape.push_back(size);
    at::Tensor result = at::empty(outShape, at::TensorOptions().dtype(start.dtype()).device(start.options().device()));
    auto round = caseNum == 5 ? 50 : 1;
    for (size_t i = 0; i < round; i++)
    {
       EXEC_NPU_CMD(aclnnLinSpace, start, stop, num, result);
    }
    return result;
}



// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
		m.def("my_op(Tensor start, Tensor stop, Tensor num, int size, int caseNum) -> Tensor");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
		m.impl("my_op", &my_op_impl_npu);
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("custom_op", &my_op_impl_npu, "torch.linspace");
}
