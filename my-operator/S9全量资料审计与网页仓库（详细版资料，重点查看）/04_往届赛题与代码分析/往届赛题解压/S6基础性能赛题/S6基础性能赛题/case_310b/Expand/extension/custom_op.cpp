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

at::Tensor create_tensor_with_shape(const at::IntArrayRef& size, const at::Tensor& input) {
    // 使用input的配置（数据类型、设备等），结合指定的size创建张量
    // 可根据需求替换为at::zeros、at::empty、at::randn等函数
    at::Tensor result = at::zeros(size, input.options());
    return result;
}

at::Tensor my_op_impl_npu(const at::Tensor& input, const at::IntArrayRef & size, int64_t caseNum) {
    
    auto round = 50 ;
    at::Tensor result;
    for (size_t i = 0; i < round; i++)
    {
        result = create_tensor_with_shape(size, input);
        EXEC_NPU_CMD(aclnnExpand, input, size, result);
    }
    return result;
}



// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
		m.def("my_op(Tensor input, int[] dim, int caseNum) -> Tensor");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
		m.impl("my_op", &my_op_impl_npu);
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("custom_op", &my_op_impl_npu, "torch logcumsumexp");
}
