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

std::tuple<at::Tensor, at::Tensor, at::Tensor> my_op_impl_npu(const at::Tensor &input, tensor_list hx, tensor_list params,
                                                    bool has_biases, int64_t num_layers, double dropout, bool train,
                                                    bool bidirectional, bool batch_first,const at::Tensor &outputOutGolden,const at::Tensor &hNOutGolden,const at::Tensor &cNOutGolden)
{
    // at::Tensor result = at::empty_like(input);
    at::Tensor outputOut;
    at::Tensor hNOut;
    at::Tensor cNOut;
    auto round = 50 ;
    for (size_t i = 0; i < round; i++)
    {
        auto tmp_x = input.clone();
        at::TensorList input_hx = at::TensorList(hx);
        at::TensorList input_params = at::TensorList(params);
        outputOut = at::empty_like(outputOutGolden,at::TensorOptions().dtype(outputOutGolden.dtype()).device(input.options().device()));
        hNOut = at::empty_like(hNOutGolden,at::TensorOptions().dtype(hNOutGolden.dtype()).device(input.options().device()));
        cNOut = at::empty_like(cNOutGolden,at::TensorOptions().dtype(cNOutGolden.dtype()).device(input.options().device()));
        EXEC_NPU_CMD(aclnnLstm,tmp_x, input_hx, input_params, has_biases, num_layers, dropout, train, bidirectional, batch_first, outputOut, hNOut, cNOut);
    }
    return {outputOut, hNOut, cNOut};
}

// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
		m.def("my_op(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, Tensor outputOutGolden, Tensor hNOutGolden, Tensor cNOutGolden) -> (Tensor, Tensor,Tensor)");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
		m.impl("my_op", &my_op_impl_npu);
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("custom_op", &my_op_impl_npu, "torch lstm");
}
