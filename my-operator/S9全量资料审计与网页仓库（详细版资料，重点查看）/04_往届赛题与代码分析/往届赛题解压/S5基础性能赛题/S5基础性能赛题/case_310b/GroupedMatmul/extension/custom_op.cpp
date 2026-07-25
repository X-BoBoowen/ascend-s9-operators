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

std::vector<at::Tensor>  my_op_impl_npu(tensor_list x, tensor_list weight,
    tensor_list bias, tensor_list scale,
    tensor_list per_token_scale, const at::Tensor& group_list,
    int64_t splitItem, int64_t group_type, int64_t group_list_type, 
    int64_t act_type,tensor_list golden, int64_t caseNum) {

    std::vector<at::Tensor> result_list;
    for (size_t i = 0; i < golden.size(); i++)
    {
        auto tmp_result_shape = golden[i].sizes().data();
        std::vector<int64_t> outShape;
        for (size_t j = 0; j < golden[i].sizes().size(); j++)
        {
            outShape.push_back(tmp_result_shape[j]);
        }
        auto type = at::kBFloat16;
        if (scale[0].dtype() == at::kFloat)
        {
            type = at::kHalf;
        }
        
        at::Tensor result = at::empty(outShape, at::TensorOptions().dtype(type).device(scale[0].options().device()));
        result_list.push_back(result);
    }
    
    at::TensorList result_ = at::TensorList(result_list);
    at::TensorList input_x = at::TensorList(x);
    at::TensorList input_weight = at::TensorList(weight);
    at::TensorList input_bias = at::TensorList(bias);
    at::TensorList input_scale = at::TensorList(scale);
    at::TensorList input_per_token_scale = at::TensorList(per_token_scale);

    auto round = 50 ;
    for (size_t i = 0; i < round; i++)
    {
        EXEC_NPU_CMD(aclnnGroupedMatmul, 
        input_x, 
        input_weight, 
        input_bias, 
        input_scale, 
        input_per_token_scale, 
        group_list,
        splitItem,group_type, group_list_type, act_type,
        result_);
    }
    return result_list;
}

// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
		m.def("my_op(Tensor[] x, Tensor[] weight,Tensor[] bias,Tensor[] scale, Tensor[] per_token_scale, Tensor group_list, int splitItem ,int group_type,int group_list_type,int act_type, Tensor[] golden, int caseNum) -> Tensor[]");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
		m.impl("my_op", &my_op_impl_npu);
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("custom_op", &my_op_impl_npu, "torch matmul");
}
