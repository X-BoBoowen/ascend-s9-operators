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
at::Tensor my_op_impl_npu(const at::Tensor& input, const at::Tensor& index, const at::Tensor& src, int64_t dim, std::string reduce, bool includeSelf) {
    // 创建输出，根据实际需求判断根据第几个输入创建输出，确保输出类型正确
    std::string reduce_str = std::string(reduce);
    char *reduce_ptr = const_cast<char *>(reduce_str.c_str());
    
    at::Tensor result = at::empty_like(input);
    EXEC_NPU_CMD(aclnnScatterReduce, input, index, src, dim,reduce_ptr,includeSelf,result);
    return result;

}

// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
    m.def("my_op(Tensor input, Tensor index, Tensor src, int dim ,str reduce, bool includeSelf) -> Tensor");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("my_op", &my_op_impl_npu);
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_op", &my_op_impl_npu, "tf.where");
}
