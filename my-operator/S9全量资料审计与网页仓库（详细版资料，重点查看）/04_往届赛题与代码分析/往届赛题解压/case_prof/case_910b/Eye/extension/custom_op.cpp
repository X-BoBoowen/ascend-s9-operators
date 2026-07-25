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
at::Tensor my_op_impl_npu(const at::Tensor& self,int64_t numRows,int64_t numColumns,const c10::optional<at::IntArrayRef> & batchShape,int64_t dtype) {
    // 创建输出，根据实际需求判断根据第几个输入创建输出，确保输出类型正确
    

    // 调用aclnn接口计算
    EXEC_NPU_CMD(aclnnEye, self, numRows, numColumns,batchShape,dtype);
    return self;
}



// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
    m.def("my_op(Tensor self,int numRows,int numColumns,int[]? batchShape,int dtype) -> Tensor");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("my_op", &my_op_impl_npu);
    
}


// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_op", &my_op_impl_npu, "tf.where");
}
