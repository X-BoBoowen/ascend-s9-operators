/**
*
* Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <torch/extension.h>
#include "../common/pytorch_npu_helper.hpp"
using namespace at;

// 修改输入
at::Tensor my_op_impl_npu(const at::Tensor &x, const at::Tensor &shape, int64_t axis, int64_t numAxes) {
    std::vector<int64_t> new_shape;
    for (int64_t i = 0; i < shape.numel(); ++i) {
        new_shape.push_back(shape[i].item<int64_t>());
    }
    at::Tensor result = at::empty(new_shape, x.options());
    // 调用aclnn接口计算
    EXEC_NPU_CMD(aclnnReshape, x, shape, axis, numAxes, result);
    return result;
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_op", &my_op_impl_npu, "caffe.reshape");
}
