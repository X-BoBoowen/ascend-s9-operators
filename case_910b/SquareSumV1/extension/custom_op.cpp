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

static bool has_expected_fast_shape(
    const at::Tensor& input,
    const at::IntArrayRef& result_shape) {
    if (result_shape.size() != static_cast<size_t>(input.dim())) {
        return false;
    }
    for (int64_t i = 0; i < input.dim() - 1; ++i) {
        if (result_shape[i] != input.size(i)) {
            return false;
        }
    }
    return result_shape[input.dim() - 1] == 1;
}

at::Tensor my_op_impl_npu(
    const at::Tensor& input,
    const at::IntArrayRef& axis,
    bool keep_dims,
    const at::IntArrayRef& result_shape) {
    auto round = 30;
    const int64_t reduce_len =
        input.dim() > 0 ? input.size(input.dim() - 1) : 0;
    const int64_t outer =
        reduce_len > 0 ? input.numel() / reduce_len : 0;
    const bool last_axis =
        axis.size() == 1 &&
        (axis[0] == -1 || axis[0] == input.dim() - 1);
    const bool use_fast_path =
        input.scalar_type() == at::kHalf &&
        input.dim() >= 1 &&
        input.is_contiguous() &&
        input.storage_offset() == 0 &&
        keep_dims &&
        last_axis &&
        reduce_len > 0 &&
        reduce_len <= 64 &&
        outer > 0 &&
        has_expected_fast_shape(input, result_shape);

    at::Tensor result;
    auto dtype = input.scalar_type();
    auto a = at::empty(
        {4096, 4096},
        at::TensorOptions()
            .device(at::kPrivateUse1)
            .dtype(at::kFloat));
    auto b = at::empty(
        {4096, 4096},
        at::TensorOptions()
            .device(at::kPrivateUse1)
            .dtype(at::kFloat));
    auto c = at::empty(
        {4096, 4096},
        at::TensorOptions()
            .device(at::kPrivateUse1)
            .dtype(at::kFloat));
    at::Tensor ones;
    if (use_fast_path) {
        ones = at::ones({reduce_len}, input.options());
    }

    for (size_t i = 0; i < round; i++) {
        EXEC_NPU_CMD(aclnnMul, a, b, c);
        if (use_fast_path) {
            at::Tensor squared = at::empty_like(input);
            EXEC_NPU_CMD(aclnnMul, input, input, squared);
            result = at::mv(
                squared.reshape({outer, reduce_len}),
                ones).reshape(result_shape);
        } else {
            result = at::empty(result_shape, input.options());
            at::Tensor squared = at::empty_like(input);
            EXEC_NPU_CMD(aclnnMul, input, input, squared);
            EXEC_NPU_CMD(
                aclnnReduceSum,
                squared,
                axis,
                keep_dims,
                dtype,
                result);
        }
    }
    return result;
}

TORCH_LIBRARY(myops, m) {
    m.def(
        "my_op(Tensor input, int[] axis, bool keep_dims, "
        "int[] result_shape) -> Tensor");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("my_op", &my_op_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "custom_op",
        &my_op_impl_npu,
        "torch.sum(torch.square)");
}
