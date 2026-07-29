#include <torch/extension.h>

#include "../common/pytorch_npu_helper.hpp"

using namespace at;

at::Tensor square_sum_v1_custom(
    const at::Tensor& input,
    const at::IntArrayRef& axis,
    const bool keepDims,
    const at::IntArrayRef& resultShape)
{
    auto output = at::empty(resultShape, input.options());
    EXEC_NPU_CMD(
        aclnnSquareSumV1,
        input,
        axis,
        keepDims,
        output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("square_sum_v1", &square_sum_v1_custom);
}
