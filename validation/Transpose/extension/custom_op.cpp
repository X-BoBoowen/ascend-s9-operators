#include <torch/extension.h>

#include "../../../case_910b/Transpose/common/pytorch_npu_helper.hpp"

using namespace at;

at::Tensor transpose_custom(
    const at::Tensor& input,
    const at::IntArrayRef& dims)
{
    std::vector<int64_t> resultShape;
    resultShape.reserve(dims.size());
    for (const int64_t rawAxis : dims) {
        const int64_t axis =
            rawAxis < 0 ? rawAxis + input.dim() : rawAxis;
        resultShape.push_back(input.size(axis));
    }
    auto output = at::empty(resultShape, input.options());
    EXEC_NPU_CMD(aclnnTranspose, input, dims, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("transpose", &transpose_custom);
}
