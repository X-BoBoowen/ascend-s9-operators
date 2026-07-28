#include <torch/extension.h>

#include "../common/pytorch_npu_helper.hpp"

using namespace at;

at::Tensor index_add_custom(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source)
{
    auto output = at::empty_like(self);
    EXEC_NPU_CMD(
        aclnnIndexAdd,
        self,
        index,
        source,
        dim,
        output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("index_add", &index_add_custom);
}
