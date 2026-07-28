#include <torch/extension.h>

#include "../common/pytorch_npu_helper.hpp"

extern "C" int aclnnGreaterGetWorkspaceSize(
    const aclTensor* self,
    const aclTensor* other,
    const aclTensor* output,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

extern "C" int aclnnGreater(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

at::Tensor greater_custom(
    const at::Tensor& self,
    const at::Tensor& other)
{
    const auto outputShape =
        at::infer_size(self.sizes(), other.sizes());
    auto output = at::empty(
        outputShape,
        self.options().dtype(at::kBool));

    aclTensor* aclSelf = ConvertType(self);
    aclTensor* aclOther = ConvertType(other);
    aclTensor* aclOutput = ConvertType(output);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto workspaceStatus = aclnnGreaterGetWorkspaceSize(
        aclSelf,
        aclOther,
        aclOutput,
        &workspaceSize,
        &executor);
    TORCH_CHECK(
        workspaceStatus == 0,
        "aclnnGreaterGetWorkspaceSize failed: ",
        aclGetRecentErrMsg());

    at::Tensor workspaceTensor;
    void* workspace = nullptr;
    if (workspaceSize != 0) {
        auto options = at::TensorOptions(
            torch_npu::utils::get_npu_device_type()).dtype(at::kByte);
        workspaceTensor = at::empty(
            {static_cast<int64_t>(workspaceSize)},
            options);
        workspace = const_cast<void*>(
            workspaceTensor.storage().data());
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const auto status = aclnnGreater(
        workspace,
        workspaceSize,
        executor,
        stream);
    TORCH_CHECK(
        status == 0,
        "aclnnGreater failed: ",
        aclGetRecentErrMsg());
    Release(aclSelf);
    Release(aclOther);
    Release(aclOutput);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("greater", &greater_custom);
}
