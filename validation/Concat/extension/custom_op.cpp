#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>

#include "../common/pytorch_npu_helper.hpp"

using tensor_list = std::vector<at::Tensor>;

extern "C" int aclnnConcatGetWorkspaceSize(
    const aclTensorList* inputs,
    int64_t dim,
    const aclTensor* output,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

extern "C" int aclnnConcat(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

static void execute_concat(
    const tensor_list& inputs,
    int64_t dim,
    const at::Tensor& output)
{
    static const auto initMemAddr =
        GetOpApiFuncAddr("InitHugeMemThreadLocal");
    static const auto uninitMemAddr =
        GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    static const auto releaseMemAddr =
        GetOpApiFuncAddr("ReleaseHugeMem");

    auto initMem =
        reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
    auto uninitMem =
        reinterpret_cast<UnInitHugeMemThreadLocal>(uninitMemAddr);
    if (initMem != nullptr) {
        initMem(nullptr, false);
    }

    at::TensorList inputList(inputs);
    aclTensorList* aclInputs = ConvertType(inputList);
    aclTensor* aclOutput = ConvertType(output);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto workspaceStatus = aclnnConcatGetWorkspaceSize(
        aclInputs,
        dim,
        aclOutput,
        &workspaceSize,
        &executor);
    TORCH_CHECK(
        workspaceStatus == 0,
        "aclnnConcatGetWorkspaceSize failed: ",
        aclGetRecentErrMsg());

    at::Tensor workspaceTensor;
    void* workspace = nullptr;
    if (workspaceSize != 0) {
        auto options = at::TensorOptions(
            torch_npu::utils::get_npu_device_type()).dtype(at::kByte);
        workspaceTensor = at::empty(
            {static_cast<int64_t>(workspaceSize)},
            options);
        workspace =
            const_cast<void*>(workspaceTensor.storage().data());
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto aclCall = [
        aclInputs,
        aclOutput,
        workspace,
        workspaceSize,
        executor,
        stream,
        releaseMemAddr]() -> int {
        const auto status = aclnnConcat(
            workspace,
            workspaceSize,
            executor,
            stream);
        TORCH_CHECK(
            status == 0,
            "aclnnConcat failed: ",
            aclGetRecentErrMsg());
        Release(aclInputs);
        Release(aclOutput);
        auto releaseMem =
            reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);
        if (releaseMem != nullptr) {
            releaseMem(nullptr, false);
        }
        return status;
    };

    at_npu::native::OpCommand command;
    command.Name("aclnnConcat");
    command.SetCustomHandler(aclCall);
    command.Run();
    if (uninitMem != nullptr) {
        uninitMem(nullptr, false);
    }
}

at::Tensor concat_custom(
    const tensor_list inputs,
    int64_t dim)
{
    TORCH_CHECK(!inputs.empty(), "inputs must not be empty");
    const int64_t rank = inputs[0].dim();
    const int64_t normalizedDim = dim < 0 ? dim + rank : dim;
    TORCH_CHECK(
        normalizedDim >= 0 && normalizedDim < rank,
        "invalid dim");

    std::vector<int64_t> outputShape(inputs[0].sizes().begin(),
                                     inputs[0].sizes().end());
    outputShape[normalizedDim] = 0;
    for (const auto& input : inputs) {
        outputShape[normalizedDim] += input.size(normalizedDim);
    }

    auto output = at::empty(outputShape, inputs[0].options());
    execute_concat(inputs, dim, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("concat", &concat_custom);
}
