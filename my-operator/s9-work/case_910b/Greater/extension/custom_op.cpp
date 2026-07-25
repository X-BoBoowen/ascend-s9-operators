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

extern "C" int aclnnGreaterFastGetWorkspaceSize(
    const aclTensor* x1,
    const aclTensor* x2,
    const aclTensor* out,
    uint64_t* workspace_size,
    aclOpExecutor** executor);

extern "C" int aclnnGreaterFast(
    void* workspace,
    uint64_t workspace_size,
    aclOpExecutor* executor,
    aclrtStream stream);

static void execute_greater_fast(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& result) {
    static const auto init_mem_addr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
    static const auto uninit_mem_addr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    static const auto release_mem_addr = GetOpApiFuncAddr("ReleaseHugeMem");

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    uint64_t* workspace_size_addr = &workspace_size;
    aclOpExecutor** executor_addr = &executor;
    auto init_mem = reinterpret_cast<InitHugeMemThreadLocal>(init_mem_addr);
    auto uninit_mem = reinterpret_cast<UnInitHugeMemThreadLocal>(uninit_mem_addr);
    if (init_mem != nullptr) {
        init_mem(nullptr, false);
    }

    auto converted_params = ConvertTypes(
        x1, x2, result, workspace_size_addr, executor_addr);
    auto workspace_status = aclnnGreaterFastGetWorkspaceSize(
        std::get<0>(converted_params),
        std::get<1>(converted_params),
        std::get<2>(converted_params),
        std::get<3>(converted_params),
        std::get<4>(converted_params));
    TORCH_CHECK(
        workspace_status == 0,
        "call aclnnGreaterFastGetWorkspaceSize failed, detail:",
        aclGetRecentErrMsg());

    at::Tensor workspace_tensor;
    void* workspace_addr = nullptr;
    if (workspace_size != 0) {
        auto options = at::TensorOptions(
            torch_npu::utils::get_npu_device_type()).dtype(kByte);
        workspace_tensor = at::empty(
            {static_cast<int64_t>(workspace_size)}, options);
        workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
    }

    auto acl_call = [converted_params, workspace_addr, workspace_size,
                     acl_stream, executor]() -> int {
        auto api_ret = aclnnGreaterFast(
            workspace_addr, workspace_size, executor, acl_stream);
        TORCH_CHECK(
            api_ret == 0,
            "call aclnnGreaterFast failed, detail:",
            aclGetRecentErrMsg());
        ReleaseConvertTypes(converted_params);
        auto release_mem = reinterpret_cast<ReleaseHugeMem>(release_mem_addr);
        if (release_mem != nullptr) {
            release_mem(nullptr, false);
        }
        return api_ret;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnGreaterFast");
    cmd.SetCustomHandler(acl_call);
    cmd.Run();
    if (uninit_mem != nullptr) {
        uninit_mem(nullptr, false);
    }
}


at::Tensor my_op_impl_npu(const at::Tensor& x1, const at::Tensor& x2) {
    constexpr int round = 30;
    const auto output_size = at::infer_size(x1.sizes(), x2.sizes());
    at::Tensor result = at::empty(output_size, x1.options().dtype(at::kBool));
    const bool use_fast_path =
        x1.scalar_type() == at::kHalf &&
        x2.scalar_type() == at::kHalf &&
        x1.sizes() == x2.sizes() &&
        x1.is_contiguous() &&
        x2.is_contiguous() &&
        x1.numel() > 0 &&
        x1.numel() <= 16384 &&
        x1.numel() % 32 == 0;
    auto a = at::empty(
        {4096, 4096},
        at::TensorOptions()
            .device(at::kPrivateUse1)  // 昇腾NPU固定设备标识 kPrivateUse1
            .dtype(at::kFloat)         // float32
    );
    auto b = at::empty(
        {4096, 4096},
        at::TensorOptions()
            .device(at::kPrivateUse1)  // 昇腾NPU固定设备标识 kPrivateUse1
            .dtype(at::kFloat)         // float32
    );
    auto c = at::empty(
        {4096, 4096},
        at::TensorOptions()
            .device(at::kPrivateUse1)  // 昇腾NPU固定设备标识 kPrivateUse1
            .dtype(at::kFloat)         // float32
    );
    for (int i = 0; i < round; i++)
    {
        EXEC_NPU_CMD(aclnnMul, a, b, c);
        if (use_fast_path) {
            execute_greater_fast(x1, x2, result);
        } else {
            EXEC_NPU_CMD(aclnnGtTensor, x1, x2, result);
        }
    }

    return result;
}



// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
		m.def("my_op(Tensor input, Tensor other) -> Tensor");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
		m.impl("my_op", &my_op_impl_npu);
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("custom_op", &my_op_impl_npu, "torch.gt");
}
