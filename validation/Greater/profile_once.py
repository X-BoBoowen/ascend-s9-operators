import sys

import torch
import torch_npu

import greater_validation_lib


torch.npu.config.allow_internal_format = False
torch.manual_seed(20260727)


def build_case(name):
    if name == "public":
        return (
            torch.randn((32, 64), dtype=torch.float16).npu(),
            torch.randn((32, 64), dtype=torch.float16).npu(),
        )
    if name == "same_large":
        return (
            torch.randn((1024, 1024), dtype=torch.float16).npu(),
            torch.randn((1024, 1024), dtype=torch.float16).npu(),
        )
    if name == "same_large_fp32":
        return (
            torch.randn((1024, 1024), dtype=torch.float32).npu(),
            torch.randn((1024, 1024), dtype=torch.float32).npu(),
        )
    if name == "same_large_bf16":
        return (
            torch.randn((1024, 1024), dtype=torch.bfloat16).npu(),
            torch.randn((1024, 1024), dtype=torch.bfloat16).npu(),
        )
    if name == "same_large_int8":
        return (
            torch.randint(-128, 128, (1024, 1024),
                          dtype=torch.int8).npu(),
            torch.randint(-128, 128, (1024, 1024),
                          dtype=torch.int8).npu(),
        )
    if name == "scalar_other":
        return (
            torch.randn((1024, 1024), dtype=torch.float16).npu(),
            torch.tensor(0.25, dtype=torch.float16).npu(),
        )
    if name == "scalar_self":
        return (
            torch.tensor(-0.5, dtype=torch.float16).npu(),
            torch.randn((1024, 1024), dtype=torch.float16).npu(),
        )
    if name == "scalar_other_fp32":
        return (
            torch.randn((1024, 1024), dtype=torch.float32).npu(),
            torch.tensor(0.25, dtype=torch.float32).npu(),
        )
    if name == "scalar_other_bf16":
        return (
            torch.randn((1024, 1024), dtype=torch.bfloat16).npu(),
            torch.tensor(0.25, dtype=torch.bfloat16).npu(),
        )
    if name == "scalar_other_int8":
        return (
            torch.randint(-128, 128, (1024, 1024),
                          dtype=torch.int8).npu(),
            torch.tensor(17, dtype=torch.int8).npu(),
        )
    if name == "last_broadcast":
        return (
            torch.randn((512, 512), dtype=torch.float16).npu(),
            torch.randn((512,), dtype=torch.float16).npu(),
        )
    if name == "mixed_broadcast":
        return (
            torch.randn((64, 1, 128), dtype=torch.float16).npu(),
            torch.randn((1, 32, 1), dtype=torch.float16).npu(),
        )
    if name == "int32_same":
        return (
            torch.randint(
                torch.iinfo(torch.int32).min,
                torch.iinfo(torch.int32).max,
                (1024, 1024),
                dtype=torch.int32,
            ).npu(),
            torch.randint(
                torch.iinfo(torch.int32).min,
                torch.iinfo(torch.int32).max,
                (1024, 1024),
                dtype=torch.int32,
            ).npu(),
        )
    raise ValueError(name)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "public"
    self_tensor, other_tensor = build_case(name)
    expected = torch.gt(self_tensor.cpu(), other_tensor.cpu())
    for _ in range(10):
        result = greater_validation_lib.greater(
            self_tensor,
            other_tensor,
        )
    torch.npu.synchronize()
    for _ in range(20):
        result = greater_validation_lib.greater(
            self_tensor,
            other_tensor,
        )
    torch.npu.synchronize()
    if not torch.equal(result.cpu(), expected):
        raise AssertionError(name)
    print(f"PROFILE_PASS {name}")


if __name__ == "__main__":
    main()
