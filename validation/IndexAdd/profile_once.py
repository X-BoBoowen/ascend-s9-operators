import sys

import torch
import torch_npu

import index_add_validation_lib


torch.npu.config.allow_internal_format = False
torch.manual_seed(20260727)


def build_case(name):
    if name == "public":
        self_tensor = torch.randint(
            -50, 50, (32, 128), dtype=torch.int8
        )
        index = torch.randint(
            0, 32, (120,), dtype=torch.int32
        )
        source = torch.randint(
            -10, 10, (120, 128), dtype=torch.int8
        )
        return self_tensor, 0, index, source
    if name == "fp16_dim0":
        self_tensor = torch.randn(
            (1024, 256), dtype=torch.float16
        )
        index = torch.randint(
            0, 1024, (2048,), dtype=torch.int32
        )
        source = torch.randn(
            (2048, 256), dtype=torch.float16
        )
        return self_tensor, 0, index, source
    if name == "bf16_dim0":
        self_tensor = torch.randn(
            (256, 257), dtype=torch.bfloat16
        )
        index = torch.randint(
            0, 256, (1024,), dtype=torch.int32
        )
        source = torch.randn(
            (1024, 257), dtype=torch.bfloat16
        )
        return self_tensor, 0, index, source
    if name == "fp32_middle":
        self_tensor = torch.randn(
            (8, 128, 65), dtype=torch.float32
        )
        index = torch.randint(
            0, 128, (512,), dtype=torch.int32
        )
        source = torch.randn(
            (8, 512, 65), dtype=torch.float32
        )
        return self_tensor, 1, index, source
    if name == "int32_last":
        self_tensor = torch.randint(
            -(2**30),
            2**30,
            (8, 33, 128),
            dtype=torch.int32,
        )
        index = torch.randint(
            0, 128, (256,), dtype=torch.int32
        )
        source = torch.randint(
            -(2**30),
            2**30,
            (8, 33, 256),
            dtype=torch.int32,
        )
        return self_tensor, -1, index, source
    if name == "int8_repeat":
        self_tensor = torch.randint(
            -128, 128, (64, 129), dtype=torch.int8
        )
        index = torch.zeros((1024,), dtype=torch.int32)
        source = torch.randint(
            -128, 128, (1024, 129), dtype=torch.int8
        )
        return self_tensor, 0, index, source
    if name == "index_max":
        self_tensor = torch.randn(
            (32, 17), dtype=torch.float16
        )
        index = torch.randint(
            0, 32, (8000,), dtype=torch.int32
        )
        source = torch.randn(
            (8000, 17), dtype=torch.float16
        )
        return self_tensor, 0, index, source
    raise ValueError(name)


def assert_result(name, actual, expected):
    if expected.dtype in (
        torch.float32,
        torch.bfloat16,
        torch.float16,
    ):
        tolerance = 1e-4 if expected.dtype == torch.float32 else 1e-3
        torch.testing.assert_close(
            actual,
            expected,
            rtol=tolerance,
            atol=tolerance,
            msg=lambda message: f"{name}: {message}",
        )
    elif not torch.equal(actual, expected):
        mismatch = torch.count_nonzero(actual != expected).item()
        raise AssertionError(f"{name}: integer mismatch={mismatch}")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "public"
    self_cpu, dim, index_cpu, source_cpu = build_case(name)
    expected = torch.index_add(
        self_cpu, dim, index_cpu, source_cpu
    )
    self_tensor = self_cpu.npu()
    index = index_cpu.npu()
    source = source_cpu.npu()
    for _ in range(10):
        result = index_add_validation_lib.index_add(
            self_tensor, dim, index, source
        )
    torch.npu.synchronize()
    for _ in range(20):
        result = index_add_validation_lib.index_add(
            self_tensor, dim, index, source
        )
    torch.npu.synchronize()
    assert_result(name, result.cpu(), expected)
    print(f"PROFILE_PASS {name}")


if __name__ == "__main__":
    main()
