import statistics

import torch
import torch_npu

import index_add_validation_lib


torch.npu.config.allow_internal_format = False
torch.manual_seed(20260728)


def make_tensor(shape, dtype):
    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        return torch.randn(shape, dtype=dtype)
    if dtype == torch.int32:
        return torch.randint(-(2**30), 2**30, shape, dtype=dtype)
    return torch.randint(-128, 128, shape, dtype=dtype)


CASES = (
    ("fp16_m2048_inner4096", (1024, 4096), 0, 2048, torch.float16),
    ("fp16_m8000_inner4096", (512, 4096), 0, 8000, torch.float16),
    ("fp32_outer2_m1024_inner2048", (2, 512, 2048), 1, 1024, torch.float32),
    ("fp32_outer2_m8000_inner2048", (2, 512, 2048), 1, 8000, torch.float32),
    ("bf16_outer2_m8000_inner2048", (2, 512, 2048), 1, 8000, torch.bfloat16),
    ("fp32_m8000_inner4096", (512, 4096), 0, 8000, torch.float32),
    ("int32_outer8_m2048_inner1024", (8, 256, 1024), 1, 2048, torch.int32),
    ("int8_m4096_inner2048", (512, 2048), 0, 4096, torch.int8),
    ("fp32_dim2048_m1024_inner2048", (2048, 2048), 0, 1024, torch.float32),
    ("int32_dim2048_m1024_inner2048", (2048, 2048), 0, 1024, torch.int32),
    ("int8_dim2048_m1024_inner2048", (2048, 2048), 0, 1024, torch.int8),
    ("fp32_dim4096_m1024_inner2048", (4096, 2048), 0, 1024, torch.float32),
    ("int32_dim4096_m1024_inner2048", (4096, 2048), 0, 1024, torch.int32),
    ("int8_dim4096_m1024_inner2048", (4096, 2048), 0, 1024, torch.int8),
)


def run_case(name, self_shape, dim, index_count, dtype):
    self_cpu = make_tensor(self_shape, dtype)
    dim_size = self_shape[dim]
    index_cpu = torch.randint(
        0, dim_size, (index_count,), dtype=torch.int32
    )
    source_shape = list(self_shape)
    source_shape[dim] = index_count
    source_cpu = make_tensor(tuple(source_shape), dtype)
    expected = torch.index_add(
        self_cpu, dim, index_cpu, source_cpu
    )
    self_npu = self_cpu.npu()
    index_npu = index_cpu.npu()
    source_npu = source_cpu.npu()
    for _ in range(3):
        result = index_add_validation_lib.index_add(
            self_npu, dim, index_npu, source_npu
        )
    torch.npu.synchronize()
    values = []
    for _ in range(3):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            result = index_add_validation_lib.index_add(
                self_npu, dim, index_npu, source_npu
            )
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / 5)
    actual = result.cpu()
    if expected.dtype in (torch.int8, torch.int32):
        if not torch.equal(actual, expected):
            raise AssertionError(name)
    elif not torch.allclose(actual, expected, rtol=1e-3, atol=1e-3):
        mismatch = torch.max(torch.abs(actual.float() - expected.float()))
        raise AssertionError(f"{name}: max_abs={mismatch.item()}")
    print(
        f"{name}: median_us={statistics.median(values):.3f}, "
        f"trials_us={[round(value, 3) for value in values]}"
    )


def main():
    for case in CASES:
        run_case(*case)


if __name__ == "__main__":
    main()
