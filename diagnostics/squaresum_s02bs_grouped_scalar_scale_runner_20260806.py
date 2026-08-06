import gc
import statistics
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080607
CASES = (
    ("tiny_r2", (1, 256, 2, 1), (0, 2), False),
    ("small_r8", (3, 256, 8, 1), (0, 2), False),
    ("medium_r32", (16, 256, 32, 1), (0, 2), False),
    ("large_r64", (64, 256, 64, 1), (0, 2), False),
    ("tail257", (3, 257, 64, 1), (0, 2), False),
    ("outer_rows", (2, 3, 129, 64, 1), (1, 3), False),
)
DTYPES = (
    ("fp16", torch.float16, 2),
    ("fp32", torch.float32, 4),
)


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def tolerances(dtype):
    if dtype == torch.float16:
        return 3e-3, 3e-3
    return 1e-4, 1e-4


def measure(label, name, shape, axes, keep_dims, dtype_name, dtype, type_bytes):
    inputs = product(shape)
    generator = torch.Generator().manual_seed(SEED + inputs)
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 0.2
        - 0.1
    ).to(dtype)
    expected = torch.sum(
        torch.square(input_cpu), dim=axes, keepdim=keep_dims
    )
    input_npu = input_cpu.npu()
    result = None
    for _ in range(8):
        result = square_sum_v1_validation_lib.square_sum_v1(
            input_npu, axes, keep_dims, list(expected.shape)
        )
    torch.npu.synchronize()

    repeats = 40 if inputs < 1 << 16 else (15 if inputs < 1 << 20 else 5)
    samples = []
    for _ in range(7):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            result = square_sum_v1_validation_lib.square_sum_v1(
                input_npu, axes, keep_dims, list(expected.shape)
            )
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / repeats)

    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(
        result.cpu(), expected, rtol=rtol, atol=atol, equal_nan=True
    )
    median = statistics.median(samples)
    print(
        f"RESULT label={label} case={name} dtype={dtype_name} "
        f"inputs={inputs} outputs={expected.numel()} median_us={median:.6f} "
        f"input_gbps={inputs * type_bytes / median / 1000.0:.6f}"
    )
    del result, input_npu, expected, input_cpu
    gc.collect()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    passed = 0
    for name, shape, axes, keep_dims in CASES:
        for dtype_name, dtype, type_bytes in DTYPES:
            measure(
                label,
                name,
                shape,
                axes,
                keep_dims,
                dtype_name,
                dtype,
                type_bytes,
            )
            passed += 1
    print(f"SUMMARY label={label} passed={passed}/{len(CASES) * len(DTYPES)}")


if __name__ == "__main__":
    main()
