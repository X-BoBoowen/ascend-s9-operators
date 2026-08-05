import gc
import statistics
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080502


CASES = (
    ("fp1_last_o1", (1, 1048576), (1,)),
    ("fp1_last_o64", (64, 16384), (1,)),
    ("fp1_last_o1024", (1024, 4096), (1,)),
    ("fp2_middle_i4", (262144, 4), (0,)),
    ("fp2_middle_i8", (131072, 8), (0,)),
    ("fp2_middle_i64", (32768, 64), (0,)),
    ("fp2_middle_o8_i8", (8, 32768, 8), (1,)),
    ("fp2_middle_o32_i32", (32, 4096, 32), (1,)),
    ("fp3_tail16", (16, 32, 16, 32, 16), (0, 2, 4)),
    ("fp3_tail64", (8, 16, 8, 16, 64), (0, 2, 4)),
    ("fp3_tail256", (8, 16, 8, 16, 256), (0, 2, 4)),
    ("fp3_tail1024", (4, 16, 4, 16, 1024), (0, 2, 4)),
    ("fp3_tail4096", (4, 8, 4, 8, 4096), (0, 2, 4)),
    ("fp3_small_outputs", (32, 16, 64, 64), (0, 2, 3)),
    ("fp4_inner1", (64, 256, 64, 1), (0, 2)),
    ("fp4_inner8", (64, 32, 64, 8), (0, 2)),
    ("fp4_inner32", (64, 8, 64, 32), (0, 2)),
    ("fp4_inner64", (32, 8, 64, 64), (0, 2)),
    ("fp4_inner65", (32, 17, 64, 65), (0, 2)),
    ("fp4_inner128", (32, 8, 32, 128), (0, 2)),
    ("fp4_inner256", (16, 8, 32, 256), (0, 2)),
    ("fp4_inner1024", (8, 8, 16, 1024), (0, 2)),
    ("fp4_rank5_inner256", (16, 8, 32, 4, 64), (0, 2)),
)

DTYPES = (
    ("fp16", torch.float16, 2),
    ("bf16", torch.bfloat16, 2),
    ("fp32", torch.float32, 4),
)


def tolerances(dtype):
    if dtype == torch.float16:
        return 3e-3, 3e-3
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    return 1e-4, 1e-4


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def measure(label, case_name, shape, axes, dtype_name, dtype, type_bytes):
    input_elements = product(shape)
    repeats = 30 if input_elements <= 1 << 20 else (20 if input_elements <= 3 << 20 else 10)
    generator = torch.Generator().manual_seed(SEED + len(shape))
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 2.0
        - 1.0
    ).to(dtype)
    expected = torch.sum(torch.square(input_cpu), dim=axes, keepdim=False)
    input_npu = input_cpu.npu()
    result = None
    for _ in range(12):
        result = square_sum_v1_validation_lib.square_sum_v1(
            input_npu, axes, False, list(expected.shape)
        )
    torch.npu.synchronize()

    samples = []
    for _ in range(5):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            result = square_sum_v1_validation_lib.square_sum_v1(
                input_npu, axes, False, list(expected.shape)
            )
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / repeats)

    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(
        result.cpu(), expected, rtol=rtol, atol=atol, equal_nan=True
    )
    median = statistics.median(samples)
    input_gbps = input_elements * type_bytes / median / 1000.0
    print(
        f"RESULT label={label} case={case_name} dtype={dtype_name} "
        f"inputs={input_elements} outputs={expected.numel()} "
        f"median_us={median:.6f} input_gbps={input_gbps:.6f} "
        f"samples_us={[round(value, 6) for value in samples]}"
    )
    del result, input_npu, expected, input_cpu
    gc.collect()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    passed = 0
    for case_name, shape, axes in CASES:
        for dtype_name, dtype, type_bytes in DTYPES:
            measure(
                label,
                case_name,
                shape,
                axes,
                dtype_name,
                dtype,
                type_bytes,
            )
            passed += 1
    print(f"SUMMARY label={label} passed={passed}/{len(CASES) * len(DTYPES)}")


if __name__ == "__main__":
    main()
