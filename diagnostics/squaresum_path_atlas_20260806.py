import gc
import statistics
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080606

CASES = (
    ("fp1_o1", "fast1", (1, 4194304), (1,)),
    ("fp1_o8", "fast1", (8, 524288), (1,)),
    ("fp1_o64", "fast1", (64, 65536), (1,)),
    ("fp1_o4096", "fast1", (4096, 1024), (1,)),
    ("fp2_i8_o64", "fast2", (8, 65536, 8), (1,)),
    ("fp2_i8_o512", "fast2", (64, 8192, 8), (1,)),
    ("fp2_i64_o4096", "fast2", (64, 1024, 64), (1,)),
    ("fp2_rank4", "fast2", (8, 4096, 16, 8), (1, 2)),
    ("fp3_tail8192_o8", "fast3", (64, 8, 8192), (0, 2)),
    ("fp3_tail1024_o64", "fast3", (64, 64, 1024), (0, 2)),
    ("fp3_tail256_o64", "fast3", (256, 64, 256), (0, 2)),
    ("fp3_tail65536_o8", "fast3", (8, 8, 65536), (0, 2)),
    ("fp3_rank4_o32", "fast3", (64, 4, 8, 2048), (0, 3)),
    ("fp4_split_i16", "fast4", (256, 16, 64, 16), (0, 2)),
    ("fp4_split_i8", "fast4", (128, 64, 64, 8), (0, 2)),
    ("fp4_grouped_i2", "fast4", (64, 512, 64, 2), (0, 2)),
    ("fp4_long_i2", "fast4", (64, 32, 1024, 2), (0, 2)),
    ("fp4_generic_i65", "fast4", (32, 31, 64, 65), (0, 2)),
    ("fp4_rank5_split", "fast4", (128, 2, 16, 64, 4), (0, 3)),
    ("fp4_large_inner", "fast4", (16, 8, 16, 8, 256), (0, 2)),
    ("fp4_last128_i32", "fast4", (32, 32, 128, 32), (0, 2)),
)

DTYPES = (
    ("fp16", torch.float16, 2, 3e-3),
    ("fp32", torch.float32, 4, 2e-4),
)


def product(values):
    value = 1
    for item in values:
        value *= item
    return value


def run_case(label, index, name, route, shape, axes, dtype_info):
    dtype_name, dtype, type_bytes, tolerance = dtype_info
    elements = product(shape)
    generator = torch.Generator().manual_seed(SEED + index + elements)
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 0.06
        - 0.03
    ).to(dtype)
    expected = torch.sum(torch.square(input_cpu), dim=axes, keepdim=False)
    input_npu = input_cpu.npu()
    result = None
    for _ in range(4):
        result = square_sum_v1_validation_lib.square_sum_v1(
            input_npu, axes, False, list(expected.shape)
        )
    torch.npu.synchronize()

    repeats = 3
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

    actual = result.cpu()
    torch.testing.assert_close(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
        equal_nan=True,
    )
    median = statistics.median(samples)
    input_gbps = elements * type_bytes / median / 1000.0
    print(
        f"ATLAS label={label} case={name} route={route} dtype={dtype_name} "
        f"inputs={elements} outputs={expected.numel()} median_us={median:.6f} "
        f"input_gbps={input_gbps:.6f} "
        f"samples_us={[round(value, 3) for value in samples]}"
    )
    del actual, result, input_npu, expected, input_cpu
    gc.collect()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    passed = 0
    for index, (name, route, shape, axes) in enumerate(CASES):
        for dtype_info in DTYPES:
            run_case(label, index, name, route, shape, axes, dtype_info)
            passed += 1
    print(f"ATLAS_SUMMARY label={label} passed={passed}/{len(CASES) * len(DTYPES)}")


if __name__ == "__main__":
    main()
