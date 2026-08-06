import gc
import statistics
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080608

# A route-oriented 32M-element matrix.  These are synthetic public probes, not
# inferred official cases.
CASES = (
    ("fp1_o1", "fast1", (1, 33554432), (1,)),
    ("fp1_o64", "fast1", (64, 524288), (1,)),
    ("fp1_o4096", "fast1", (4096, 8192), (1,)),
    ("fp2_i8_o64", "fast2", (8, 524288, 8), (1,)),
    ("fp2_i64_o4096", "fast2", (64, 8192, 64), (1,)),
    ("fp3_o8_tail65536", "fast3", (64, 8, 65536), (0, 2)),
    ("fp3_o32_tail16384", "fast3", (64, 32, 16384), (0, 2)),
    ("fp3_o64_tail8192", "fast3", (64, 64, 8192), (0, 2)),
    ("fp4_inner2", "fast4", (64, 256, 1024, 2), (0, 2)),
    ("fp4_inner8", "fast4", (128, 32, 1024, 8), (0, 2)),
    ("fp4_inner65", "fast4", (32, 31, 512, 65), (0, 2)),
    ("fp4_inner256", "fast4", (16, 32, 256, 256), (0, 2)),
    ("fp4_rank5", "fast4", (64, 2, 16, 1024, 16), (0, 3)),
)

DTYPES = (
    ("fp16", torch.float16, 2, 3e-3),
    ("fp32", torch.float32, 4, 3e-4),
)


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def run_case(label, index, name, route, shape, axes, dtype_info):
    dtype_name, dtype, type_bytes, tolerance = dtype_info
    elements = product(shape)
    generator = torch.Generator().manual_seed(SEED + index + elements)
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 0.02
        - 0.01
    ).to(dtype)
    expected = torch.sum(torch.square(input_cpu), dim=axes, keepdim=False)
    input_npu = input_cpu.npu()
    result = None
    for _ in range(3):
        result = square_sum_v1_validation_lib.square_sum_v1(
            input_npu, axes, False, list(expected.shape)
        )
    torch.npu.synchronize()

    samples = []
    for _ in range(5):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(2):
            result = square_sum_v1_validation_lib.square_sum_v1(
                input_npu, axes, False, list(expected.shape)
            )
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / 2.0)

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
        f"LARGE_ATLAS label={label} case={name} route={route} "
        f"dtype={dtype_name} inputs={elements} outputs={expected.numel()} "
        f"median_us={median:.6f} input_gbps={input_gbps:.6f} "
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
    total = len(CASES) * len(DTYPES)
    print(f"LARGE_ATLAS_SUMMARY label={label} passed={passed}/{total}")


if __name__ == "__main__":
    main()
