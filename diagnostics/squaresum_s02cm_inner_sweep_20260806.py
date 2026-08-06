import gc
import statistics
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080615
TARGET_INPUTS = 16 * 1024 * 1024
OUTER = 32
OUTPUT_ROWS = 31
INNERS = (17, 31, 33, 47, 63, 65, 79, 95, 127, 129, 255, 257, 511, 513, 1023)
DTYPES = (
    ("fp16", torch.float16, 3e-3),
    ("fp32", torch.float32, 3e-4),
)


def run_case(label, inner, dtype_name, dtype, tolerance):
    last_reduce = max(
        8, TARGET_INPUTS // (OUTER * OUTPUT_ROWS * inner)
    )
    shape = (OUTER, OUTPUT_ROWS, last_reduce, inner)
    generator = torch.Generator().manual_seed(SEED + inner)
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 0.02
        - 0.01
    ).to(dtype)
    expected = torch.sum(torch.square(input_cpu), dim=(0, 2))
    input_npu = input_cpu.npu()
    result = None
    for _ in range(3):
        result = square_sum_v1_validation_lib.square_sum_v1(
            input_npu, (0, 2), False, list(expected.shape)
        )
    torch.npu.synchronize()

    samples = []
    for _ in range(5):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        result = square_sum_v1_validation_lib.square_sum_v1(
            input_npu, (0, 2), False, list(expected.shape)
        )
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)

    actual = result.cpu()
    torch.testing.assert_close(
        actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True
    )
    median = statistics.median(samples)
    print(
        f"S02CM_INNER_SWEEP label={label} dtype={dtype_name} "
        f"inner={inner} last_reduce={last_reduce} inputs={input_cpu.numel()} "
        f"median_us={median:.6f} "
        f"samples_us={[round(value, 3) for value in samples]} PASS"
    )
    del actual, result, input_npu, expected, input_cpu
    gc.collect()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    passed = 0
    for inner in INNERS:
        for dtype_name, dtype, tolerance in DTYPES:
            run_case(label, inner, dtype_name, dtype, tolerance)
            passed += 1
    print(f"S02CM_INNER_SWEEP_SUMMARY label={label} passed={passed}/30")


if __name__ == "__main__":
    main()
