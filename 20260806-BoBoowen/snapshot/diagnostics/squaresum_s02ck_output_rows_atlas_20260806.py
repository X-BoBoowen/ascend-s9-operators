import gc
import statistics
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080613

# Public route probes.  Shapes vary output-row parallelism while keeping the
# input in the 16M-element range; none are inferred from official cases.
CASES = (
    ("rows1_i256", (64, 1, 1024, 256), (0, 2)),
    ("rows2_i256", (32, 2, 1024, 256), (0, 2)),
    ("rows4_i256", (16, 4, 1024, 256), (0, 2)),
    ("rows8_i256", (8, 8, 1024, 256), (0, 2)),
    ("rows16_i256", (8, 16, 512, 256), (0, 2)),
    ("rows24_i256", (8, 24, 384, 256), (0, 2)),
    ("rows31_i65", (32, 31, 256, 65), (0, 2)),
    ("rows40_i33", (32, 40, 384, 33), (0, 2)),
    ("rows16_i33", (64, 16, 512, 33), (0, 2)),
    ("rows8_i65", (32, 8, 1024, 65), (0, 2)),
    ("rows41_fallback", (16, 41, 384, 65), (0, 2)),
    ("rank5_rows32_i128", (16, 2, 16, 256, 128), (0, 3)),
)

DTYPES = (
    ("fp16", torch.float16, 3e-3),
    ("fp32", torch.float32, 3e-4),
)


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def run_case(label, index, name, shape, axes, dtype_name, dtype, tolerance):
    elements = product(shape)
    generator = torch.Generator().manual_seed(SEED + index + elements)
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 0.02
        - 0.01
    ).to(dtype)
    expected = torch.sum(torch.square(input_cpu), dim=axes)
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
        actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True
    )
    print(
        f"S02CK_ROWS label={label} case={name} dtype={dtype_name} "
        f"inputs={elements} outputs={expected.numel()} "
        f"median_us={statistics.median(samples):.6f} "
        f"samples_us={[round(value, 3) for value in samples]} PASS"
    )
    del actual, result, input_npu, expected, input_cpu
    gc.collect()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    passed = 0
    for index, (name, shape, axes) in enumerate(CASES):
        for dtype_name, dtype, tolerance in DTYPES:
            run_case(
                label,
                index,
                name,
                shape,
                axes,
                dtype_name,
                dtype,
                tolerance,
            )
            passed += 1
    print(f"S02CK_ROWS_SUMMARY label={label} passed={passed}/24")


if __name__ == "__main__":
    main()
