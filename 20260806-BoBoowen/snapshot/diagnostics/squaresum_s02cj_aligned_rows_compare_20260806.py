import gc
import statistics
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080612

CASES = (
    ("inner65_rows31", (32, 31, 512, 65), (0, 2)),
    ("inner256_rows32", (16, 32, 256, 256), (0, 2)),
)

DTYPES = (
    ("fp16", torch.float16, 3e-3),
    ("fp32", torch.float32, 3e-4),
)


def run_case(label, name, shape, axes, dtype_name, dtype, tolerance):
    generator = torch.Generator().manual_seed(SEED + len(shape) + shape[-1])
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 0.02
        - 0.01
    ).to(dtype)
    expected = torch.sum(torch.square(input_cpu), dim=axes)
    input_npu = input_cpu.npu()
    result = None
    for _ in range(4):
        result = square_sum_v1_validation_lib.square_sum_v1(
            input_npu, axes, False, list(expected.shape)
        )
    torch.npu.synchronize()

    samples = []
    for _ in range(9):
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
        f"S02CJ_COMPARE label={label} case={name} dtype={dtype_name} "
        f"median_us={statistics.median(samples):.6f} "
        f"samples_us={[round(value, 3) for value in samples]} PASS"
    )
    del actual, result, input_npu, expected, input_cpu
    gc.collect()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    for name, shape, axes in CASES:
        for dtype_name, dtype, tolerance in DTYPES:
            run_case(
                label,
                name,
                shape,
                axes,
                dtype_name,
                dtype,
                tolerance,
            )
    print(f"S02CJ_COMPARE_SUMMARY label={label} passed=4/4")


if __name__ == "__main__":
    main()
