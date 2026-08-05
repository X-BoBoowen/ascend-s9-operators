import statistics
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080501


CASES = (
    ("target_o64", (64, 1, 256, 64), (0, 2)),
    ("target_o128", (64, 4, 128, 32), (0, 2)),
    ("target_o512", (32, 8, 64, 64), (0, 2)),
    ("target_o1105", (32, 17, 64, 65), (0, 2)),
    ("target_o2015", (16, 31, 128, 65), (0, 2)),
    ("target_o2159", (32, 17, 64, 127), (0, 2)),
    ("control_small_fp4", (8, 17, 64, 65), (0, 2)),
    ("control_fp2", (200, 1000, 64), (0, 1)),
    ("control_fp3", (8, 17, 9, 33, 65), (0, 2, 4)),
)
DTYPES = (
    ("fp16", torch.float16),
    ("bf16", torch.bfloat16),
    ("fp32", torch.float32),
)


def tolerances(dtype):
    if dtype == torch.float16:
        return 3e-3, 3e-3
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    return 1e-4, 1e-4


def measure(label, case_name, shape, axes, dtype_name, dtype):
    generator = torch.Generator().manual_seed(SEED + len(shape))
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 4.0
        - 2.0
    ).to(dtype)
    expected = torch.sum(torch.square(input_cpu), dim=axes, keepdim=False)
    input_npu = input_cpu.npu()
    result = None
    for _ in range(20):
        result = square_sum_v1_validation_lib.square_sum_v1(
            input_npu, axes, False, list(expected.shape)
        )
    torch.npu.synchronize()

    repeats = 30
    samples = []
    for _ in range(7):
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
    print(
        f"RESULT label={label} case={case_name} dtype={dtype_name} "
        f"shape={shape} axes={axes} median_us={median:.6f} "
        f"samples_us={[round(value, 6) for value in samples]}"
    )


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    passed = 0
    for case_name, shape, axes in CASES:
        for dtype_name, dtype in DTYPES:
            measure(label, case_name, shape, axes, dtype_name, dtype)
            passed += 1
    print(f"SUMMARY label={label} passed={passed}/{len(CASES) * len(DTYPES)}")


if __name__ == "__main__":
    main()
