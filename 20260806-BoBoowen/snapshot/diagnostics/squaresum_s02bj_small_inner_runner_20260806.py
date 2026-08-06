import statistics
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080601


CASES = (
    ("inner1_tail", (1, 262147, 1), (1,), False, True),
    ("inner2_tail", (1, 131075, 2), (1,), False, True),
    ("inner3_tail", (1, 87383, 3), (1,), False, True),
    ("inner4_tail", (1, 65539, 4), (1,), False, True),
    ("inner5_tail", (1, 52433, 5), (1,), False, True),
    ("inner6_tail", (1, 43693, 6), (1,), False, True),
    ("inner7_tail", (1, 37451, 7), (1,), False, True),
    ("inner8_tail", (1, 32771, 8), (1,), False, True),
    ("inner1_exact", (1, 262144, 1), (1,), False, True),
    ("inner4_outer3", (3, 32771, 4), (1,), False, True),
    ("inner7_outer5", (5, 8193, 7), (1,), False, True),
    ("singleton_gap", (2, 1, 32771, 4), (1, 2), False, True),
    ("keepdims_inner4", (1, 65539, 4), (1,), True, True),
    ("inner9_control", (1, 32771, 9), (1,), False, False),
    ("input_below_control", (1, 32767, 8), (1,), False, False),
)
DTYPES = (
    ("fp16", torch.float16, 2),
    ("bf16", torch.bfloat16, 2),
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
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    return 1e-4, 1e-4


def output_shape(shape, axes, keepdims):
    reduced = set(axes)
    if keepdims:
        return [1 if axis in reduced else extent
                for axis, extent in enumerate(shape)]
    return [extent for axis, extent in enumerate(shape)
            if axis not in reduced]


def measure(label, case, dtype_name, dtype, type_bytes):
    name, shape, axes, keepdims, expected_compact = case
    input_elements = product(shape)
    repeats = 30 if input_elements <= 1 << 20 else 10
    generator = torch.Generator().manual_seed(
        SEED + len(shape) * 97 + sum(shape) % 1009
    )
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 0.2
        - 0.1
    ).to(dtype)
    expected = torch.sum(
        torch.square(input_cpu), dim=axes, keepdim=keepdims
    )
    input_npu = input_cpu.npu()
    result = None
    expected_shape = output_shape(shape, axes, keepdims)
    for _ in range(15):
        result = square_sum_v1_validation_lib.square_sum_v1(
            input_npu, axes, keepdims, expected_shape
        )
    torch.npu.synchronize()

    samples = []
    for _ in range(7):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            result = square_sum_v1_validation_lib.square_sum_v1(
                input_npu, axes, keepdims, expected_shape
            )
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / repeats)

    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(
        result.cpu(), expected, rtol=rtol, atol=atol, equal_nan=True
    )
    median = statistics.median(samples)
    bandwidth = input_elements * type_bytes / median / 1000.0
    print(
        f"RESULT label={label} case={name} dtype={dtype_name} "
        f"expected_compact={int(expected_compact)} inputs={input_elements} "
        f"outputs={expected.numel()} median_us={median:.6f} "
        f"input_gbps={bandwidth:.6f} "
        f"samples_us={[round(value, 6) for value in samples]}"
    )


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    passed = 0
    for case in CASES:
        for dtype_name, dtype, type_bytes in DTYPES:
            measure(label, case, dtype_name, dtype, type_bytes)
            passed += 1
    print(f"SUMMARY label={label} passed={passed}/{len(CASES) * len(DTYPES)}")


if __name__ == "__main__":
    main()
