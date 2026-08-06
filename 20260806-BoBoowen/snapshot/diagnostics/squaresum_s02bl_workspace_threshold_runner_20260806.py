import statistics
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080602


# These cases sweep the generic fastPath2 workspace boundary.  They are
# deliberately expressed as shape/axis families rather than platform cases.
CASES = (
    ("middle_i8_32760", (1, 4095, 8), (1,), False, False),
    ("middle_i8_32768", (1, 4096, 8), (1,), False, True),
    ("middle_i8_65536", (1, 8192, 8), (1,), False, True),
    ("middle_i8_131072", (1, 16384, 8), (1,), False, True),
    ("middle_i8_196608", (1, 24576, 8), (1,), False, True),
    ("middle_i8_262136", (1, 32767, 8), (1,), False, True),
    ("middle_i2_32768", (1, 16384, 2), (1,), False, True),
    ("middle_i4_32768", (1, 8192, 4), (1,), False, True),
    ("middle_i16_32768", (1, 2048, 16), (1,), False, True),
    ("middle_i32_65536", (1, 2048, 32), (1,), False, True),
    ("middle_i64_131072", (1, 2048, 64), (1,), False, True),
    ("middle_i128_262144", (1, 2048, 128), (1,), False, True),
    ("middle_outer8_i8_262080", (8, 4095, 8), (1,), False, True),
    ("middle_keepdims_i8_131072", (1, 16384, 8), (1,), True, True),
    ("middle_gap_i8_131072", (1, 16384, 1, 8), (1, 2), False, True),
    ("middle_reduce_below", (1, 2047, 128), (1,), False, False),
    ("last_o8_262136_control", (8, 32767), (1,), False, False),
    ("last_o1_262143_control", (1, 262143), (1,), False, False),
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


def output_shape(shape, axes, keepdims):
    reduced = set(axes)
    if keepdims:
        return [1 if axis in reduced else extent
                for axis, extent in enumerate(shape)]
    return [extent for axis, extent in enumerate(shape)
            if axis not in reduced]


def tolerances(dtype):
    if dtype == torch.float16:
        return 3e-3, 3e-3
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    return 1e-4, 1e-4


def measure(label, case, dtype_name, dtype, type_bytes):
    name, shape, axes, keepdims, target_lower_threshold = case
    input_elements = product(shape)
    repeats = 40 if input_elements <= 1 << 17 else 20
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
    expected_shape = output_shape(shape, axes, keepdims)
    input_npu = input_cpu.npu()
    result = None
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
        f"target_lower_threshold={int(target_lower_threshold)} "
        f"inputs={input_elements} outputs={expected.numel()} "
        f"median_us={median:.6f} input_gbps={bandwidth:.6f} "
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
