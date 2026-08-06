import statistics
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080602


CASES = (
    ("target_o8", (4, 8, 65536), (0, 2), True),
    ("target_o9", (4, 9, 65536), (0, 2), True),
    ("max_o16", (4, 16, 65536), (0, 2), True),
    ("o17_control", (4, 17, 65536), (0, 2), False),
    ("tail_boundary", (8, 8, 16385), (0, 2), True),
    ("tail_remainder", (4, 8, 65539), (0, 2), True),
    ("two_rows_long", (2, 8, 131072), (0, 2), True),
    ("work_units_below", (7, 8, 20000), (0, 2), False),
    ("rank5", (2, 4, 2, 2, 65539), (0, 2, 4), True),
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


def measure(label, name, shape, axes, expected_splitk, dtype_name, dtype, type_bytes):
    input_elements = product(shape)
    repeats = 20 if input_elements <= 1 << 20 else 10
    generator = torch.Generator().manual_seed(SEED + len(shape) + input_elements)
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 0.2
        - 0.1
    ).to(dtype)
    expected = torch.sum(torch.square(input_cpu), dim=axes, keepdim=False)
    input_npu = input_cpu.npu()
    result = None
    for _ in range(15):
        result = square_sum_v1_validation_lib.square_sum_v1(
            input_npu, axes, False, list(expected.shape)
        )
    torch.npu.synchronize()

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
    input_gbps = input_elements * type_bytes / median / 1000.0
    print(
        f"RESULT label={label} case={name} dtype={dtype_name} "
        f"expected_splitk={int(expected_splitk)} inputs={input_elements} "
        f"outputs={expected.numel()} median_us={median:.6f} "
        f"input_gbps={input_gbps:.6f} "
        f"samples_us={[round(value, 6) for value in samples]}"
    )


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    passed = 0
    for name, shape, axes, expected_splitk in CASES:
        for dtype_name, dtype, type_bytes in DTYPES:
            measure(
                label,
                name,
                shape,
                axes,
                expected_splitk,
                dtype_name,
                dtype,
                type_bytes,
            )
            passed += 1
    print(f"SUMMARY label={label} passed={passed}/{len(CASES) * len(DTYPES)}")


if __name__ == "__main__":
    main()
