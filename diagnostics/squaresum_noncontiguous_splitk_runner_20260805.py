import statistics
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080503


CASES = (
    ("fp3_o1_boundary", (8, 1, 32768), (0, 2), True),
    ("fp3_o2", (8, 2, 32768), (0, 2), True),
    ("fp3_o8", (4, 8, 65536), (0, 2), True),
    ("fp3_o9_control", (4, 9, 65536), (0, 2), False),
    ("fp3_reduce_below", (8, 8, 4095), (0, 2), False),
    ("fp3_input_below", (1, 1, 32768), (0, 2), False),
    ("fp3_rank5_o8", (8, 2, 16, 4, 4096), (0, 2, 4), True),
    ("fp4_o1_boundary", (64, 1, 4096, 1), (0, 2), True),
    ("fp4_o4", (64, 1, 4096, 4), (0, 2), True),
    ("fp4_o8", (64, 1, 2048, 8), (0, 2), True),
    ("fp4_o9_control", (64, 1, 2048, 9), (0, 2), False),
    ("fp4_reduce_below", (8, 1, 4095, 8), (0, 2), False),
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
    repeats = 30 if input_elements <= 1 << 20 else 10
    generator = torch.Generator().manual_seed(SEED + len(shape))
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 0.2
        - 0.1
    ).to(dtype)
    expected = torch.sum(torch.square(input_cpu), dim=axes, keepdim=False)
    assert expected.numel() <= 9
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
