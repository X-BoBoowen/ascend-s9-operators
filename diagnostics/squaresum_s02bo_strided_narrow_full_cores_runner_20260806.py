import gc
import statistics
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080604


CASES = (
    ("small_input_control", (64, 64, 64, 1), (0, 2), False),
    ("inner1", (64, 256, 64, 1), (0, 2), True),
    ("inner2", (64, 128, 64, 2), (0, 2), True),
    ("inner4", (64, 64, 64, 4), (0, 2), True),
    ("inner8", (64, 32, 64, 8), (0, 2), True),
    ("inner16", (64, 16, 64, 16), (0, 2), True),
    ("inner32", (64, 8, 64, 32), (0, 2), True),
    ("inner33_control", (64, 8, 64, 33), (0, 2), False),
    ("inner64_control", (64, 4, 64, 64), (0, 2), False),
    ("one_output_control", (256, 1, 4096, 1), (0, 2), True),
    ("rank5_inner8", (64, 16, 32, 2, 4), (0, 2), True),
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


def measure(label, name, shape, axes, expected_route, dtype_name, dtype, type_bytes):
    input_elements = product(shape)
    generator = torch.Generator().manual_seed(SEED + len(shape) + input_elements)
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 0.2
        - 0.1
    ).to(dtype)
    expected = torch.sum(torch.square(input_cpu), dim=axes, keepdim=False)
    input_npu = input_cpu.npu()
    result = None
    for _ in range(8):
        result = square_sum_v1_validation_lib.square_sum_v1(
            input_npu, axes, False, list(expected.shape)
        )
    torch.npu.synchronize()

    repeats = 5 if input_elements >= 1 << 20 else 10
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
        f"RESULT label={label} case={name} dtype={dtype_name} "
        f"expected_route={int(expected_route)} inputs={input_elements} "
        f"outputs={expected.numel()} median_us={median:.6f} "
        f"input_gbps={input_elements * type_bytes / median / 1000.0:.6f} "
        f"samples_us={[round(value, 6) for value in samples]}"
    )
    del result, input_npu, expected, input_cpu
    gc.collect()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    passed = 0
    for name, shape, axes, expected_route in CASES:
        for dtype_name, dtype, type_bytes in DTYPES:
            measure(
                label,
                name,
                shape,
                axes,
                expected_route,
                dtype_name,
                dtype,
                type_bytes,
            )
            passed += 1
    print(f"SUMMARY label={label} passed={passed}/{len(CASES) * len(DTYPES)}")


if __name__ == "__main__":
    main()
