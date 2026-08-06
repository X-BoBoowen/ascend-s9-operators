import gc
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080607

CASES = (
    ("min_outer_last128", (40, 64, 128, 2), (0, 2), False),
    ("outer41_tail_last256", (41, 73, 256, 2), (0, 2), False),
    ("width8_last512", (64, 16, 512, 2), (0, 2), False),
    ("width4_last1024", (64, 32, 1024, 2), (0, 2), False),
    ("width2_last2048", (40, 16, 2048, 2), (0, 2), False),
    ("width1_last4096", (40, 8, 4096, 2), (0, 2), False),
    ("width4_output_tail", (64, 63, 1024, 2), (0, 2), False),
    ("output_256", (40, 128, 128, 2), (0, 2), False),
    ("rank5_outer_output", (2, 40, 16, 1024, 2), (1, 3), False),
    ("negative_axes_keepdims", (2, 40, 16, 512, 2), (-2, 1), True),
    ("unordered_axes", (64, 31, 512, 2), (2, 0), False),
    ("below_outer_gate", (39, 64, 128, 2), (0, 2), False),
    ("non_power_last", (40, 32, 768, 2), (0, 2), False),
    ("above_output_gate", (40, 257, 128, 2), (0, 2), False),
)

DTYPES = (
    ("fp16", torch.float16, 3e-3),
    ("bf16", torch.bfloat16, 3e-2),
    ("fp32", torch.float32, 2e-4),
)


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def run_case(label, name, shape, axes, keep_dims, dtype_name, dtype, tol):
    elements = product(shape)
    generator = torch.Generator().manual_seed(
        SEED + elements + len(shape) + int(keep_dims)
    )
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 0.06
        - 0.03
    ).to(dtype)
    expected = torch.sum(
        torch.square(input_cpu), dim=axes, keepdim=keep_dims
    )
    input_npu = input_cpu.npu()
    result = square_sum_v1_validation_lib.square_sum_v1(
        input_npu, axes, keep_dims, list(expected.shape)
    )
    torch.npu.synchronize()
    actual = result.cpu()
    torch.testing.assert_close(
        actual, expected, rtol=tol, atol=tol, equal_nan=True
    )
    max_abs = float((actual.float() - expected.float()).abs().max())
    print(
        f"PASS label={label} case={name} dtype={dtype_name} "
        f"shape={shape} axes={axes} keep_dims={int(keep_dims)} "
        f"inputs={elements} outputs={expected.numel()} max_abs={max_abs:.6g}"
    )
    del actual, result, input_npu, expected, input_cpu
    gc.collect()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    passed = 0
    for name, shape, axes, keep_dims in CASES:
        for dtype_name, dtype, tol in DTYPES:
            run_case(
                label,
                name,
                shape,
                axes,
                keep_dims,
                dtype_name,
                dtype,
                tol,
            )
            passed += 1
    total = len(CASES) * len(DTYPES)
    print(f"SUMMARY_S02CB_INNER2 label={label} passed={passed}/{total}")


if __name__ == "__main__":
    main()
