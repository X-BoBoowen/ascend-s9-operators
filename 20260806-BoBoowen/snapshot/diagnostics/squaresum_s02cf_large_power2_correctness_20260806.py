import gc
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080609

# Public, route-derived boundary coverage for adaptive 8/4/2/1-row compact
# split-K and its normal/long UB chunk boundary.
CASES = (
    ("width8_normal", (64, 16, 128, 8), (0, 2), False),
    ("width4_normal", (64, 16, 256, 8), (0, 2), False),
    ("width2_normal", (64, 16, 512, 8), (0, 2), False),
    ("width1_normal", (64, 16, 1024, 8), (0, 2), False),
    ("width1_long_inner16", (64, 16, 1024, 16), (0, 2), False),
    ("width1_long_inner4", (64, 16, 4096, 4), (0, 2), False),
    ("inner4_normal", (64, 16, 1024, 4), (0, 2), False),
    ("grouped_tail7", (64, 7, 128, 8), (0, 2), False),
    ("rank5_long", (64, 2, 16, 1024, 16), (0, 3), False),
    ("rank5_outer_rows", (2, 64, 16, 1024, 8), (1, 3), False),
    ("negative_axes_keepdims", (2, 64, 16, 1024, 8), (-2, 1), True),
    ("output512_boundary", (64, 32, 1024, 16), (0, 2), False),
    ("output520_fallback", (64, 65, 1024, 8), (0, 2), False),
    ("nonpower_inner5_legacy", (64, 16, 64, 5), (0, 2), False),
)

DTYPES = (
    ("fp16", torch.float16, 3e-3),
    ("bf16", torch.bfloat16, 3e-2),
    ("fp32", torch.float32, 3e-4),
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
        torch.rand(shape, dtype=torch.float32, generator=generator) * 0.02
        - 0.01
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
    print(f"SUMMARY_S02CF_LARGE_POWER2 label={label} passed={passed}/{total}")


if __name__ == "__main__":
    main()
