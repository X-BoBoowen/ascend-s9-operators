import gc
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080607

# These cases exercise only public, shape-derived routing boundaries.  They do
# not encode or infer any official hidden case.
CASES = (
    ("output9_min_input", (32, 9, 1024), (0, 2), False),
    ("output15_rank4", (32, 3, 5, 1024), (0, 3), False),
    ("output16_min_rows", (16, 16, 2048), (0, 2), False),
    ("output17_rows17", (17, 17, 2048), (0, 2), False),
    ("output18_multidim", (32, 2, 9, 1024), (0, 3), False),
    ("output21_negative_axes", (16, 3, 7, 2048), (-1, 0), False),
    ("output24_three_reduce_axes", (8, 4, 24, 1024), (3, 0, 1), False),
    ("output31", (32, 31, 1024), (0, 2), False),
    ("output32_tail16384", (16, 32, 16384), (0, 2), False),
    ("output32_keepdims", (32, 4, 8, 1024), (0, 3), True),
    ("output33_fallback", (32, 33, 1024), (0, 2), False),
    ("tail1023_fallback", (33, 32, 1023), (0, 2), False),
    ("tail16385_fallback", (16, 32, 16385), (0, 2), False),
    ("rows15_fallback", (15, 32, 2185), (0, 2), False),
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
    print(f"SUMMARY_S02CE_O32 label={label} passed={passed}/{total}")


if __name__ == "__main__":
    main()
