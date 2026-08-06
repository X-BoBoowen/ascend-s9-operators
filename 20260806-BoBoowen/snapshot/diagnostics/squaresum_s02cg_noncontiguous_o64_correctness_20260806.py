import gc
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080611

CASES = (
    ("output33", (32, 33, 1024), (0, 2), False),
    ("output40", (32, 40, 1024), (0, 2), False),
    ("output47_rank4", (32, 1, 47, 1024), (0, 3), False),
    ("output48", (32, 48, 1024), (0, 2), False),
    ("output63", (32, 63, 1024), (0, 2), False),
    ("output64_tail1024", (32, 64, 1024), (0, 2), False),
    ("output64_tail8192", (64, 64, 8192), (0, 2), False),
    ("output64_tail16384", (16, 64, 16384), (0, 2), False),
    ("output64_multidim", (32, 4, 16, 1024), (0, 3), False),
    ("output64_keepdims", (32, 8, 8, 1024), (0, 3), True),
    ("output64_negative_axes", (16, 8, 8, 2048), (-1, 0), False),
    ("output65_fallback", (32, 65, 1024), (0, 2), False),
    ("rows15_fallback", (15, 64, 2185), (0, 2), False),
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
    print(f"SUMMARY_S02CG_O64 label={label} passed={passed}/{total}")


if __name__ == "__main__":
    main()
