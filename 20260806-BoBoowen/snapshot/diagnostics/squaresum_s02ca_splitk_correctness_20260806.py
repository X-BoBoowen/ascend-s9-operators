import gc
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080605

CASES = (
    ("min_outer", (40, 64, 64, 4), (0, 2), False),
    ("outer41_output_tail", (41, 73, 32, 7), (0, 2), False),
    ("last_reduce_2", (256, 64, 2, 8), (0, 2), False),
    ("last_reduce_4", (128, 64, 4, 8), (0, 2), False),
    ("last_reduce_8", (64, 64, 8, 8), (0, 2), False),
    ("last_reduce_16", (40, 64, 16, 8), (0, 2), False),
    ("inner_5_output_tail", (40, 101, 64, 5), (0, 2), False),
    ("inner_15_output_tail", (40, 34, 32, 15), (0, 2), False),
    ("inner_16", (40, 16, 64, 16), (0, 2), False),
    ("rank5_inner_8", (64, 16, 32, 2, 4), (0, 2), False),
    ("rank5_outer_output", (2, 40, 16, 64, 4), (1, 3), False),
    ("negative_axes_keepdims", (2, 40, 16, 64, 4), (-2, 1), True),
)

DTYPES = (
    ("fp16", torch.float16, 3e-3),
    ("bf16", torch.bfloat16, 3e-2),
    ("fp32", torch.float32, 1e-4),
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
        torch.rand(shape, dtype=torch.float32, generator=generator) * 0.2
        - 0.1
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
    print(f"SUMMARY_S02CA_SPLITK label={label} passed={passed}/{total}")


if __name__ == "__main__":
    main()
