import gc
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080614

CASES = (
    ("rows15_fallback", (64, 15, 256, 33), (0, 2), False),
    ("rows16_inner17", (64, 16, 64, 17), (0, 2), False),
    ("rows17_inner31", (32, 17, 128, 31), (0, 2), False),
    ("rows24_inner33", (32, 24, 64, 33), (0, 2), False),
    ("rows31_inner65", (16, 31, 64, 65), (0, 2), False),
    ("rows32_inner64", (16, 32, 64, 64), (0, 2), False),
    ("rows39_inner127", (8, 39, 32, 127), (0, 2), False),
    ("rows40_inner128", (8, 40, 32, 128), (0, 2), False),
    ("rows41_fallback", (8, 41, 32, 129), (0, 2), False),
    ("rank5_rows32", (8, 2, 16, 64, 65), (0, 3), False),
    ("rank5_keepdims", (8, 2, 16, 64, 65), (0, 3), True),
    ("negative_axes", (16, 31, 64, 65), (-2, 0), False),
    ("unordered_axes", (16, 31, 64, 65), (2, 0), False),
    ("below_input_gate", (8, 16, 64, 33), (0, 2), False),
    ("inner16_fallback", (64, 16, 64, 16), (0, 2), False),
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


def run_case(label, index, name, shape, axes, keep_dims, dtype_info):
    dtype_name, dtype, tolerance = dtype_info
    elements = product(shape)
    generator = torch.Generator().manual_seed(SEED + index + elements)
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
        actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True
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
    for index, (name, shape, axes, keep_dims) in enumerate(CASES):
        for dtype_info in DTYPES:
            run_case(
                label, index, name, shape, axes, keep_dims, dtype_info
            )
            passed += 1
    print(f"SUMMARY_S02CL_ALIGNED_ROWS label={label} passed={passed}/45")


if __name__ == "__main__":
    main()
