import gc
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080704

DTYPES = (
    ("fp16", torch.float16, 3e-3),
    ("bf16", torch.bfloat16, 3e-2),
    ("fp32", torch.float32, 3e-4),
)

# Straddle every host-side gate for S03K tiling key 6, exercise odd tails,
# and cover equivalent higher-rank contiguous middle reductions.  None of
# these shapes are competition hidden cases or selected from platform data.
CASES = (
    ("inner2_hit", (64, 8192, 2), (1,), False),
    ("inner3_hit", (64, 4097, 3), (1,), False),
    ("inner4_hit", (64, 4096, 4), (1,), False),
    ("inner5_hit", (64, 4097, 5), (1,), False),
    ("inner7_hit", (64, 4097, 7), (1,), False),
    ("inner8_hit", (64, 2048, 8), (1,), False),
    ("inner9_fallback", (64, 2048, 9), (1,), False),
    ("rows39_fallback", (39, 8192, 8), (1,), False),
    ("rows40_hit", (40, 4096, 8), (1,), False),
    ("rows41_hit", (41, 4097, 8), (1,), False),
    ("rows73_uneven", (73, 4097, 8), (1,), False),
    ("reduce2047_fallback", (65, 2047, 8), (1,), False),
    ("reduce2048_hit", (64, 2048, 8), (1,), False),
    ("reduce2049_hit", (64, 2049, 8), (1,), False),
    ("input_below_gate", (63, 2048, 8), (1,), False),
    ("input_at_gate", (64, 2048, 8), (1,), False),
    ("long_reduce_inner2", (64, 16387, 2), (1,), False),
    ("rank4_axes12", (64, 32, 64, 8), (1, 2), False),
    ("rank5_flat_outer", (2, 32, 32, 64, 8), (2, 3), False),
    ("rank5_keep_dims", (2, 32, 32, 64, 8), (2, 3), True),
    ("negative_axes", (64, 32, 64, 8), (-3, -2), False),
    ("unsorted_axes", (64, 32, 64, 8), (2, 1), False),
    ("singleton_gap", (64, 32, 1, 64, 8), (1, 3), False),
)


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def run_case(label, case_index, case, dtype_info):
    name, shape, axes, keep_dims = case
    dtype_name, dtype, tolerance = dtype_info
    generator = torch.Generator().manual_seed(
        SEED + case_index * 17 + product(shape) % 104729
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
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
        equal_nan=True,
    )
    max_abs = float((actual.float() - expected.float()).abs().max())
    print(
        f"S03K_CORRECTNESS label={label} case={name} "
        f"dtype={dtype_name} shape={shape} axes={axes} "
        f"keep_dims={int(keep_dims)} inputs={input_cpu.numel()} "
        f"outputs={expected.numel()} max_abs={max_abs:.7g} PASS"
    )
    del actual, result, input_npu, expected, input_cpu
    gc.collect()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    passed = 0
    for case_index, case in enumerate(CASES):
        for dtype_info in DTYPES:
            run_case(label, case_index, case, dtype_info)
            passed += 1
    expected_count = len(CASES) * len(DTYPES)
    print(
        f"S03K_CORRECTNESS_SUMMARY label={label} "
        f"passed={passed}/{expected_count}"
    )


if __name__ == "__main__":
    main()
