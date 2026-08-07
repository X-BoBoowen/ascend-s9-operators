import gc
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080705

DTYPES = (
    ("fp16", torch.float16, 4e-3),
    ("bf16", torch.bfloat16, 4e-2),
    ("fp32", torch.float32, 4e-4),
)

# Public-spec shapes only.  The cases straddle every host-side S03L gate and
# exercise grouped-row tails; they are not derived from competition samples.
CASES = (
    # Group-width selection and partial final tasks.
    ("group7_width4_tail3", (16, 16, 7, 128, 8), (1, 3), False),
    ("group8_width8", (16, 16, 8, 128, 8), (1, 3), False),
    ("group9_width8_tail1", (16, 16, 9, 128, 8), (1, 3), True),
    ("group15_width8_tail7", (16, 16, 15, 128, 8), (3, 1), False),
    ("group16_width8", (16, 16, 16, 128, 8), (-4, -2), True),
    ("group17_width8_tail1", (16, 16, 17, 128, 8), (1, 3), False),
    ("group31_width8_tail7", (8, 16, 31, 128, 8), (1, 3), False),
    ("group32_width8", (8, 16, 32, 128, 8), (3, 1), True),
    ("group33_width8_tail1", (8, 16, 33, 128, 8), (-4, -2), False),

    # UB row-capacity boundaries select widths 8, 4, and 2, then fall back.
    ("last64_width8", (8, 32, 32, 64, 8), (1, 3), False),
    ("last128_width8", (8, 16, 32, 128, 8), (1, 3), False),
    ("last256_width4", (8, 8, 32, 256, 8), (1, 3), False),
    ("last512_width2", (8, 4, 32, 512, 8), (1, 3), True),
    ("last1024_fallback", (8, 2, 32, 1024, 8), (1, 3), False),

    # Inner-width alignment/capacity boundaries.
    ("inner7_fallback", (8, 32, 32, 64, 7), (1, 3), False),
    ("inner8_hit", (8, 32, 32, 64, 8), (1, 3), False),
    ("inner15_fallback", (8, 32, 32, 64, 15), (1, 3), True),
    ("inner16_hit", (8, 32, 32, 64, 16), (1, 3), False),
    ("inner24_hit", (8, 32, 32, 64, 24), (1, 3), False),
    ("inner32_hit", (8, 32, 32, 64, 32), (1, 3), True),
    ("inner64_hit", (8, 32, 32, 64, 64), (1, 3), False),
    ("inner128_fallback", (8, 32, 32, 64, 128), (1, 3), False),

    # The 32-task gate and 1 Mi-element input gate.
    ("tasks31_fallback", (31, 16, 8, 128, 8), (1, 3), False),
    ("tasks32_hit", (32, 16, 8, 128, 8), (1, 3), False),
    ("tasks33_hit", (33, 16, 8, 128, 8), (1, 3), True),
    ("input_below_gate", (31, 16, 2, 128, 8), (1, 3), False),
    ("input_at_gate", (32, 16, 2, 128, 8), (1, 3), False),
    ("input_above_gate", (33, 16, 2, 128, 8), (1, 3), True),

    # Last-reduction power-of-two gate and axis normalization.
    ("last63_fallback", (8, 33, 32, 63, 8), (1, 3), False),
    ("last64_hit", (8, 32, 32, 64, 8), (-4, -2), True),
    ("last65_fallback", (8, 32, 32, 65, 8), (3, 1), False),
    ("last127_fallback", (8, 17, 32, 127, 8), (1, 3), False),
    ("last129_fallback", (8, 16, 32, 129, 8), (1, 3), True),

    # A singleton between the reduction axes has no groupable retained row
    # and must stay on the existing rank-5 fallback route.
    ("singleton_gap_fallback", (64, 16, 1, 128, 8), (1, 3), False),
    ("singleton_gap_keep", (64, 16, 1, 128, 8), (-4, -2), True),
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
        SEED + case_index * 37 + product(shape) % 104729
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
        f"S03L_CORRECTNESS label={label} case={name} "
        f"dtype={dtype_name} shape={shape} axes={axes} "
        f"keep_dims={int(keep_dims)} inputs={input_cpu.numel()} "
        f"outputs={expected.numel()} max_abs={max_abs:.7g} PASS",
        flush=True,
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
        f"S03L_CORRECTNESS_SUMMARY label={label} "
        f"passed={passed}/{expected_count} seed={SEED}"
    )


if __name__ == "__main__":
    main()
