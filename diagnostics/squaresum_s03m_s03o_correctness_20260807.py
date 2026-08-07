import argparse
import gc
import math

SEED = 2026080706
TRAILING_LIMITS = (200, 1000, 10000, 10000)


S03M_CASES = (
    ("group7_width4_tail3", (16, 16, 7, 128, 8), (1, 3), False),
    ("group8_width8", (16, 16, 8, 128, 8), (1, 3), False),
    ("group9_width8_tail1", (16, 16, 9, 128, 8), (1, 3), True),
    ("group15_width8_tail7", (16, 16, 15, 128, 8), (3, 1), False),
    ("group16_width8", (16, 16, 16, 128, 8), (-4, -2), True),
    ("group17_width8_tail1", (16, 16, 17, 128, 8), (1, 3), False),
    ("group31_width8_tail7", (8, 16, 31, 128, 8), (1, 3), False),
    ("group32_width8", (8, 16, 32, 128, 8), (3, 1), True),
    ("group33_width8_tail1", (8, 16, 33, 128, 8), (-4, -2), False),
    ("last64_width8", (8, 32, 32, 64, 8), (1, 3), False),
    ("last128_width8", (8, 16, 32, 128, 8), (1, 3), False),
    ("last256_width4", (8, 8, 32, 256, 8), (1, 3), False),
    ("last512_width2", (8, 4, 32, 512, 8), (1, 3), True),
    ("last1024_fallback", (8, 2, 32, 1024, 8), (1, 3), False),
    ("inner7_fallback", (8, 32, 32, 64, 7), (1, 3), False),
    ("inner8_hit", (8, 32, 32, 64, 8), (1, 3), False),
    ("inner15_fallback", (8, 32, 32, 64, 15), (1, 3), True),
    ("inner16_hit", (8, 32, 32, 64, 16), (1, 3), False),
    ("inner24_hit", (8, 32, 32, 64, 24), (1, 3), False),
    ("inner32_hit", (8, 32, 32, 64, 32), (1, 3), True),
    ("inner64_hit", (8, 32, 32, 64, 64), (1, 3), False),
    ("inner128_fallback", (8, 32, 32, 64, 128), (1, 3), False),
    ("tasks31_fallback", (31, 16, 8, 128, 8), (1, 3), False),
    ("tasks32_hit", (32, 16, 8, 128, 8), (1, 3), False),
    ("tasks33_hit", (33, 16, 8, 128, 8), (1, 3), True),
    ("input_below_gate", (31, 16, 2, 128, 8), (1, 3), False),
    ("input_at_gate", (32, 16, 2, 128, 8), (1, 3), False),
    ("input_above_gate", (33, 16, 2, 128, 8), (1, 3), True),
    ("last63_fallback", (8, 33, 32, 63, 8), (1, 3), False),
    ("last64_negative_axes", (8, 32, 32, 64, 8), (-4, -2), True),
    ("last65_fallback", (8, 32, 32, 65, 8), (3, 1), False),
    ("last127_fallback", (8, 17, 32, 127, 8), (1, 3), False),
    ("last129_fallback", (8, 16, 32, 129, 8), (1, 3), True),
    ("singleton_gap_fallback", (64, 16, 1, 128, 8), (1, 3), False),
    ("singleton_gap_keep", (64, 16, 1, 128, 8), (-4, -2), True),
)


S03N_CASES = (
    ("last1_fallback", (8, 200, 32, 1, 16), (1, 3), False),
    ("last2_arbitrary", (64, 64, 256, 2, 16), (0, 1, 3), False),
    ("last3_arbitrary", (64, 64, 256, 3, 16), (3, 1, 0), True),
    ("last7_arbitrary", (32, 32, 256, 7, 16), (-5, -4, -2), False),
    ("last8_power2", (8, 128, 256, 8, 16), (0, 1, 3), True),
    ("last9_arbitrary", (8, 128, 256, 9, 16), (1, 3, 0), False),
    ("last31_arbitrary", (8, 128, 32, 31, 16), (1, 3), False),
    ("last32_power2", (8, 64, 32, 32, 16), (3, 1), True),
    ("last33_arbitrary", (8, 64, 32, 33, 16), (-4, -2), False),
    ("last63_arbitrary", (8, 64, 32, 63, 16), (1, 3), False),
    ("last64_power2", (8, 32, 32, 64, 16), (1, 3), True),
    ("last65_arbitrary", (8, 32, 32, 65, 16), (3, 1), False),
    ("last127_arbitrary", (8, 17, 32, 127, 16), (1, 3), False),
    ("last128_power2", (8, 16, 32, 128, 16), (-4, -2), True),
    ("last129_arbitrary", (8, 16, 32, 129, 16), (1, 3), False),
)


S03O_CASES = (
    ("inner1_padded", (8, 64, 32, 128, 1), (1, 3), False),
    ("inner2_padded", (8, 64, 32, 128, 2), (1, 3), True),
    ("inner7_padded", (8, 64, 32, 128, 7), (3, 1), False),
    ("inner8_aligned_control", (8, 64, 32, 128, 8), (-4, -2), True),
    ("inner9_padded", (8, 64, 32, 128, 9), (1, 3), False),
    ("inner15_padded", (8, 64, 32, 128, 15), (1, 3), False),
    ("inner16_aligned_control", (8, 64, 32, 128, 16), (3, 1), True),
    ("inner17_padded", (8, 64, 32, 128, 17), (1, 3), False),
    ("inner31_padded", (8, 32, 32, 128, 31), (-4, -2), False),
    ("inner33_padded", (64, 200, 2, 11, 33), (1, 3), True),
    ("inner63_padded", (64, 200, 2, 11, 63), (3, 1), False),
    ("inner65_padded", (64, 200, 2, 11, 65), (1, 3), False),
    ("inner127_padded", (64, 200, 2, 11, 127), (-4, -2), True),
    ("inner129_padded", (64, 200, 2, 11, 129), (1, 3), False),
    ("inner199_padded", (64, 200, 2, 11, 199), (1, 3), False),
)


CASES_BY_STAGE = {
    "s03m": S03M_CASES,
    "s03n": S03N_CASES,
    "s03o": S03O_CASES,
}


def validate_public_case(case):
    name, shape, axes, _ = case
    if not 1 <= len(shape) <= 5:
        raise ValueError(f"{name}: rank is outside 1..5")
    checked = shape[-4:]
    limits = TRAILING_LIMITS[-len(checked) :]
    for value, limit in zip(checked, limits):
        if value < 1 or value > limit:
            raise ValueError(
                f"{name}: dimension {value} exceeds public limit {limit}"
            )
    normalized = []
    for axis in axes:
        value = axis + len(shape) if axis < 0 else axis
        if value < 0 or value >= len(shape) or value in normalized:
            raise ValueError(f"{name}: invalid or duplicate axis {axis}")
        normalized.append(value)


def run_case(label, stage, case_index, case, dtype_info):
    name, shape, axes, keep_dims = case
    dtype_name, dtype, tolerance = dtype_info
    generator = torch.Generator().manual_seed(
        SEED + case_index * 37 + math.prod(shape) % 104729
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
        f"SQUARESUM_STAGE_CORRECTNESS label={label} stage={stage} "
        f"case={name} dtype={dtype_name} shape={shape} axes={axes} "
        f"keep_dims={int(keep_dims)} inputs={input_cpu.numel()} "
        f"outputs={expected.numel()} max_abs={max_abs:.7g} PASS",
        flush=True,
    )
    del actual, result, input_npu, expected, input_cpu
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True, choices=tuple(CASES_BY_STAGE)
    )
    parser.add_argument("--label", default="unknown")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    cases = CASES_BY_STAGE[args.stage]
    for case in cases:
        validate_public_case(case)
    if args.validate_only:
        print(
            f"SQUARESUM_STAGE_CASES_VALID stage={args.stage} "
            f"cases={len(cases)}"
        )
        return

    global torch, square_sum_v1_validation_lib
    import torch
    import torch_npu
    import square_sum_v1_validation_lib

    torch.npu.config.allow_internal_format = False
    dtypes = (
        ("fp16", torch.float16, 4e-3),
        ("bf16", torch.bfloat16, 4e-2),
        ("fp32", torch.float32, 4e-4),
    )
    passed = 0
    for case_index, case in enumerate(cases):
        for dtype_info in dtypes:
            run_case(args.label, args.stage, case_index, case, dtype_info)
            passed += 1
    expected_count = len(cases) * len(dtypes)
    print(
        f"SQUARESUM_STAGE_CORRECTNESS_SUMMARY label={args.label} "
        f"stage={args.stage} passed={passed}/{expected_count} seed={SEED}"
    )


if __name__ == "__main__":
    main()
