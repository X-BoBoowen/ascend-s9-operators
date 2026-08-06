import gc
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080619

DTYPES = (
    ("fp16", torch.float16, 3e-3),
    ("bf16", torch.bfloat16, 3e-2),
    ("fp32", torch.float32, 3e-4),
)

# These cases straddle every host-side gate for tiling key 15 and also cover
# equivalent higher-rank representations of a contiguous middle reduction.
CASES = (
    ("inner8_fallback", (40, 4096, 8), (1,), False),
    ("inner9_hit", (40, 2913, 9), (1,), False),
    ("inner15_hit", (40, 2048, 15), (1,), False),
    ("inner16_hit", (40, 2048, 16), (1,), False),
    ("inner17_hit", (40, 2048, 17), (1,), False),
    ("inner31_hit", (40, 1024, 31), (1,), False),
    ("inner32_hit", (40, 1024, 32), (1,), False),
    ("inner33_hit", (40, 1024, 33), (1,), False),
    ("inner63_hit", (40, 512, 63), (1,), False),
    ("inner64_hit", (40, 512, 64), (1,), False),
    ("inner65_hit", (40, 512, 65), (1,), False),
    ("inner127_hit", (40, 512, 127), (1,), False),
    ("inner128_hit", (40, 512, 128), (1,), False),
    ("inner129_hit", (40, 512, 129), (1,), False),
    ("inner130_fallback", (40, 512, 130), (1,), False),
    ("rows39_fallback", (39, 2048, 17), (1,), False),
    ("rows40_hit", (40, 2048, 17), (1,), False),
    ("rows41_hit", (41, 2048, 17), (1,), False),
    ("rows79_uneven", (79, 1024, 33), (1,), False),
    ("rows83_uneven", (83, 1024, 33), (1,), False),
    ("reduce511_fallback", (40, 511, 65), (1,), False),
    ("reduce512_hit", (40, 512, 65), (1,), False),
    ("reduce4031_hit", (40, 4031, 17), (1,), False),
    ("reduce4033_hit", (40, 4033, 17), (1,), False),
    ("input_below_gate", (40, 2912, 9), (1,), False),
    ("input_above_gate", (40, 2913, 9), (1,), False),
    ("rank4_axes12", (40, 31, 17, 65), (1, 2), False),
    ("rank5_flat_outer", (2, 20, 31, 17, 65), (2, 3), False),
    ("rank5_keep_dims", (2, 20, 31, 17, 65), (2, 3), True),
    ("negative_axis", (40, 31, 17, 65), (-3, -2), False),
    ("unsorted_axes", (40, 31, 17, 65), (2, 1), False),
    (
        "singleton_gap_contiguous",
        (40, 65, 1, 129, 9),
        (1, 3),
        False,
    ),
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
        f"S02CN_CORRECTNESS label={label} case={name} "
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
        f"S02CN_CORRECTNESS_SUMMARY label={label} "
        f"passed={passed}/{expected_count}"
    )


if __name__ == "__main__":
    main()
