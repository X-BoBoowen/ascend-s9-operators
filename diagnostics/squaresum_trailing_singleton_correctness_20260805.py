import itertools

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080525
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
CASES = (
    ("empty_control", (0, 1), (0,)),
    ("empty_middle", (3, 0, 1), (1,)),
    ("empty_output", (0, 3, 1), (1,)),
    ("empty_gap", (2, 0, 1, 3, 1), (1, 3)),
    ("empty_full_reduce", (0, 3), ()),
    ("unit_reduce", (1, 1), (0,)),
    ("odd_plain", (32769, 1), (0,)),
    ("two_outer", (2, 32769, 1), (1,)),
    ("eight_outer", (8, 257, 1), (1,)),
    ("nine_outer", (9, 257, 1), (1,)),
    ("thirty_two_outer", (32, 257, 1), (1,)),
    ("one_zero_two_four_outer", (1024, 257, 1), (1,)),
    ("one_zero_two_five_outer", (1025, 257, 1), (1,)),
    ("gap", (131, 1, 251, 1), (0, 2)),
    ("gap_reordered", (131, 1, 251, 1), (2, 0)),
    ("gap_negative", (131, 1, 251, 1), (-2, -4)),
    ("prefix_gap", (3, 131, 1, 251, 1), (1, 3)),
    ("two_trailing", (32769, 1, 1), (0,)),
    ("width2_control", (32769, 2), (0,)),
    ("reduced_singleton_control", (2, 1, 3), (1,)),
)


def tolerance(dtype, expected):
    scale = max(float(expected.abs().max()), 1.0)
    if dtype == torch.float16:
        return 4e-3, max(4e-3, scale * 4e-3)
    if dtype == torch.bfloat16:
        return 4e-2, max(4e-2, scale * 4e-2)
    return 2e-4, max(2e-4, scale * 2e-4)


def main():
    combinations = tuple(
        itertools.product(
            enumerate(CASES),
            enumerate(DTYPES),
            (False, True),
        )
    )
    passed = 0
    for (case_index, (case, shape, axes)), (
        dtype_index,
        dtype,
    ), keep_dims in combinations:
        generator = torch.Generator().manual_seed(
            SEED + case_index * 100 + dtype_index * 10 + keep_dims
        )
        input_cpu = (
            torch.rand(
                shape,
                dtype=torch.float32,
                generator=generator,
            )
            * 0.06
            - 0.03
        ).to(dtype)
        expected = torch.sum(
            torch.square(input_cpu),
            dim=axes,
            keepdim=keep_dims,
        )
        actual = square_sum_v1_validation_lib.square_sum_v1(
            input_cpu.npu(),
            axes,
            keep_dims,
            list(expected.shape),
        ).cpu()
        rtol, atol = tolerance(dtype, expected)
        torch.testing.assert_close(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        passed += 1
        print(
            f"PASS {passed}/{len(combinations)}: "
            f"case={case}, shape={shape}, axes={axes}, "
            f"dtype={dtype}, keep_dims={keep_dims}",
            flush=True,
        )
    print(
        f"SUMMARY_TRAILING_SINGLETON_CORRECTNESS="
        f"{passed}/{len(combinations)}"
    )


if __name__ == "__main__":
    main()
