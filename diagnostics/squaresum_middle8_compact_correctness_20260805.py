import itertools

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080522
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
CASES = (
    ("threshold_minus_one", (32767, 8), (0,)),
    ("threshold_exact", (32768, 8), (0,)),
    ("odd_plain", (32769, 8), (0,)),
    ("odd_singleton_gap", (131, 1, 251, 8), (0, 2)),
    ("two_outer", (2, 131, 1, 251, 8), (1, 3)),
    ("two_outer_reordered", (2, 131, 1, 251, 8), (3, 1)),
    ("two_outer_negative", (2, 131, 1, 251, 8), (-2, -4)),
    ("inner_seven_control", (37450, 7), (0,)),
    ("inner_nine_control", (29128, 9), (0,)),
    ("inner_sixteen_control", (16385, 16), (0,)),
)


def tolerance(dtype, expected):
    scale = max(float(expected.abs().max()), 1.0)
    if dtype == torch.float16:
        return 4e-3, max(4e-3, scale * 4e-3)
    if dtype == torch.bfloat16:
        return 4e-2, max(4e-2, scale * 4e-2)
    return 2e-4, max(2e-4, scale * 2e-4)


def main():
    passed = 0
    combinations = tuple(
        itertools.product(
            enumerate(CASES),
            enumerate(DTYPES),
            (False, True),
        )
    )
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
    print(f"SUMMARY_MIDDLE8_CORRECTNESS={passed}/{len(combinations)}")


if __name__ == "__main__":
    main()
