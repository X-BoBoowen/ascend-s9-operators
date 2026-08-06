import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080603


CASES = (
    ("one_row_o1", (1, 1, 262144), (0, 2)),
    ("two_rows_o2", (2, 2, 131071), (0, 2)),
    ("three_rows_o5", (3, 5, 57345), (0, 2)),
    ("four_rows_o8", (4, 8, 65536), (0, 2)),
    ("four_rows_o9_remainder", (4, 9, 65539), (0, 2)),
    ("four_rows_o15", (4, 15, 65539), (0, 2)),
    ("four_rows_o16", (4, 16, 65539), (0, 2)),
    ("eight_rows_boundary", (8, 8, 16385), (0, 2)),
    ("rank4_trailing_pair", (2, 4, 2, 65539), (0, 2, 3)),
    ("rank5_interleaved", (3, 3, 2, 5, 32769), (0, 2, 4)),
    ("reordered_axes", (4, 8, 65539), (2, 0)),
    ("negative_axes", (4, 8, 65539), (-1, -3)),
)
DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def tolerances(dtype):
    if dtype == torch.float16:
        return 3e-3, 3e-3
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    return 1e-4, 1e-4


def main():
    passed = 0
    total = len(CASES) * len(DTYPES) * 2
    for case_index, (name, shape, axes) in enumerate(CASES):
        for dtype_index, dtype in enumerate(DTYPES):
            generator = torch.Generator().manual_seed(
                SEED + case_index * len(DTYPES) + dtype_index
            )
            input_cpu = (
                torch.rand(shape, dtype=torch.float32, generator=generator)
                * 0.2
                - 0.1
            ).to(dtype)
            input_npu = input_cpu.npu()
            for keep_dims in (False, True):
                expected = torch.sum(
                    torch.square(input_cpu),
                    dim=axes,
                    keepdim=keep_dims,
                )
                result = square_sum_v1_validation_lib.square_sum_v1(
                    input_npu,
                    axes,
                    keep_dims,
                    list(expected.shape),
                )
                torch.npu.synchronize()
                rtol, atol = tolerances(dtype)
                torch.testing.assert_close(
                    result.cpu(),
                    expected,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                )
                passed += 1
                print(
                    f"PASS {passed}/{total} case={name} dtype={dtype} "
                    f"shape={shape} axes={axes} keep_dims={keep_dims}"
                )
    print(f"SUMMARY_S02BN_STRUCTURAL_CORRECTNESS={passed}/{total}")


if __name__ == "__main__":
    main()
