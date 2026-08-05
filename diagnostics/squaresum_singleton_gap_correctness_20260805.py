import itertools

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080519
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
CASES = (
    ((7, 1, 11, 13), (0, 2)),
    ((7, 1, 11, 13), (2, 0)),
    ((7, 1, 11, 13), (-4, -2)),
    ((7, 1, 11, 13), (0, 2, 3)),
    ((7, 1, 11, 13), (3, 0, 2)),
    ((7, 1, 11, 13), (-1, -4, -2)),
    ((3, 1, 5, 1, 7), (0, 2, 4)),
    ((3, 1, 5, 1, 7), (4, 0, 2)),
    ((3, 1, 5, 1, 7), (-1, -5, -3)),
    ((32, 1, 64, 1024), (0, 2)),
    ((32, 1, 64, 1024), (0, 2, 3)),
    ((128, 1, 64, 64), (0, 2)),
)


def tolerance(dtype):
    if dtype == torch.float16:
        return 4e-3, 4e-3
    if dtype == torch.bfloat16:
        return 4e-2, 4e-2
    return 2e-4, 2e-4


def main():
    torch.manual_seed(SEED)
    passed = 0
    total = len(CASES) * len(DTYPES) * 2
    for case_index, ((shape, axes), dtype, keep_dims) in enumerate(
        itertools.product(CASES, DTYPES, (False, True))
    ):
        generator = torch.Generator().manual_seed(SEED + case_index)
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
        rtol, atol = tolerance(dtype)
        torch.testing.assert_close(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            msg=(
                f"shape={shape}, axes={axes}, dtype={dtype}, "
                f"keep_dims={keep_dims}"
            ),
        )
        passed += 1
        print(
            f"PASS {passed}/{total}: shape={shape}, axes={axes}, "
            f"dtype={dtype}, keep_dims={keep_dims}",
            flush=True,
        )
    print(f"SUMMARY_SINGLETON_CORRECTNESS={passed}/{total}")


if __name__ == "__main__":
    main()
