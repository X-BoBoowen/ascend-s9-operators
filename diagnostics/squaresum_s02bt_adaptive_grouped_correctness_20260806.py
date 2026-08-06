import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080608
DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def tolerances(dtype):
    if dtype == torch.float16:
        return 3e-3, 3e-3
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    return 1e-4, 1e-4


def run_case(name, shape, axes, keep_dims, dtype, seed_offset):
    generator = torch.Generator().manual_seed(SEED + seed_offset)
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 0.2
        - 0.1
    ).to(dtype)
    expected = torch.sum(
        torch.square(input_cpu), dim=axes, keepdim=keep_dims
    )
    result = square_sum_v1_validation_lib.square_sum_v1(
        input_cpu.npu(), axes, keep_dims, list(expected.shape)
    )
    torch.npu.synchronize()
    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(
        result.cpu(), expected, rtol=rtol, atol=atol, equal_nan=True
    )
    print(
        f"PASS name={name} shape={shape} axes={axes} "
        f"keep_dims={int(keep_dims)} dtype={dtype}"
    )


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    cases = []
    for last_reduce in (2, 4, 8, 16, 32, 64):
        cases.append(
            (
                f"last_reduce_{last_reduce}",
                (3, 256, last_reduce, 2),
                (0, 2),
                False,
            )
        )
    for inner in (2, 3, 4, 5, 7, 8, 9, 15, 16):
        cases.append(
            (
                f"tail257_inner_{inner}",
                (3, 257, 64, inner),
                (0, 2),
                False,
            )
        )
    cases.extend(
        (
            ("width4_exact", (32, 4, 64, 2), (2,), False),
            ("width4_tail", (16, 7, 64, 3), (2,), False),
            ("width2_exact", (32, 2, 64, 4), (2,), False),
            ("width2_tail", (16, 3, 64, 5), (2,), False),
            ("width1", (32, 1, 64, 8), (2,), False),
            (
                "outer_reduce_groups",
                (2, 3, 9, 7, 64, 4),
                (1, 4),
                False,
            ),
            ("keep_dims", (3, 9, 64, 4), (0, 2), True),
            ("negative_unsorted", (3, 257, 64, 8), (-2, 0), False),
        )
    )

    passed = 0
    for case_index, (name, shape, axes, keep_dims) in enumerate(cases):
        for dtype_index, dtype in enumerate(DTYPES):
            run_case(
                name,
                shape,
                axes,
                keep_dims,
                dtype,
                case_index * len(DTYPES) + dtype_index,
            )
            passed += 1
    print(
        f"SUMMARY_S02BT_ADAPTIVE_GROUPED label={label} "
        f"passed={passed}/{len(cases) * len(DTYPES)}"
    )


if __name__ == "__main__":
    main()
