import random

import torch
import torch_npu

import greater_validation_lib


SEED = 2026072602


def make_tensor(shape, dtype):
    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        return torch.randn(shape, dtype=dtype)
    if dtype == torch.int32:
        return torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
        )
    return torch.randint(-128, 128, shape, dtype=dtype)


def run_case(name, self_cpu, other_cpu):
    expected = torch.gt(self_cpu, other_cpu)
    actual = greater_validation_lib.greater(
        self_cpu.npu(),
        other_cpu.npu(),
    ).cpu()
    if not torch.equal(actual, expected):
        mismatch = torch.count_nonzero(actual != expected).item()
        raise AssertionError(
            f"{name}: mismatch={mismatch}, "
            f"output={tuple(expected.shape)}, dtype={self_cpu.dtype}"
        )
    print(
        f"PASS {name}: dtype={self_cpu.dtype}, "
        f"self={tuple(self_cpu.shape)}, other={tuple(other_cpu.shape)}"
    )


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    cases = []
    for dtype in (
        torch.float16,
        torch.float32,
        torch.bfloat16,
        torch.int32,
        torch.int8,
    ):
        cases.extend([
            (
                f"{dtype}_same_unaligned",
                make_tensor((17, 131), dtype),
                make_tensor((17, 131), dtype),
            ),
            (
                f"{dtype}_scalar",
                make_tensor((9, 7, 5), dtype),
                make_tensor((), dtype),
            ),
            (
                f"{dtype}_last_dim_broadcast",
                make_tensor((4, 3, 11), dtype),
                make_tensor((1, 11), dtype),
            ),
            (
                f"{dtype}_prefix_broadcast",
                make_tensor((2, 1, 5, 7), dtype),
                make_tensor((1, 3, 1, 7), dtype),
            ),
        ])

    specials = torch.tensor(
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            0.0,
            -0.0,
            1.0,
            -1.0,
            float("inf"),
        ],
        dtype=torch.float32,
    )
    cases.append((
        "float32_nan_inf",
        specials,
        torch.tensor(
            [0.0, float("inf"), float("-inf"), -0.0,
             0.0, float("nan"), -2.0, float("-inf")],
            dtype=torch.float32,
        ),
    ))
    for dtype in (torch.float16, torch.bfloat16):
        cases.append((
            f"{dtype}_nan_inf_broadcast",
            specials.to(dtype).reshape(8, 1),
            torch.tensor(
                [float("-inf"), -1.0, -0.0, 0.0,
                 1.0, float("inf"), float("nan")],
                dtype=dtype,
            ).reshape(1, 7),
        ))
    cases.append((
        "int32_extremes",
        torch.tensor(
            [
                torch.iinfo(torch.int32).min,
                torch.iinfo(torch.int32).max,
                16777217,
                -16777217,
            ],
            dtype=torch.int32,
        ),
        torch.tensor(
            [
                torch.iinfo(torch.int32).max,
                torch.iinfo(torch.int32).min,
                16777216,
                -16777216,
            ],
            dtype=torch.int32,
        ),
    ))
    int32_boundaries = torch.tensor(
        [
            -(2**31),
            -(2**31) + 1,
            -65537,
            -65536,
            -65535,
            -1,
            0,
            1,
            65534,
            65535,
            65536,
            65537,
            (2**31) - 2,
            (2**31) - 1,
        ],
        dtype=torch.int32,
    )
    cases.append((
        "int32_high_low16_cartesian",
        int32_boundaries.reshape(-1, 1),
        int32_boundaries.reshape(1, -1),
    ))
    cases.append((
        "empty_output",
        torch.empty((0, 3), dtype=torch.float16),
        torch.empty((1, 3), dtype=torch.float16),
    ))

    for case in cases:
        run_case(*case)
    torch.npu.synchronize()
    print(f"ALL PASS: {len(cases)} cases")


if __name__ == "__main__":
    main()
