import torch
import torch_npu

import square_sum_v1_validation_lib


SEED = 2026072608
DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


def tolerances(dtype):
    if dtype == torch.float16:
        return 2e-3, 2e-3
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 2e-5, 2e-5


def expected_shape(expected):
    return list(expected.shape)


def run_case(name, input_cpu, axis, keep_dims):
    expected = torch.sum(
        torch.square(input_cpu),
        dim=axis,
        keepdim=keep_dims,
    )
    actual = square_sum_v1_validation_lib.square_sum_v1(
        input_cpu.npu(),
        axis,
        keep_dims,
        expected_shape(expected),
    ).cpu()
    rtol, atol = tolerances(input_cpu.dtype)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    print(
        f"PASS {name}: dtype={input_cpu.dtype}, "
        f"shape={tuple(input_cpu.shape)}, axis={axis}, "
        f"keep_dims={keep_dims}, output={tuple(actual.shape)}"
    )


def main():
    torch.manual_seed(SEED)
    cases = (
        ((7,), (-1,), True),
        ((17,), (0,), False),
        ((2, 3, 5), (2,), False),
        ((2, 3, 5), (-2,), True),
        ((2, 3, 5), (0, 2), False),
        ((2, 3, 5), (-1, -3), True),
        ((2, 3, 5), (0, 1, 2), False),
        ((2, 3, 5), (0, 1, 2), True),
        ((3, 1, 7, 11), (1, 3), False),
        ((3, 1, 7, 11), (0, 2), True),
        ((2, 3, 5, 7, 11), (1, 3), False),
        ((2, 3, 5, 7, 11), (0, 2, 4), True),
        ((23, 131), (1,), False),
        ((41, 17, 19), (0, 2), False),
        ((1, 1, 1, 1, 1), (0, 1, 2, 3, 4), False),
    )
    total = 0
    for dtype in DTYPES:
        for case_id, (shape, axis, keep_dims) in enumerate(cases):
            input_cpu = (
                torch.rand(shape, dtype=torch.float32) * 4.0 - 2.0
            ).to(dtype)
            run_case(
                f"{dtype}_case{case_id}",
                input_cpu,
                axis,
                keep_dims,
            )
            total += 1

    overflow = torch.tensor(
        [300.0, -300.0, 1.0],
        dtype=torch.float16,
    )
    run_case("float16_square_overflow", overflow, (0,), False)
    total += 1
    torch.npu.synchronize()
    print(f"ALL PASS: {total} directed cases")


if __name__ == "__main__":
    main()
