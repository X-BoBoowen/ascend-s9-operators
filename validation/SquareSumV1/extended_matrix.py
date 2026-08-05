import itertools
import random

import torch
import torch_npu

import square_sum_v1_validation_lib


SEED = 2026072813
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
BOUNDARIES = (1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129,
              255, 257, 4095, 4097, 8191, 8192, 8193, 10000)


def tolerances(dtype):
    if dtype == torch.float16:
        return 4e-3, 4e-3
    if dtype == torch.bfloat16:
        return 4e-2, 4e-2
    return 4e-5, 4e-5


def run_case(name, input_cpu, axes, keep_dims):
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
    rtol, atol = tolerances(input_cpu.dtype)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
        msg=(
            f"{name}: dtype={input_cpu.dtype}, shape={tuple(input_cpu.shape)}, "
            f"axes={axes}, keep_dims={keep_dims}"
        ),
    )
    print(
        f"PASS {name}: dtype={input_cpu.dtype}, "
        f"shape={tuple(input_cpu.shape)}, axes={axes}, "
        f"keep_dims={keep_dims}",
        flush=True,
    )


def random_input(shape, dtype):
    return (torch.rand(shape, dtype=torch.float32) * 4.0 - 2.0).to(dtype)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    total = 0

    # Every important vector/reduction boundary, including the Excel maximum N.
    for dtype, length, keep_dims in itertools.product(
        DTYPES, BOUNDARIES, (False, True)
    ):
        run_case(
            f"rank1_{dtype}_{length}_{keep_dims}",
            random_input((length,), dtype),
            (-1,),
            keep_dims,
        )
        total += 1

    # Exercise all four optimized layouts and their guarded fallbacks.
    layouts = []
    for length in (31, 63, 64, 65, 129, 257, 4097, 10000):
        layouts.extend((
            ((3, length), (-1,)),
            ((3, length, 5), (1,)),
            ((length, 5), (0,)),
            ((2, 3, 5, length), (1, 3)),
            ((2, length, 5), (1,)),
        ))
    layouts.extend((
        ((2, 3, 5, 7, 11), (0, 2, 4)),
        ((2, 3, 5, 7, 11), (-5, -3, -1)),
        ((2, 3, 5, 7, 11), (1, 3)),
        ((2, 3, 5, 7, 11), (0, 1, 3)),
        ((2, 3, 5, 7, 11), (1, 2, 4)),
    ))
    for dtype, (shape, axes), keep_dims in itertools.product(
        DTYPES, layouts, (False, True)
    ):
        run_case(
            f"layout_{dtype}_{shape}_{axes}_{keep_dims}",
            random_input(shape, dtype),
            axes,
            keep_dims,
        )
        total += 1

    # The generated aclnn wrapper rejects a zero-length IntArray before it
    # reaches tiling. Exercise the contract-equivalent explicit all-axis form;
    # empty-axis metadata handling is audited separately in the host source.
    for dtype, shape, keep_dims in itertools.product(
        DTYPES,
        ((7,), (2, 3), (2, 3, 5), (2, 3, 5, 7, 11)),
        (False, True),
    ):
        run_case(
            f"empty_axis_{dtype}_{shape}_{keep_dims}",
            random_input(shape, dtype),
            tuple(range(len(shape))),
            keep_dims,
        )
        total += 1

    # NaN, infinities, signed zero, tiny values and square-overflow behavior.
    special_values = {
        torch.float16: (
            0.0, -0.0, float("inf"), -float("inf"), float("nan"),
            2 ** -14, -(2 ** -14), 300.0, -300.0,
        ),
        torch.bfloat16: (
            0.0, -0.0, float("inf"), -float("inf"), float("nan"),
            2 ** -126, -(2 ** -126), 1.0e10, -1.0e10,
        ),
        torch.float32: (
            0.0, -0.0, float("inf"), -float("inf"), float("nan"),
            2 ** -126, -(2 ** -126), 1.0e20, -1.0e20,
        ),
    }
    for dtype in DTYPES:
        values = torch.tensor(special_values[dtype], dtype=dtype).reshape(3, 3)
        for axes in ((-1,), (0,), (0, 1)):
            for keep_dims in (False, True):
                run_case(
                    f"special_{dtype}_{axes}_{keep_dims}",
                    values,
                    axes,
                    keep_dims,
                )
                total += 1

    # Broad randomized rank/axis/order/negative-axis coverage.
    dimensions = (1, 2, 3, 5, 7, 11, 17, 31, 33, 63, 65, 127, 129)
    for case_id in range(300):
        dtype = DTYPES[case_id % len(DTYPES)]
        rank = random.randint(1, 5)
        shape = tuple(random.choice(dimensions) for _ in range(rank))
        while torch.tensor(shape).prod().item() > 1_000_000:
            shape = tuple(random.choice(dimensions) for _ in range(rank))
        if case_id % 17 == 0:
            axes = tuple(range(rank))
        else:
            axis_count = random.randint(1, rank)
            normalized = random.sample(range(rank), axis_count)
            random.shuffle(normalized)
            axes = tuple(
                axis if random.random() < 0.5 else axis - rank
                for axis in normalized
            )
        keep_dims = bool(random.getrandbits(1))
        run_case(
            f"random_{case_id}",
            random_input(shape, dtype),
            axes,
            keep_dims,
        )
        total += 1

    torch.npu.synchronize()
    print(f"ALL EXTENDED PASS: {total} cases, seed={SEED}")


if __name__ == "__main__":
    main()
