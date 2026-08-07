import itertools

import torch
import torch_npu

import square_sum_v1_validation_lib


SEED = 2026080703
DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def tolerances(dtype):
    if dtype == torch.float16:
        return 4e-3, 4e-3
    if dtype == torch.bfloat16:
        return 4e-2, 4e-2
    return 4e-5, 4e-5


def run_case(name, shape, axes, keep_dims, dtype):
    input_cpu = (
        torch.rand(shape, dtype=torch.float32) * 4.0 - 2.0
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
    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
        msg=(
            f"{name}: dtype={dtype}, shape={shape}, "
            f"axes={axes}, keep_dims={keep_dims}"
        ),
    )
    print(
        f"PASS {name}: dtype={dtype}, shape={shape}, "
        f"axes={axes}, keep_dims={keep_dims}",
        flush=True,
    )


def main():
    torch.manual_seed(SEED)
    total = 0

    # Generic fastPath2 boundary coverage around the new 2048-element route,
    # the 8192-element chunk boundary, and the inner-width guard at 64.
    for dtype, reduce_size, inner_size in itertools.product(
        DTYPES,
        (2047, 2048, 2049, 8191, 8192, 8193),
        (1, 2, 8, 31, 64, 65),
    ):
        keep_dims = bool((reduce_size + inner_size) & 1)
        run_case(
            f"middle_r{reduce_size}_i{inner_size}",
            (2, reduce_size, inner_size),
            (1,),
            keep_dims,
            dtype,
        )
        total += 1

    # The workspace tree switches at a 65536-element reduction.  Products
    # use only dimensions permitted by the public rank-4 specification.
    tree_shapes = (
        ((1, 255, 257, 2), (1, 2)),   # 65535
        ((1, 256, 256, 2), (1, 2)),   # 65536
        ((1, 80, 820, 2), (1, 2)),    # 65600
        ((1, 255, 257, 64), (1, 2)),
        ((1, 256, 256, 64), (1, 2)),
        ((1, 80, 820, 64), (1, 2)),
        ((1, 255, 257, 65), (1, 2)),
        ((1, 256, 256, 65), (1, 2)),
        ((1, 80, 820, 65), (1, 2)),
    )
    for dtype, (shape, axes) in itertools.product(DTYPES, tree_shapes):
        run_case(
            f"tree_{shape[1]}x{shape[2]}_i{shape[3]}",
            shape,
            axes,
            False,
            dtype,
        )
        total += 1

    torch.npu.synchronize()
    print(f"ALL S03J BOUNDARY PASS: {total} cases, seed={SEED}")


if __name__ == "__main__":
    main()
