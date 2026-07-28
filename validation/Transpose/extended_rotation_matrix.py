import torch
import torch_npu

import transpose_validation_lib


SEED = 2026072701
DTYPES = (
    torch.float32,
    torch.float16,
    torch.int32,
    torch.int8,
)


def make_tensor(shape, dtype):
    if dtype in (torch.float32, torch.float16):
        return torch.randn(shape, dtype=dtype)
    if dtype == torch.int32:
        return torch.randint(
            -(2**31),
            2**31 - 1,
            shape,
            dtype=dtype,
        )
    return torch.randint(-128, 128, shape, dtype=dtype)


def bit_pattern_tensor(shape, dtype):
    count = 1
    for extent in shape:
        count *= extent
    if dtype == torch.float32:
        pattern = torch.tensor(
            [
                0,
                -2147483648,
                1065353216,
                -1082130432,
                2139095040,
                -8388608,
                2143289345,
                -4194303,
                1,
                -2147483647,
            ],
            dtype=torch.int32,
        )
        return pattern.repeat((count + pattern.numel() - 1) //
                              pattern.numel())[:count].view(
                                  torch.float32).reshape(shape)
    pattern = torch.tensor(
        [
            0,
            -32768,
            15360,
            -17408,
            31744,
            -1024,
            32257,
            -511,
            1,
            -32767,
        ],
        dtype=torch.int16,
    )
    return pattern.repeat((count + pattern.numel() - 1) //
                          pattern.numel())[:count].view(
                              torch.float16).reshape(shape)


def assert_bitwise_equal(name, actual, expected):
    actual_bytes = actual.reshape(-1).view(torch.uint8)
    expected_bytes = expected.reshape(-1).view(torch.uint8)
    if not torch.equal(actual_bytes, expected_bytes):
        mismatch = torch.count_nonzero(
            actual_bytes != expected_bytes).item()
        raise AssertionError(
            f"{name}: byte mismatch={mismatch}, "
            f"actual_shape={tuple(actual.shape)}, "
            f"dtype={actual.dtype}"
        )


def run_case(name, input_cpu, dims):
    expected = input_cpu.permute(dims).contiguous()
    actual = transpose_validation_lib.transpose(
        input_cpu.npu(),
        dims,
    ).cpu()
    assert_bitwise_equal(name, actual, expected)


def cyclic_dims(rank, split, negative):
    dims = list(range(split, rank)) + list(range(split))
    if negative:
        dims = [axis - rank for axis in dims]
    return tuple(dims)


def main():
    torch.manual_seed(SEED)
    cases = []

    matrix_shapes = (
        (1, 1),
        (7, 9),
        (8, 8),
        (9, 17),
        (15, 16),
        (16, 17),
        (31, 33),
        (32, 65),
        (127, 257),
    )
    for shape in matrix_shapes:
        cases.append((shape, (1, 0), "matrix"))

    rotation_shapes = (
        (2, 3, 5),
        (7, 9, 17),
        (1, 17, 1),
        (16, 2, 16),
        (3, 5, 7, 9),
        (1, 2, 17, 33),
        (2, 3, 5, 7, 11),
        (2, 3, 4, 5, 7, 9),
    )
    for shape in rotation_shapes:
        for split in range(1, len(shape)):
            cases.append(
                (
                    shape,
                    cyclic_dims(
                        len(shape),
                        split,
                        (split + len(shape)) % 2 == 0,
                    ),
                    f"rotation_s{split}",
                )
            )

    fallback_cases = (
        ((2, 3, 5), (0, 2, 1)),
        ((7, 9, 11), (1, 0, 2)),
        ((2, 3, 5, 7), (2, 0, 3, 1)),
        ((2, 3, 4, 5, 7), (4, 1, 3, 0, 2)),
        ((2, 3, 2, 3, 2, 5), (5, 2, 0, 4, 1, 3)),
    )
    for shape, dims in fallback_cases:
        cases.append((shape, dims, "fallback"))

    total = 0
    for dtype in DTYPES:
        for case_id, (shape, dims, kind) in enumerate(cases):
            run_case(
                f"{dtype}_{kind}_{case_id}",
                make_tensor(shape, dtype),
                dims,
            )
            total += 1

    for dtype, shape, dims in (
        (torch.float32, (9, 17), (1, 0)),
        (torch.float32, (3, 5, 7), (-1, -3, -2)),
        (torch.float16, (17, 33), (1, 0)),
        (torch.float16, (2, 3, 5, 7), (2, 3, 0, 1)),
    ):
        run_case(
            f"{dtype}_special_bits_{shape}_{dims}",
            bit_pattern_tensor(shape, dtype),
            dims,
        )
        total += 1

    torch.npu.synchronize()
    print(
        f"ALL EXTENDED PASS: {total} cases, seed={SEED}"
    )


if __name__ == "__main__":
    main()
