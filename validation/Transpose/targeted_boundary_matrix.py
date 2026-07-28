import torch
import torch_npu

import transpose_validation_lib


torch.npu.config.allow_internal_format = False
torch.manual_seed(2026072817)

DTYPES = (
    torch.float16,
    torch.float32,
    torch.int32,
    torch.int8,
)


def make_tensor(shape, dtype):
    if dtype in (torch.float16, torch.float32):
        return torch.randn(shape, dtype=dtype)
    if dtype == torch.int32:
        return torch.randint(-(2**31), 2**31 - 1, shape, dtype=dtype)
    return torch.randint(-128, 128, shape, dtype=dtype)


def assert_bitwise_equal(name, actual, expected):
    actual_bytes = actual.reshape(-1).view(torch.uint8)
    expected_bytes = expected.reshape(-1).view(torch.uint8)
    if not torch.equal(actual_bytes, expected_bytes):
        mismatch = torch.count_nonzero(
            actual_bytes != expected_bytes
        ).item()
        raise AssertionError(f"{name}: byte mismatch={mismatch}")


def run_case(name, shape, dims, dtype):
    input_cpu = make_tensor(shape, dtype)
    expected = input_cpu.permute(dims).contiguous()
    actual = transpose_validation_lib.transpose(
        input_cpu.npu(),
        dims,
    ).cpu()
    assert_bitwise_equal(name, actual, expected)


def main():
    matrix_shapes = (
        (63, 65),
        (64, 64),
        (65, 65),
        (95, 97),
        (96, 128),
        (97, 129),
        (127, 191),
        (128, 192),
        (129, 193),
        (63, 767),
        (64, 768),
        (65, 769),
    )
    rotations = (
        ((3, 64, 128), (1, 2, 0)),
        ((3, 64, 128), (2, 0, 1)),
        ((2, 3, 64, 128), (2, 3, 0, 1)),
        ((2, 3, 64, 128), (3, 0, 1, 2)),
        ((4, 17, 31, 33), (1, 2, 3, 0)),
        ((4, 17, 31, 33), (2, 3, 0, 1)),
        ((4, 17, 31, 33), (3, 0, 1, 2)),
        ((2, 3, 5, 7, 11), (2, 3, 4, 0, 1)),
        ((2, 3, 5, 7, 11), (4, 0, 1, 2, 3)),
    )

    total = 0
    for dtype in DTYPES:
        for shape in matrix_shapes:
            run_case(
                f"{dtype}_matrix_{shape}",
                shape,
                (1, 0),
                dtype,
            )
            total += 1
        for shape, dims in rotations:
            run_case(
                f"{dtype}_rotation_{shape}_{dims}",
                shape,
                dims,
                dtype,
            )
            total += 1

    torch.npu.synchronize()
    print(f"ALL TARGETED BOUNDARY PASS: {total} cases")


if __name__ == "__main__":
    main()
