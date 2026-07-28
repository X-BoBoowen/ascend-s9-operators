import random

import torch
import torch_npu

import concat_validation_lib


torch.npu.config.allow_internal_format = False
torch.manual_seed(20260726)
random.seed(20260726)


def make_tensor(shape, dtype):
    if dtype in (torch.float16, torch.float32):
        return torch.randn(shape, dtype=dtype)
    if dtype == torch.int32:
        return torch.randint(-100000, 100000, shape, dtype=dtype)
    if dtype == torch.int8:
        return torch.randint(-128, 128, shape, dtype=dtype)
    raise ValueError(dtype)


def run_case(name, shapes, dim, dtype):
    cpu_inputs = [make_tensor(shape, dtype) for shape in shapes]
    npu_inputs = [tensor.npu() for tensor in cpu_inputs]
    expected = torch.cat(cpu_inputs, dim=dim)
    actual = concat_validation_lib.concat(npu_inputs, dim).cpu()
    if not torch.equal(actual, expected):
        mismatch = torch.count_nonzero(actual != expected).item()
        raise AssertionError(
            f"{name}: mismatch={mismatch}, "
            f"actual_shape={tuple(actual.shape)}, "
            f"expected_shape={tuple(expected.shape)}"
        )
    print(
        f"PASS {name}: dtype={dtype}, dim={dim}, "
        f"inputs={len(shapes)}, output={tuple(expected.shape)}"
    )


def main():
    public_sizes = [27, 40, 63, 24, 50, 26, 19, 2, 5]
    cases = [
        (
            "public_case1",
            [(128, size) for size in public_sizes],
            -1,
            torch.float16,
        ),
        (
            "float32_dim0",
            [(3, 5, 7), (2, 5, 7), (4, 5, 7)],
            0,
            torch.float32,
        ),
        (
            "int32_middle",
            [(2, 3, 5), (2, 7, 5), (2, 1, 5)],
            1,
            torch.int32,
        ),
        (
            "int8_negative_last_unaligned",
            [(11, 13), (11, 2), (11, 19)],
            -1,
            torch.int8,
        ),
        (
            "thirty_three_inputs",
            [(2, 1, 3)] * 33,
            1,
            torch.float16,
        ),
        (
            "wide_rows_chunked",
            [(2, 20000), (2, 19001)],
            1,
            torch.float16,
        ),
        (
            "zero_length_first_and_middle",
            [(4, 0, 3), (4, 7, 3), (4, 0, 3), (4, 5, 3)],
            1,
            torch.float32,
        ),
        (
            "rank4_dim2",
            [(2, 3, 4, 5), (2, 3, 7, 5)],
            2,
            torch.int8,
        ),
        (
            "single_input_rank1",
            [(17,)],
            0,
            torch.int32,
        ),
    ]
    for case in cases:
        run_case(*case)
    torch.npu.synchronize()
    print(f"ALL PASS: {len(cases)} cases")


if __name__ == "__main__":
    main()
