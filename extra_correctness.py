import sys

import torch
import torch_npu

import custom_ops_lib


torch.npu.config.allow_internal_format = False


def assert_same(actual, expected):
    actual = actual.cpu()
    if expected.dtype.is_floating_point:
        if expected.dtype == torch.float32:
            torch.testing.assert_close(
                actual, expected, rtol=1e-4, atol=1e-4, equal_nan=True
            )
        else:
            torch.testing.assert_close(
                actual, expected, rtol=1e-2, atol=1e-2, equal_nan=True
            )
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def run_greater():
    cases = [
        (
            "fp16_broadcast_special",
            torch.tensor(
                [[[float("nan")], [float("inf")], [-float("inf")]]],
                dtype=torch.float16,
            ),
            torch.tensor(
                [[[0.0, float("nan")], [float("inf"), 1.0], [-1.0, -float("inf")]]],
                dtype=torch.float16,
            ),
        ),
        (
            "fp32_broadcast",
            torch.linspace(-2, 2, 24, dtype=torch.float32).reshape(2, 3, 4),
            torch.tensor([[[0.0], [1.0], [-1.0]]], dtype=torch.float32),
        ),
        (
            "int32_equal_boundaries",
            torch.tensor([-2, -1, 0, 1, 2], dtype=torch.int32),
            torch.tensor([-3, -1, 1, 0, 3], dtype=torch.int32),
        ),
    ]
    for name, left, right in cases:
        expected = torch.gt(left, right)
        actual = custom_ops_lib.custom_op(left.npu(), right.npu())
        assert_same(actual, expected)
        print(f"EXTRA_CASE_PASS task=Greater name={name}")


def run_index_add():
    cases = [
        (
            "int8_duplicate_dim0",
            torch.arange(20, dtype=torch.int8).reshape(5, 4),
            torch.tensor([1, 1, 3, 1], dtype=torch.int32),
            torch.tensor(
                [[2, -1, 3, 0], [1, 1, -2, 4], [-3, 2, 0, 1], [5, -4, 1, 2]],
                dtype=torch.int8,
            ),
            0,
        ),
        (
            "fp32_duplicate_dim1",
            torch.linspace(-1, 1, 15, dtype=torch.float32).reshape(3, 5),
            torch.tensor([0, 2, 2, 4], dtype=torch.int32),
            torch.linspace(-2, 2, 12, dtype=torch.float32).reshape(3, 4),
            1,
        ),
    ]
    for name, input_tensor, index, source, dim in cases:
        expected = torch.index_add(input_tensor, dim, index, source)
        actual = custom_ops_lib.custom_op(
            input_tensor.npu(), index.npu(), source.npu(), dim
        )
        assert_same(actual, expected)
        print(f"EXTRA_CASE_PASS task=IndexAdd name={name}")


def run_concat():
    cases = [
        (
            "fp16_negative_dim_with_empty",
            [
                torch.arange(6, dtype=torch.float16).reshape(2, 3),
                torch.empty((2, 0), dtype=torch.float16),
                torch.arange(4, dtype=torch.float16).reshape(2, 2),
            ],
            -1,
        ),
        (
            "fp32_dim0",
            [
                torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32),
            ],
            0,
        ),
    ]
    for name, inputs, dim in cases:
        expected = torch.cat(inputs, dim=dim)
        actual = custom_ops_lib.custom_op(
            [tensor.npu() for tensor in inputs], dim, list(expected.shape)
        )
        assert_same(actual, expected)
        print(f"EXTRA_CASE_PASS task=Concat name={name}")


def run_transpose():
    cases = [
        (
            "fp16_3d_permutation",
            torch.arange(24, dtype=torch.float16).reshape(2, 3, 4),
            (2, 0, 1),
        ),
        (
            "fp32_identity",
            torch.linspace(-1, 1, 20, dtype=torch.float32).reshape(4, 5),
            (0, 1),
        ),
    ]
    for name, input_tensor, dims in cases:
        expected = torch.permute(input_tensor, dims)
        actual = custom_ops_lib.custom_op(
            input_tensor.npu(), dims, list(expected.shape)
        )
        assert_same(actual, expected)
        print(f"EXTRA_CASE_PASS task=Transpose name={name}")


def run_square_sum():
    cases = [
        (
            "fp16_negative_axis_keepdim",
            torch.linspace(-3, 3, 35, dtype=torch.float16).reshape(5, 7),
            (-1,),
            True,
        ),
        (
            "fp32_multi_axis",
            torch.linspace(-2, 2, 48, dtype=torch.float32).reshape(2, 3, 8),
            (0, 2),
            False,
        ),
    ]
    for name, input_tensor, axis, keep_dims in cases:
        expected = torch.sum(torch.square(input_tensor), axis, keepdim=keep_dims)
        actual = custom_ops_lib.custom_op(
            input_tensor.npu(), axis, keep_dims, list(expected.shape)
        )
        assert_same(actual, expected)
        print(f"EXTRA_CASE_PASS task=SquareSumV1 name={name}")


RUNNERS = {
    "Concat": run_concat,
    "Greater": run_greater,
    "IndexAdd": run_index_add,
    "SquareSumV1": run_square_sum,
    "Transpose": run_transpose,
}


if __name__ == "__main__":
    task = sys.argv[1]
    RUNNERS[task]()
    print(f"EXTRA_CORRECTNESS_PASS task={task}")
