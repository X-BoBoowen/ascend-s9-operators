import random

import torch
import torch_npu

import greater_validation_lib


SEED = 2026072603
DTYPES = (
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.int32,
    torch.int8,
)


def make_tensor(shape, dtype):
    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        return torch.randn(shape, dtype=dtype)
    if dtype == torch.int32:
        return torch.randint(-(2**31), 2**31 - 1, shape, dtype=dtype)
    return torch.randint(-128, 128, shape, dtype=dtype)


def operand_shape(output_shape):
    if not output_shape:
        return ()
    drop = random.randint(0, len(output_shape))
    retained = output_shape[drop:]
    return tuple(
        dimension if random.random() < 0.55 else 1
        for dimension in retained
    )


def check(case_id, dtype, self_shape, other_shape):
    self_cpu = make_tensor(self_shape, dtype)
    other_cpu = make_tensor(other_shape, dtype)
    expected = torch.gt(self_cpu, other_cpu)
    actual = greater_validation_lib.greater(
        self_cpu.npu(),
        other_cpu.npu(),
    ).cpu()
    if not torch.equal(actual, expected):
        mismatch = torch.count_nonzero(actual != expected).item()
        raise AssertionError(
            f"case={case_id}, dtype={dtype}, self={self_shape}, "
            f"other={other_shape}, output={tuple(expected.shape)}, "
            f"mismatch={mismatch}"
        )


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    total = 0

    directed = [
        ((200_003,), (200_003,)),
        ((41, 127, 43), (1, 127, 1)),
        ((2, 3, 5, 7, 11, 13), (3, 1, 7, 1, 13)),
        ((), ()),
    ]
    for dtype in DTYPES:
        for self_shape, other_shape in directed:
            check(total, dtype, self_shape, other_shape)
            total += 1

    dimensions = (1, 2, 3, 5, 7, 11, 17)
    for case_id in range(200):
        dtype = DTYPES[case_id % len(DTYPES)]
        rank = random.randint(0, 6)
        output_shape = tuple(
            random.choice(dimensions) for _ in range(rank)
        )
        self_shape = operand_shape(output_shape)
        other_shape = operand_shape(output_shape)
        check(total, dtype, self_shape, other_shape)
        total += 1

    torch.npu.synchronize()
    print(f"ALL RANDOM PASS: {total} cases, seed={SEED}")


if __name__ == "__main__":
    main()
