import random

import torch
import torch_npu

import transpose_validation_lib


SEED = 2026072607
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
        return torch.randint(-100_000, 100_000, shape, dtype=dtype)
    return torch.randint(-128, 128, shape, dtype=dtype)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    dimensions = (1, 2, 3, 4, 5, 7, 11, 17, 33)
    total = 0
    for case_id in range(200):
        dtype = DTYPES[case_id % len(DTYPES)]
        rank = random.randint(1, 6)
        shape = tuple(random.choice(dimensions) for _ in range(rank))
        permutation = list(range(rank))
        random.shuffle(permutation)
        dims = tuple(
            axis if random.random() < 0.5 else axis - rank
            for axis in permutation
        )
        input_cpu = make_tensor(shape, dtype)
        expected = input_cpu.permute(dims).contiguous()
        actual = transpose_validation_lib.transpose(
            input_cpu.npu(),
            dims,
        ).cpu()
        if not torch.equal(actual, expected):
            mismatch = torch.count_nonzero(actual != expected).item()
            raise AssertionError(
                f"case={case_id}, dtype={dtype}, shape={shape}, "
                f"dims={dims}, mismatch={mismatch}"
            )
        total += 1
    torch.npu.synchronize()
    print(f"ALL RANDOM PASS: {total} cases, seed={SEED}")


if __name__ == "__main__":
    main()
