import random

import torch
import torch_npu

import index_add_validation_lib


SEED = 2026072605
DTYPES = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int8,
)


def make_tensor(shape, dtype, scale=1.0):
    if dtype in (torch.float32, torch.bfloat16, torch.float16):
        return torch.randn(shape, dtype=dtype) * scale
    if dtype == torch.int32:
        return torch.randint(-1_000_000, 1_000_000, shape, dtype=dtype)
    return torch.randint(-128, 128, shape, dtype=dtype)


def assert_result(name, actual, expected):
    if expected.dtype in (torch.float32, torch.bfloat16, torch.float16):
        tolerance = 1e-4 if expected.dtype == torch.float32 else 1e-3
        torch.testing.assert_close(
            actual,
            expected,
            rtol=tolerance,
            atol=tolerance,
            msg=lambda message: f"{name}: {message}",
        )
    elif not torch.equal(actual, expected):
        mismatch = torch.count_nonzero(actual != expected).item()
        raise AssertionError(f"{name}: mismatch={mismatch}")


def check(case_id, dtype, shape, dim, index_count, repeats=1):
    normalized_dim = dim if dim >= 0 else dim + len(shape)
    index = torch.randint(
        0,
        shape[normalized_dim],
        (index_count,),
        dtype=torch.int32,
    )
    source_shape = list(shape)
    source_shape[normalized_dim] = index_count
    source_scale = 0.01 if index_count > 256 else 1.0
    self_cpu = make_tensor(shape, dtype)
    source_cpu = make_tensor(tuple(source_shape), dtype, source_scale)
    expected = torch.index_add(
        self_cpu,
        dim,
        index,
        source_cpu,
    )
    self_npu = self_cpu.npu()
    index_npu = index.npu()
    source_npu = source_cpu.npu()
    for repeat in range(repeats):
        actual = index_add_validation_lib.index_add(
            self_npu,
            dim,
            index_npu,
            source_npu,
        ).cpu()
        assert_result(f"case{case_id}/repeat{repeat}", actual, expected)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    total = 0

    large_cases = (
        ((4, 4097), 0, 101),
        ((3, 13, 17), 1, 8000),
        ((2, 5, 9001), 1, 31),
        ((2, 17), -1, 8000),
    )
    for dtype in DTYPES:
        for shape, dim, index_count in large_cases:
            check(total, dtype, shape, dim, index_count, repeats=2)
            total += 1

    dimensions = (1, 2, 3, 5, 7, 11, 17, 33)
    for case_id in range(150):
        dtype = DTYPES[case_id % len(DTYPES)]
        rank = random.randint(1, 6)
        shape = tuple(random.choice(dimensions) for _ in range(rank))
        normalized_dim = random.randrange(rank)
        dim = (
            normalized_dim
            if random.random() < 0.5
            else normalized_dim - rank
        )
        index_count = random.choice((0, 1, 2, 3, 7, 17, 41))
        check(total, dtype, shape, dim, index_count, repeats=2)
        total += 1

    torch.npu.synchronize()
    print(f"ALL RANDOM PASS: {total} cases, seed={SEED}")


if __name__ == "__main__":
    main()
