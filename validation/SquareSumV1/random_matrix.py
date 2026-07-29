import random

import torch
import torch_npu

import square_sum_v1_validation_lib


SEED = 2026072609
DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


def tolerances(dtype):
    if dtype == torch.float16:
        return 3e-3, 3e-3
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    return 3e-5, 3e-5


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    dimensions = (1, 2, 3, 5, 7, 11, 17, 33)
    total = 0
    for case_id in range(150):
        dtype = DTYPES[case_id % len(DTYPES)]
        rank = random.randint(1, 5)
        shape = tuple(
            random.choice(dimensions) for _ in range(rank)
        )
        axis_count = random.randint(1, rank)
        normalized = random.sample(
            list(range(rank)),
            axis_count,
        )
        axes = tuple(
            axis if random.random() < 0.5 else axis - rank
            for axis in normalized
        )
        keep_dims = bool(random.getrandbits(1))
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
        )
        total += 1
    torch.npu.synchronize()
    print(f"ALL RANDOM PASS: {total} cases, seed={SEED}")


if __name__ == "__main__":
    main()
