import random

import torch
import torch_npu

import concat_validation_lib


SEED = 2026072601
DTYPES = (torch.float16, torch.float32, torch.int32, torch.int8)


def make_tensor(shape, dtype, generator):
    if dtype in (torch.float16, torch.float32):
        return torch.randn(shape, dtype=dtype, generator=generator)
    if dtype == torch.int32:
        return torch.randint(
            -100000,
            100000,
            shape,
            dtype=dtype,
            generator=generator,
        )
    return torch.randint(
        -128,
        128,
        shape,
        dtype=dtype,
        generator=generator,
    )


def main():
    random.seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    for case_index in range(100):
        rank = random.randint(1, 5)
        dim = random.randrange(rank)
        dtype = random.choice(DTYPES)
        input_count = random.choice((1, 2, 3, 4, 7, 17, 33))
        base_shape = [random.randint(0, 8) for _ in range(rank)]
        concat_sizes = [random.randint(0, 15) for _ in range(input_count)]
        shapes = []
        for concat_size in concat_sizes:
            shape = list(base_shape)
            shape[dim] = concat_size
            shapes.append(tuple(shape))

        cpu_inputs = [
            make_tensor(shape, dtype, generator)
            for shape in shapes
        ]
        expected = torch.cat(cpu_inputs, dim=dim)
        actual = concat_validation_lib.concat(
            [value.npu() for value in cpu_inputs],
            dim if case_index % 2 == 0 else dim - rank,
        ).cpu()
        if not torch.equal(actual, expected):
            mismatch = torch.count_nonzero(actual != expected).item()
            raise AssertionError(
                f"case={case_index}, dtype={dtype}, dim={dim}, "
                f"shapes={shapes}, mismatch={mismatch}"
            )
        print(
            f"PASS random_{case_index:03d}: dtype={dtype}, "
            f"dim={dim}, inputs={input_count}, output={tuple(expected.shape)}"
        )
    torch.npu.synchronize()
    print("ALL PASS: 100 randomized cases")


if __name__ == "__main__":
    main()
