import statistics

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026073015


def measure(last_reduce_dim, repeats=100, trials=5):
    shape = (32, 17, last_reduce_dim, 65)
    generator = torch.Generator().manual_seed(SEED)
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 4.0
        - 2.0
    ).to(torch.float16)
    expected = torch.sum(
        torch.square(input_cpu),
        dim=(0, 2),
        keepdim=False,
    )
    input_npu = input_cpu.npu()

    result = None
    for _ in range(20):
        result = square_sum_v1_validation_lib.square_sum_v1(
            input_npu,
            (0, 2),
            False,
            list(expected.shape),
        )
    torch.npu.synchronize()

    values = []
    for _ in range(trials):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            result = square_sum_v1_validation_lib.square_sum_v1(
                input_npu,
                (0, 2),
                False,
                list(expected.shape),
            )
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / repeats)

    torch.testing.assert_close(
        result.cpu(),
        expected,
        rtol=3e-3,
        atol=3e-3,
        equal_nan=True,
    )
    print(
        f"strided_fp16_lastdim{last_reduce_dim}: "
        f"median_us={statistics.median(values):.3f}, "
        f"trials_us={[round(value, 3) for value in values]}"
    )


def main():
    for last_reduce_dim in (4, 8, 16, 24, 32, 48, 64, 96, 128):
        measure(last_reduce_dim)


if __name__ == "__main__":
    main()
