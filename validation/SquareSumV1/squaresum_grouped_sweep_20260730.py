import statistics

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026073016


CASES = (
    ("small_aligned_fp16", (8, 8, 16, 64, 16), torch.float16),
    ("medium_aligned_fp16", (16, 8, 32, 64, 32), torch.float16),
    ("large_aligned_fp16", (16, 8, 32, 64, 64), torch.float16),
    ("wide_output_fp16", (16, 17, 32, 64, 32), torch.float16),
    ("medium_aligned_bf16", (16, 8, 32, 64, 32), torch.bfloat16),
    ("medium_aligned_fp32", (16, 8, 32, 64, 32), torch.float32),
    ("unaligned_tail_fp16", (16, 8, 32, 64, 33), torch.float16),
)


def measure(name, shape, dtype, repeats=30, trials=5):
    generator = torch.Generator().manual_seed(SEED)
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 2.0
        - 1.0
    ).to(dtype)
    expected = torch.sum(
        torch.square(input_cpu),
        dim=(0, 2, 4),
        keepdim=False,
    )
    input_npu = input_cpu.npu()

    result = None
    for _ in range(20):
        result = square_sum_v1_validation_lib.square_sum_v1(
            input_npu,
            (0, 2, 4),
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
                (0, 2, 4),
                False,
                list(expected.shape),
            )
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / repeats)

    if dtype == torch.float16:
        rtol = atol = 3e-3
    elif dtype == torch.bfloat16:
        rtol = atol = 3e-2
    else:
        rtol = atol = 1e-4
    torch.testing.assert_close(
        result.cpu(),
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    print(
        f"{name}: shape={shape}, median_us={statistics.median(values):.3f}, "
        f"trials_us={[round(value, 3) for value in values]}"
    )


def main():
    for name, shape, dtype in CASES:
        measure(name, shape, dtype)


if __name__ == "__main__":
    main()
