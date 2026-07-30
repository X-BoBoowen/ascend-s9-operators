import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026073017


CASES = (
    ("fp16_outer3_batch17", (3, 5, 17, 8, 16), (0, 2, 4), False, torch.float16),
    ("fp16_batch31", (2, 7, 31, 16, 32), (0, 2, 4), False, torch.float16),
    ("fp16_batch32_split", (2, 3, 32, 8, 16), (0, 2, 4), False, torch.float16),
    ("fp16_batch33_split", (2, 3, 33, 24, 16), (0, 2, 4), False, torch.float16),
    ("fp16_41_vector_tasks", (2, 41, 3, 8, 16), (0, 2, 4), False, torch.float16),
    ("fp16_tail64", (3, 4, 9, 16, 64), (0, 2, 4), False, torch.float16),
    ("fp16_tail1024_boundary", (2, 3, 3, 8, 1024), (0, 2, 4), False, torch.float16),
    ("fp16_keepdim", (3, 5, 17, 8, 16), (0, 2, 4), True, torch.float16),
    ("fp16_negative_axes", (3, 5, 17, 8, 16), (-5, -3, -1), False, torch.float16),
    ("fp16_outer1_suffix2", (17, 3, 8, 4, 16), (0, 3, 4), False, torch.float16),
    ("fp16_shifted_suffix2", (3, 17, 8, 4, 16), (1, 3, 4), False, torch.float16),
    ("bf16_outer3_batch17", (3, 5, 17, 8, 16), (0, 2, 4), False, torch.bfloat16),
    ("bf16_batch32_split", (2, 3, 32, 16, 32), (0, 2, 4), False, torch.bfloat16),
    ("fp32_tail8", (3, 5, 17, 8, 8), (0, 2, 4), False, torch.float32),
    ("fp32_tail32", (2, 7, 31, 16, 32), (0, 2, 4), False, torch.float32),
    ("fp32_keepdim", (3, 5, 17, 8, 16), (0, 2, 4), True, torch.float32),
)


def tolerance(dtype):
    if dtype == torch.float16:
        return 3e-3, 3e-3
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    return 1e-4, 1e-4


def run_case(index, name, shape, axes, keep_dims, dtype):
    generator = torch.Generator().manual_seed(SEED + index)
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 2.0
        - 1.0
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
    rtol, atol = tolerance(dtype)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    max_abs = torch.max(
        torch.abs(actual.float() - expected.float())
    ).item()
    print(
        f"PASS {name}: shape={shape}, axes={axes}, "
        f"keep_dims={keep_dims}, dtype={dtype}, max_abs={max_abs:.8g}"
    )


def main():
    for index, case in enumerate(CASES):
        run_case(index, *case)
    print(f"SUMMARY: {len(CASES)}/{len(CASES)} passed")


if __name__ == "__main__":
    main()
