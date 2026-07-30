import statistics
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026072801


CASES = {
    "public": ((123, 31), torch.float16, (-1,), True),
    "last_fp16": ((1024, 1024), torch.float16, (-1,), True),
    "last_fp32": ((256, 4097), torch.float32, (-1,), False),
    "last_bf16": ((512, 257), torch.bfloat16, (-1,), False),
    "middle_fp16": ((64, 257, 65), torch.float16, (1,), False),
    "middle_fp32": ((17, 131, 19), torch.float32, (1,), True),
    "multi_fp16": ((8, 33, 17, 65), torch.float16, (1, 3), False),
    "multi_bf16": ((8, 33, 17, 65), torch.bfloat16, (0, 2), True),
    "multi3_fp16": (
        (8, 17, 9, 33, 65),
        torch.float16,
        (0, 2, 4),
        False,
    ),
    "multi3_bf16": (
        (8, 17, 9, 33, 65),
        torch.bfloat16,
        (0, 2, 4),
        True,
    ),
    "multi3_fp32": (
        (4, 17, 7, 33, 31),
        torch.float32,
        (0, 2, 4),
        False,
    ),
    "strided_tree_fp16": (
        (32, 17, 64, 65),
        torch.float16,
        (0, 2),
        False,
    ),
    "strided_tree_bf16": (
        (32, 17, 64, 65),
        torch.bfloat16,
        (0, 2),
        False,
    ),
    "strided_tree_fp32": (
        (32, 17, 64, 65),
        torch.float32,
        (0, 2),
        False,
    ),
    "suffix_fp16": ((64, 33, 65), torch.float16, (1, 2), False),
    "prefix_fp16": ((17, 33, 65), torch.float16, (0, 1), False),
    "axis0_fp32": ((257, 64), torch.float32, (0,), False),
    "all_fp32": ((41, 17, 19), torch.float32, (0, 1, 2), False),
    "axis0_large_fp16": ((10000, 64), torch.float16, (0,), False),
    "axis0_large_bf16": ((10000, 64), torch.bfloat16, (0,), False),
    "axis0_large_fp32": ((10000, 64), torch.float32, (0,), False),
    "prefix_large_fp16": (
        (200, 1000, 64),
        torch.float16,
        (0, 1),
        False,
    ),
    "prefix_large_bf16": (
        (200, 1000, 64),
        torch.bfloat16,
        (0, 1),
        False,
    ),
    "prefix_large_fp32": (
        (200, 1000, 64),
        torch.float32,
        (0, 1),
        False,
    ),
    "all_large_fp16": (
        (200, 1000, 64),
        torch.float16,
        (0, 1, 2),
        False,
    ),
    "all_large_bf16": (
        (200, 1000, 64),
        torch.bfloat16,
        (0, 1, 2),
        False,
    ),
    "all_large_fp32": (
        (200, 1000, 64),
        torch.float32,
        (0, 1, 2),
        False,
    ),
    "last_workspace_o8_fp16": (
        (8, 4, 8192),
        torch.float16,
        (1, 2),
        False,
    ),
    "last_workspace_o8_bf16": (
        (8, 4, 8192),
        torch.bfloat16,
        (1, 2),
        False,
    ),
    "last_10000_fp16": ((64, 10000), torch.float16, (-1,), False),
    "last_10000_bf16": ((64, 10000), torch.bfloat16, (-1,), False),
    "last_10000_fp32": ((64, 10000), torch.float32, (-1,), False),
    "small_fp16": ((7,), torch.float16, (0,), False),
}


def tolerances(dtype):
    if dtype == torch.float16:
        return 3e-3, 3e-3
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    return 1e-4, 1e-4


def build_case(name):
    shape, dtype, axis, keep_dims = CASES[name]
    generator = torch.Generator().manual_seed(SEED)
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 4.0
        - 2.0
    ).to(dtype)
    expected = torch.sum(
        torch.square(input_cpu),
        dim=axis,
        keepdim=keep_dims,
    )
    return input_cpu.npu(), expected, axis, keep_dims


def run_once(input_npu, expected, axis, keep_dims):
    return square_sum_v1_validation_lib.square_sum_v1(
        input_npu,
        axis,
        keep_dims,
        list(expected.shape),
    )


def measure(name, repeats=100, trials=5):
    input_npu, expected, axis, keep_dims = build_case(name)
    result = None
    for _ in range(20):
        result = run_once(input_npu, expected, axis, keep_dims)
    torch.npu.synchronize()

    values = []
    for _ in range(trials):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            result = run_once(input_npu, expected, axis, keep_dims)
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / repeats)

    rtol, atol = tolerances(expected.dtype)
    torch.testing.assert_close(
        result.cpu(),
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    print(
        f"{name}: median_us={statistics.median(values):.3f}, "
        f"trials_us={[round(value, 3) for value in values]}"
    )


def profile(name, repeats):
    input_npu, expected, axis, keep_dims = build_case(name)
    result = None
    for _ in range(10):
        result = run_once(input_npu, expected, axis, keep_dims)
    torch.npu.synchronize()
    for _ in range(repeats):
        result = run_once(input_npu, expected, axis, keep_dims)
    torch.npu.synchronize()
    rtol, atol = tolerances(expected.dtype)
    torch.testing.assert_close(
        result.cpu(),
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    print(f"PROFILE_OK name={name} repeats={repeats}")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--profile":
        profile(sys.argv[2], int(sys.argv[3]) if len(sys.argv) >= 4 else 30)
        return
    names = sys.argv[1:] or list(CASES)
    for name in names:
        measure(name)


if __name__ == "__main__":
    main()
