import statistics
import sys

import torch
import torch_npu

import transpose_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026072901


CASES = {
    "public_fp16": ((128, 256), (1, 0), torch.float16),
    "matrix_fp16": ((1024, 1024), (1, 0), torch.float16),
    "matrix_tail_fp16": ((1000, 1000), (1, 0), torch.float16),
    "matrix_fp32": ((1024, 1024), (1, 0), torch.float32),
    "matrix_int32": ((1024, 1024), (1, 0), torch.int32),
    "matrix_int8": ((1024, 1024), (1, 0), torch.int8),
    "last2_fp32": ((32, 256, 512), (0, 2, 1), torch.float32),
    "last2_int8": ((32, 256, 512), (0, 2, 1), torch.int8),
    "nchw_nhwc_fp16": ((8, 64, 32, 32), (0, 2, 3, 1), torch.float16),
    "nchw_nhwc_fp32": ((8, 64, 32, 32), (0, 2, 3, 1), torch.float32),
    "nchw_nhwc_int8": ((8, 64, 32, 32), (0, 2, 3, 1), torch.int8),
    "gather_a_fp16": ((32, 64, 128), (1, 2, 0), torch.float16),
    "gather_a_fp32": ((32, 64, 128), (1, 2, 0), torch.float32),
    "gather_a_int8": ((32, 64, 128), (1, 2, 0), torch.int8),
    "gather_b_fp16": ((32, 64, 128), (2, 0, 1), torch.float16),
    "gather_b_fp32": ((32, 64, 128), (2, 0, 1), torch.float32),
    "gather_b_int8": ((32, 64, 128), (2, 0, 1), torch.int8),
    "identity_fp32": ((2048, 2048), (0, 1), torch.float32),
}


def make_tensor(shape, dtype):
    generator = torch.Generator().manual_seed(SEED)
    if dtype in (torch.float16, torch.float32):
        return torch.randn(shape, dtype=dtype, generator=generator)
    if dtype == torch.int32:
        return torch.randint(
            -(2**31),
            2**31 - 1,
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


def build_case(name):
    shape, dims, dtype = CASES[name]
    input_cpu = make_tensor(shape, dtype)
    expected = input_cpu.permute(dims).contiguous()
    return input_cpu.npu(), expected, dims


def run_once(input_npu, dims):
    return transpose_validation_lib.transpose(input_npu, dims)


def run_reference(input_npu, dims):
    return input_npu.permute(dims).contiguous()


def check(result, expected, name):
    actual = result.cpu()
    if not torch.equal(actual, expected):
        mismatch = torch.count_nonzero(actual != expected).item()
        raise AssertionError(f"{name}: mismatch={mismatch}")


def measure(name, repeats=50, trials=5):
    input_npu, expected, dims = build_case(name)
    result = None
    for _ in range(10):
        result = run_once(input_npu, dims)
    torch.npu.synchronize()

    values = []
    for _ in range(trials):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            result = run_once(input_npu, dims)
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / repeats)

    check(result, expected, name)
    print(
        f"{name}: median_us={statistics.median(values):.3f}, "
        f"trials_us={[round(value, 3) for value in values]}"
    )


def measure_reference(name, repeats=50, trials=5):
    input_npu, expected, dims = build_case(name)
    result = None
    for _ in range(10):
        result = run_reference(input_npu, dims)
    torch.npu.synchronize()

    values = []
    for _ in range(trials):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            result = run_reference(input_npu, dims)
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / repeats)

    check(result, expected, name)
    print(
        f"reference_{name}: "
        f"median_us={statistics.median(values):.3f}, "
        f"trials_us={[round(value, 3) for value in values]}"
    )


def profile(name, repeats):
    input_npu, expected, dims = build_case(name)
    result = None
    for _ in range(10):
        result = run_once(input_npu, dims)
    torch.npu.synchronize()
    for _ in range(repeats):
        result = run_once(input_npu, dims)
    torch.npu.synchronize()
    check(result, expected, name)
    print(f"PROFILE_OK name={name} repeats={repeats}")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--profile":
        profile(sys.argv[2], int(sys.argv[3]) if len(sys.argv) >= 4 else 20)
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "--reference":
        names = sys.argv[2:] or list(CASES)
        for name in names:
            measure_reference(name)
        return
    names = sys.argv[1:] or list(CASES)
    for name in names:
        measure(name)


if __name__ == "__main__":
    main()
