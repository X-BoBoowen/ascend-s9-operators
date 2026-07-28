import sys

import torch
import torch_npu

import transpose_validation_lib


torch.npu.config.allow_internal_format = False


CASES = {
    "public": ((128, 256), (1, 0), torch.float16),
    "identity": ((128, 256), (0, 1), torch.float16),
    "fp32_2d": ((128, 256), (1, 0), torch.float32),
    "int8_2d": ((128, 256), (1, 0), torch.int8),
    "tail": ((127, 257), (1, 0), torch.float16),
    "general_a": ((32, 64, 128), (1, 2, 0), torch.float16),
    "general_b": ((32, 64, 128), (2, 0, 1), torch.float16),
    "rank4": ((8, 16, 32, 64), (2, 0, 3, 1), torch.float16),
    "nchw_to_nhwc": (
        (8, 64, 32, 32),
        (0, 2, 3, 1),
        torch.float16,
    ),
    "nhwc_to_nchw": (
        (8, 32, 32, 64),
        (0, 3, 1, 2),
        torch.float16,
    ),
    "last2_tail": ((8, 17, 33), (0, 2, 1), torch.float16),
    "last2_fp32": ((8, 64, 128), (0, 2, 1), torch.float32),
}


def make_tensor(shape, dtype):
    if dtype in (torch.float32, torch.float16):
        return torch.randn(shape, dtype=dtype)
    if dtype == torch.int32:
        return torch.randint(-1_000_000, 1_000_000, shape, dtype=dtype)
    return torch.randint(-128, 128, shape, dtype=dtype)


def main():
    name = sys.argv[1]
    shape, dims, dtype = CASES[name]
    input_cpu = make_tensor(shape, dtype)
    expected = input_cpu.permute(dims).contiguous()
    input_npu = input_cpu.npu()
    for _ in range(10):
        output_npu = transpose_validation_lib.transpose(input_npu, dims)
    torch.npu.synchronize()
    for _ in range(30):
        output_npu = transpose_validation_lib.transpose(input_npu, dims)
    torch.npu.synchronize()
    if not torch.equal(output_npu.cpu(), expected):
        raise AssertionError(name)
    print(f"PROFILE_CASE_PASS {name}")


if __name__ == "__main__":
    main()
