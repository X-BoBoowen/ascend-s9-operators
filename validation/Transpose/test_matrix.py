import torch
import torch_npu

import transpose_validation_lib


SEED = 2026072606
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
        return torch.randint(-1_000_000, 1_000_000, shape, dtype=dtype)
    return torch.randint(-128, 128, shape, dtype=dtype)


def run_case(name, input_cpu, dims):
    expected = input_cpu.permute(dims).contiguous()
    actual = transpose_validation_lib.transpose(
        input_cpu.npu(),
        dims,
    ).cpu()
    if not torch.equal(actual, expected):
        mismatch = torch.count_nonzero(actual != expected).item()
        raise AssertionError(
            f"{name}: mismatch={mismatch}, shape={tuple(input_cpu.shape)}, "
            f"dims={dims}, dtype={input_cpu.dtype}"
        )
    print(
        f"PASS {name}: dtype={input_cpu.dtype}, "
        f"shape={tuple(input_cpu.shape)}, dims={dims}"
    )


def main():
    torch.manual_seed(SEED)
    cases = (
        ((4,), (0,)),
        ((4, 3), (1, 0)),
        ((16, 16), (1, 0)),
        ((32, 48), (1, 0)),
        ((17, 131), (1, 0)),
        ((2, 3, 5), (2, 0, 1)),
        ((2, 3, 5), (-1, -3, -2)),
        ((2, 3, 5, 7), (0, 1, 2, 3)),
        ((2, 3, 5, 7), (2, 0, 3, 1)),
        ((2, 3, 2, 3, 2, 5), (5, 2, 0, 4, 1, 3)),
        ((0, 3, 5), (2, 0, 1)),
        ((41, 127, 43), (1, 2, 0)),
    )
    total = 0
    for dtype in DTYPES:
        for case_id, (shape, dims) in enumerate(cases):
            run_case(
                f"{dtype}_case{case_id}",
                make_tensor(shape, dtype),
                dims,
            )
            total += 1
    torch.npu.synchronize()
    print(f"ALL PASS: {total} directed cases")


if __name__ == "__main__":
    main()
