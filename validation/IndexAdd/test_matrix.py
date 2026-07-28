import torch
import torch_npu

import index_add_validation_lib


SEED = 2026072604
DTYPES = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int8,
)


def make_tensor(shape, dtype):
    if dtype in (torch.float32, torch.bfloat16, torch.float16):
        return torch.randn(shape, dtype=dtype)
    if dtype == torch.int32:
        return torch.randint(-100_000, 100_000, shape, dtype=dtype)
    return torch.randint(-100, 101, shape, dtype=dtype)


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
        raise AssertionError(f"{name}: integer mismatch={mismatch}")


def run_case(name, self_cpu, dim, index_cpu, source_cpu, repeats=1):
    expected = torch.index_add(
        self_cpu,
        dim,
        index_cpu,
        source_cpu,
    )
    self_npu = self_cpu.npu()
    index_npu = index_cpu.npu()
    source_npu = source_cpu.npu()
    for repeat in range(repeats):
        actual = index_add_validation_lib.index_add(
            self_npu,
            dim,
            index_npu,
            source_npu,
        ).cpu()
        assert_result(f"{name}/repeat{repeat}", actual, expected)
    print(
        f"PASS {name}: dtype={self_cpu.dtype}, "
        f"shape={tuple(self_cpu.shape)}, dim={dim}, "
        f"index={index_cpu.numel()}, repeats={repeats}"
    )


def main():
    torch.manual_seed(SEED)
    cases = 0
    for dtype in DTYPES:
        index = torch.tensor([4, 0, 4, 2, 0, 4], dtype=torch.int32)
        run_case(
            f"{dtype}_rank2_dim0_repeat",
            make_tensor((7, 37), dtype),
            0,
            index,
            make_tensor((index.numel(), 37), dtype),
            repeats=3,
        )
        cases += 1

        index = torch.tensor([16, 1, 16, 0, 8], dtype=torch.int32)
        run_case(
            f"{dtype}_rank2_negative_last",
            make_tensor((5, 17), dtype),
            -1,
            index,
            make_tensor((5, index.numel()), dtype),
        )
        cases += 1

        index = torch.tensor([6, 1, 6, 0], dtype=torch.int32)
        run_case(
            f"{dtype}_rank4_middle",
            make_tensor((2, 7, 3, 11), dtype),
            1,
            index,
            make_tensor((2, index.numel(), 3, 11), dtype),
        )
        cases += 1

        empty_index = torch.empty((0,), dtype=torch.int32)
        run_case(
            f"{dtype}_empty_index",
            make_tensor((3, 5, 7), dtype),
            1,
            empty_index,
            make_tensor((3, 0, 7), dtype),
        )
        cases += 1

    repeated_index = torch.zeros((256,), dtype=torch.int32)
    run_case(
        "float32_repeat_stress",
        torch.zeros((2, 131), dtype=torch.float32),
        0,
        repeated_index,
        torch.randn((256, 131), dtype=torch.float32) * 0.01,
        repeats=10,
    )
    cases += 1

    run_case(
        "int8_wraparound",
        torch.full((2, 33), 120, dtype=torch.int8),
        0,
        torch.zeros((8,), dtype=torch.int32),
        torch.full((8, 33), 120, dtype=torch.int8),
        repeats=5,
    )
    cases += 1

    max_repeat_index = torch.zeros((8000,), dtype=torch.int32)
    max_repeat_source = torch.empty(
        (8000, 17), dtype=torch.int8
    )
    max_repeat_source[0::2].fill_(127)
    max_repeat_source[1::2].fill_(-128)
    run_case(
        "int8_max_index_repeat_wraparound",
        torch.arange(34, dtype=torch.int8).reshape(2, 17),
        0,
        max_repeat_index,
        max_repeat_source,
        repeats=2,
    )
    cases += 1

    int32_self = torch.tensor(
        [[2**31 - 1, -(2**31)]],
        dtype=torch.int32,
    )
    int32_source = torch.tensor(
        [[1, -1], [2**31 - 1, -(2**31)]],
        dtype=torch.int32,
    )
    run_case(
        "int32_wraparound",
        int32_self,
        0,
        torch.tensor([0, 0], dtype=torch.int32),
        int32_source,
        repeats=5,
    )
    cases += 1

    torch.npu.synchronize()
    print(f"ALL PASS: {cases} directed cases")


if __name__ == "__main__":
    main()
