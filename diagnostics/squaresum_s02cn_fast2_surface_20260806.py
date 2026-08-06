import gc
import statistics
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080618
TARGET_INPUTS = 16 * 1024 * 1024
OUTER = 64
INNERS = (
    2,
    3,
    4,
    5,
    7,
    8,
    15,
    16,
    17,
    31,
    32,
    33,
    63,
    64,
    65,
    127,
    128,
    129,
    255,
    256,
    257,
    511,
    512,
    1024,
)
DTYPES = (
    ("fp16", torch.float16, 2, 3e-3),
    ("fp32", torch.float32, 4, 3e-4),
)


def run_case(label, inner, dtype_name, dtype, type_bytes, tolerance):
    reduce = max(2, TARGET_INPUTS // (OUTER * inner))
    shape = (OUTER, reduce, inner)
    generator = torch.Generator().manual_seed(SEED + inner)
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 0.02
        - 0.01
    ).to(dtype)
    expected = torch.sum(torch.square(input_cpu), dim=(1,))
    input_npu = input_cpu.npu()
    result = None
    for _ in range(3):
        result = square_sum_v1_validation_lib.square_sum_v1(
            input_npu, (1,), False, list(expected.shape)
        )
    torch.npu.synchronize()

    samples = []
    for _ in range(5):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(2):
            result = square_sum_v1_validation_lib.square_sum_v1(
                input_npu, (1,), False, list(expected.shape)
            )
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 500.0)

    actual = result.cpu()
    torch.testing.assert_close(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
        equal_nan=True,
    )
    median = statistics.median(samples)
    input_gbps = input_cpu.numel() * type_bytes / median / 1000.0
    print(
        f"S02CN_FAST2 label={label} dtype={dtype_name} inner={inner} "
        f"outer={OUTER} reduce={reduce} inputs={input_cpu.numel()} "
        f"outputs={expected.numel()} median_us={median:.6f} "
        f"input_gbps={input_gbps:.6f} "
        f"samples_us={[round(value, 3) for value in samples]} PASS"
    )
    del actual, result, input_npu, expected, input_cpu
    gc.collect()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    passed = 0
    for inner in INNERS:
        for dtype_info in DTYPES:
            run_case(label, inner, *dtype_info)
            passed += 1
    print(f"S02CN_FAST2_SUMMARY label={label} passed={passed}/{len(INNERS) * 2}")


if __name__ == "__main__":
    main()
