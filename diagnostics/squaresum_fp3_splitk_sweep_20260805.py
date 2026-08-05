import statistics
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080504
AXES = (0, 2, 4)


# All layouts are fastPath3.  The first group varies the number of natural
# grouped-suffix DMA rows while keeping the trailing row width fixed.  The
# second group varies that width with at least 40 natural rows.  The final two
# cases vary the real interleaved output width.  Keeping that width greater
# than one is intentional: a unit extent removes the physical source gap and
# makes the layout contiguous after stride coalescing.
CASES = tuple(
    (
        f"rows_{rows}_tail4096_o8",
        (1, 1, rows, 8, 4096),
        rows,
        4096,
    )
    for rows in (8, 16, 24, 32, 40, 64, 80, 128)
) + (
    ("rows_128_tail256_o8", (1, 1, 128, 8, 256), 128, 256),
    ("rows_40_tail1024_o8", (1, 1, 40, 8, 1024), 40, 1024),
    ("rows_40_tail16384_o8", (1, 1, 40, 8, 16384), 40, 16384),
    ("rows_128_tail4096_o2", (1, 1, 128, 2, 4096), 128, 4096),
    ("rows_128_tail4096_o4", (1, 1, 128, 4, 4096), 128, 4096),
)

DTYPES = (
    ("fp16", torch.float16, 2),
    ("bf16", torch.bfloat16, 2),
    ("fp32", torch.float32, 4),
)


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def tolerances(dtype):
    if dtype == torch.float16:
        return 3e-3, 3e-3
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    return 1e-4, 1e-4


def measure(label, name, shape, natural_rows, tail, dtype_name, dtype, type_bytes):
    input_elements = product(shape)
    generator = torch.Generator().manual_seed(
        SEED + natural_rows + tail + type_bytes
    )
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 0.2
        - 0.1
    ).to(dtype)
    expected = torch.sum(torch.square(input_cpu), dim=AXES, keepdim=False)
    input_npu = input_cpu.npu()

    result = None
    for _ in range(15):
        result = square_sum_v1_validation_lib.square_sum_v1(
            input_npu, AXES, False, list(expected.shape)
        )
    torch.npu.synchronize()

    repeats = 10
    samples = []
    for _ in range(7):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            result = square_sum_v1_validation_lib.square_sum_v1(
                input_npu, AXES, False, list(expected.shape)
            )
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / repeats)

    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(
        result.cpu(), expected, rtol=rtol, atol=atol, equal_nan=True
    )
    median = statistics.median(samples)
    input_gbps = input_elements * type_bytes / median / 1000.0
    print(
        f"RESULT label={label} case={name} dtype={dtype_name} "
        f"natural_rows={natural_rows} tail={tail} "
        f"inputs={input_elements} outputs={expected.numel()} "
        f"median_us={median:.6f} input_gbps={input_gbps:.6f} "
        f"samples_us={[round(value, 6) for value in samples]}"
    )


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    passed = 0
    for name, shape, natural_rows, tail in CASES:
        for dtype_name, dtype, type_bytes in DTYPES:
            measure(
                label,
                name,
                shape,
                natural_rows,
                tail,
                dtype_name,
                dtype,
                type_bytes,
            )
            passed += 1
    print(f"SUMMARY label={label} passed={passed}/{len(CASES) * len(DTYPES)}")


if __name__ == "__main__":
    main()
