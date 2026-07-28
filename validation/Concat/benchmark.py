import statistics

import torch
import torch_npu

import concat_validation_lib


def benchmark(name, shapes, dim, dtype, warmup=20, repeats=100):
    values = [
        torch.randn(shape, dtype=dtype).npu()
        if dtype in (torch.float16, torch.float32)
        else torch.randint(-100, 100, shape, dtype=dtype).npu()
        for shape in shapes
    ]
    for _ in range(warmup):
        concat_validation_lib.concat(values, dim)
    torch.npu.synchronize()

    elapsed_us = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        concat_validation_lib.concat(values, dim)
        end.record()
        end.synchronize()
        elapsed_us.append(start.elapsed_time(end) * 1000.0)

    ordered = sorted(elapsed_us)
    p50 = statistics.median(ordered)
    p90 = ordered[int(0.90 * (len(ordered) - 1))]
    print(
        f"{name}: min={min(ordered):.3f}us, "
        f"p50={p50:.3f}us, p90={p90:.3f}us"
    )


def main():
    benchmark(
        "public_case1",
        [(128, size) for size in (27, 40, 63, 24, 50, 26, 19, 2, 5)],
        -1,
        torch.float16,
    )
    benchmark(
        "wide_rows",
        [(64, 20000), (64, 19001)],
        1,
        torch.float16,
    )
    benchmark(
        "many_inputs",
        [(32, 1, 64)] * 33,
        1,
        torch.float16,
    )


if __name__ == "__main__":
    main()
