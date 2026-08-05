import argparse
import json
import statistics

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080517
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
CASES = (
    ("last_gap_medium", (32, 1, 64, 1024), (0, 2, 3)),
    ("middle_gap_medium", (32, 1, 64, 1024), (0, 2)),
    ("last_gap_large", (200, 1, 1000, 64), (0, 2, 3)),
    ("middle_gap_large", (200, 1, 1000, 64), (0, 2)),
    ("multi_gap", (16, 1, 32, 1, 128), (0, 2, 4)),
)


def tolerance(dtype):
    if dtype == torch.float16:
        return 4e-3, 4e-3
    if dtype == torch.bfloat16:
        return 4e-2, 4e-2
    return 2e-4, 2e-4


def run_once(input_npu, axes, output_shape):
    return square_sum_v1_validation_lib.square_sum_v1(
        input_npu,
        axes,
        False,
        list(output_shape),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    passed = 0
    for case_index, (name, shape, axes) in enumerate(CASES):
        elements = 1
        for extent in shape:
            elements *= extent
        for dtype_index, dtype in enumerate(DTYPES):
            generator = torch.Generator().manual_seed(
                SEED + case_index * 10 + dtype_index
            )
            input_cpu = (
                torch.rand(
                    shape,
                    dtype=torch.float32,
                    generator=generator,
                )
                * 0.06
                - 0.03
            ).to(dtype)
            expected = torch.sum(torch.square(input_cpu), dim=axes)
            input_npu = input_cpu.npu()
            actual = None
            for _ in range(5):
                actual = run_once(input_npu, axes, expected.shape)
            torch.npu.synchronize()

            repeats = max(5, min(60, (1 << 25) // elements))
            samples = []
            for _ in range(7):
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(repeats):
                    actual = run_once(input_npu, axes, expected.shape)
                end.record()
                end.synchronize()
                samples.append(
                    start.elapsed_time(end) * 1000.0 / repeats
                )

            rtol, atol = tolerance(dtype)
            torch.testing.assert_close(
                actual.cpu(),
                expected,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
            passed += 1
            median_us = statistics.median(samples)
            bytes_per_element = 4 if dtype == torch.float32 else 2
            print(
                json.dumps(
                    {
                        "kind": "singleton_gap_sweep",
                        "label": args.label,
                        "case": name,
                        "shape": shape,
                        "axes": axes,
                        "dtype": str(dtype).replace("torch.", ""),
                        "elements": elements,
                        "outputs": expected.numel(),
                        "repeats": repeats,
                        "median_us": round(median_us, 6),
                        "effective_gbps": round(
                            elements
                            * bytes_per_element
                            / median_us
                            / 1000.0,
                            6,
                        ),
                        "samples_us": [
                            round(value, 6) for value in samples
                        ],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    print(f"SUMMARY_SINGLETON_GAP={passed}/{len(CASES) * len(DTYPES)}")


if __name__ == "__main__":
    main()
