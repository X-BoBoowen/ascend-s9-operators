import argparse
import json
import statistics

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080523
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
CASES = (
    ("w1_threshold", (262144, 1), (0,)),
    ("w1_long", (1000000, 1), (0,)),
    ("w2_threshold", (131072, 2), (0,)),
    ("w2_long", (500000, 2), (0,)),
    ("w4_threshold", (65536, 4), (0,)),
    ("w4_long", (250000, 4), (0,)),
    ("w8_threshold", (32768, 8), (0,)),
    ("w8_long", (200000, 8), (0,)),
    ("w1_two_outer", (2, 262145, 1), (1,)),
    ("w4_two_outer", (2, 65537, 4), (1,)),
    ("w9_control", (30000, 9), (0,)),
    ("w16_control", (16385, 16), (0,)),
)


def tolerance(dtype):
    if dtype == torch.float16:
        return 4e-3, 4e-3
    if dtype == torch.bfloat16:
        return 4e-2, 4e-2
    return 2e-4, 2e-4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    passed = 0
    for case_index, (case, shape, axes) in enumerate(CASES):
        elements = 1
        for extent in shape:
            elements *= extent
        output_elements = elements
        for axis in axes:
            output_elements //= shape[axis]
        reduce_elements = elements // output_elements

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
                actual = square_sum_v1_validation_lib.square_sum_v1(
                    input_npu,
                    axes,
                    False,
                    list(expected.shape),
                )
            torch.npu.synchronize()

            repeats = max(5, min(60, (1 << 25) // elements))
            samples = []
            for _ in range(7):
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(repeats):
                    actual = square_sum_v1_validation_lib.square_sum_v1(
                        input_npu,
                        axes,
                        False,
                        list(expected.shape),
                    )
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
            print(
                json.dumps(
                    {
                        "kind": "small_inner_compact_sweep",
                        "label": args.label,
                        "case": case,
                        "shape": shape,
                        "axes": axes,
                        "dtype": str(dtype).replace("torch.", ""),
                        "reduce_elements": reduce_elements,
                        "output_elements": output_elements,
                        "elements": elements,
                        "repeats": repeats,
                        "median_us": round(
                            statistics.median(samples), 6
                        ),
                        "samples_us": [
                            round(value, 6) for value in samples
                        ],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    total = len(CASES) * len(DTYPES)
    print(f"SUMMARY_SMALL_INNER_COMPACT={passed}/{total}")


if __name__ == "__main__":
    main()
