import argparse
import json
import statistics

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080524
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
CASES = (
    ("r63", (63, 1), (0,)),
    ("r64", (64, 1), (0,)),
    ("r65", (65, 1), (0,)),
    ("r1024", (1024, 1), (0,)),
    ("r4096", (4096, 1), (0,)),
    ("r8192", (8192, 1), (0,)),
    ("r32768", (32768, 1), (0,)),
    ("r262144", (262144, 1), (0,)),
    ("r1000000", (1000000, 1), (0,)),
    ("two_outer", (2, 262145, 1), (1,)),
    ("six_outer", (2, 3, 32769, 1), (2,)),
    ("gap_and_trailing", (3, 131, 1, 251, 1), (1, 3)),
    ("two_trailing", (262145, 1, 1), (0,)),
    ("width2_control", (131072, 2), (0,)),
    ("width4_control", (65536, 4), (0,)),
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
                        "kind": "trailing_singleton_sweep",
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
    print(f"SUMMARY_TRAILING_SINGLETON={passed}/{total}")


if __name__ == "__main__":
    main()
