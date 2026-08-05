import argparse
import json
import statistics

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080518
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
CASES = (
    (8, 64, 1024),
    (32, 64, 64),
    (32, 64, 256),
    (32, 64, 1024),
    (128, 64, 64),
    (64, 128, 512),
    (128, 256, 256),
    (100, 1000, 128),
    (200, 1000, 8),
    (200, 1000, 64),
)
AXES = (0, 2)


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
    for case_index, (outer_a, outer_b, outputs) in enumerate(CASES):
        shape = (outer_a, 1, outer_b, outputs)
        reduce_elements = outer_a * outer_b
        elements = reduce_elements * outputs
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
            expected = torch.sum(torch.square(input_cpu), dim=AXES)
            input_npu = input_cpu.npu()

            actual = None
            for _ in range(5):
                actual = square_sum_v1_validation_lib.square_sum_v1(
                    input_npu,
                    AXES,
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
                        AXES,
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
                        "kind": "singleton_middle_crossover",
                        "label": args.label,
                        "shape": shape,
                        "dtype": str(dtype).replace("torch.", ""),
                        "reduce_elements": reduce_elements,
                        "outputs": outputs,
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
    print(f"SUMMARY_CROSSOVER={passed}/{len(CASES) * len(DTYPES)}")


if __name__ == "__main__":
    main()
