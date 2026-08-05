import argparse
import json
import statistics

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080514

CASES = tuple(
    (outputs, total // outputs)
    for total in (1 << 18, 1 << 20, 1 << 22)
    for outputs in (1, 2, 4, 8)
)


def run_once(input_npu, outputs, reduce):
    return square_sum_v1_validation_lib.square_sum_v1(
        input_npu,
        (1,),
        False,
        [outputs],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    passed = 0
    for case_index, (outputs, reduce) in enumerate(CASES):
        generator = torch.Generator().manual_seed(SEED + case_index)
        input_cpu = (
            torch.rand(
                (outputs, reduce),
                dtype=torch.float32,
                generator=generator,
            )
            * 0.06
            - 0.03
        )
        input_npu = input_cpu.npu()
        expected = torch.sum(torch.square(input_cpu), dim=(1,))

        actual = None
        for _ in range(8):
            actual = run_once(input_npu, outputs, reduce)
        torch.npu.synchronize()

        repeats = max(8, min(100, (1 << 25) // (outputs * reduce)))
        samples = []
        for _ in range(7):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                actual = run_once(input_npu, outputs, reduce)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0 / repeats)

        torch.testing.assert_close(
            actual.cpu(),
            expected,
            rtol=2e-4,
            atol=2e-4,
            equal_nan=True,
        )
        passed += 1
        print(
            json.dumps(
                {
                    "kind": "atomic_workspace_sweep",
                    "label": args.label,
                    "outputs": outputs,
                    "reduce": reduce,
                    "elements": outputs * reduce,
                    "repeats": repeats,
                    "median_us": round(statistics.median(samples), 6),
                    "samples_us": [round(value, 6) for value in samples],
                },
                ensure_ascii=False,
            )
        )
    print(f"SUMMARY_SWEEP={passed}/{len(CASES)}")


if __name__ == "__main__":
    main()
