import argparse
import json
import statistics

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080515
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
WIDTH_CASES = (
    (4, 16, "width4"),
    (6, 15, "width2"),
    (7, 31, "width1"),
)


def tolerance(dtype):
    if dtype == torch.float16:
        return 3e-3, 3e-3
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    return 2e-4, 2e-4


def run_once(input_npu, output_shape):
    return square_sum_v1_validation_lib.square_sum_v1(
        input_npu,
        (0, 2, 4),
        False,
        list(output_shape),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--batch-dims", default="127,2048,8192")
    args = parser.parse_args()
    batch_dims = tuple(
        int(value) for value in args.batch_dims.split(",")
    )

    total_cases = len(batch_dims) * len(WIDTH_CASES) * len(DTYPES)
    passed = 0
    for batch_index, batch_dim in enumerate(batch_dims):
        for width_index, (output_width, trailing, path) in enumerate(
            WIDTH_CASES
        ):
            shape = (2, 5, batch_dim, output_width, trailing)
            elements = 1
            for extent in shape:
                elements *= extent
            repeats = max(5, min(64, (1 << 26) // elements))
            for dtype_index, dtype in enumerate(DTYPES):
                generator = torch.Generator().manual_seed(
                    SEED
                    + batch_index * 100
                    + width_index * 10
                    + dtype_index
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
                input_npu = input_cpu.npu()
                expected = torch.sum(
                    torch.square(input_cpu), dim=(0, 2, 4)
                )

                actual = None
                for _ in range(6):
                    actual = run_once(input_npu, expected.shape)
                torch.npu.synchronize()

                samples = []
                for _ in range(7):
                    start = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    start.record()
                    for _ in range(repeats):
                        actual = run_once(input_npu, expected.shape)
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
                            "kind": "grouped_short_scale_sweep",
                            "label": args.label,
                            "path": path,
                            "shape": shape,
                            "dtype": str(dtype).replace("torch.", ""),
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
                    )
                )
    print(f"SUMMARY_GROUPED_SHORT={passed}/{total_cases}")


if __name__ == "__main__":
    main()
