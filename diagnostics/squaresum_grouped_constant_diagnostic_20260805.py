import json

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SHAPE = (8, 8, 16, 64, 16)
AXES = (0, 2, 4)
DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def run_once(input_npu, output_shape):
    return square_sum_v1_validation_lib.square_sum_v1(
        input_npu,
        AXES,
        False,
        list(output_shape),
    )


def main():
    for value in (1.0, 0.5):
        for dtype in DTYPES:
            input_cpu = torch.full(SHAPE, value, dtype=dtype)
            expected = torch.sum(
                torch.square(input_cpu), dim=AXES
            )
            actual = run_once(input_cpu.npu(), expected.shape).cpu()
            difference = actual.float() - expected.float()
            print(
                json.dumps(
                    {
                        "value": value,
                        "dtype": str(dtype).replace("torch.", ""),
                        "expected": float(expected.flatten()[0]),
                        "actual_min": float(actual.float().min()),
                        "actual_max": float(actual.float().max()),
                        "max_abs_difference": float(
                            difference.abs().max()
                        ),
                        "wrong": int(
                            torch.count_nonzero(difference).item()
                        ),
                        "outputs": actual.numel(),
                        "first16": [
                            float(item)
                            for item in actual.flatten()[:16]
                        ],
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
