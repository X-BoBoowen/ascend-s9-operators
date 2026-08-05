import json

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
AXES = (1, 3)


def main():
    passed = 0
    for output_width, expected_path in (
        (4, "width4"),
        (6, "width2"),
        (5, "width1"),
    ):
        shape = (2, 3, output_width, 64)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            input_cpu = torch.full(shape, 0.5, dtype=dtype)
            expected = torch.sum(torch.square(input_cpu), dim=AXES)
            actual = square_sum_v1_validation_lib.square_sum_v1(
                input_cpu.npu(),
                AXES,
                False,
                list(expected.shape),
            ).cpu()
            difference = actual.float() - expected.float()
            wrong = torch.nonzero(
                difference.abs() > 0,
                as_tuple=False,
            )
            print(
                json.dumps(
                    {
                        "path": expected_path,
                        "shape": shape,
                        "dtype": str(dtype),
                        "outputs": actual.numel(),
                        "wrong": wrong.shape[0],
                        "max_abs_diff": float(difference.abs().max()),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if wrong.numel() != 0:
                raise AssertionError(
                    f"{expected_path} {dtype}: {wrong.shape[0]} wrong"
                )
            passed += 1
    print(f"SUMMARY: {passed}/9 passed")


if __name__ == "__main__":
    main()
