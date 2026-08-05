import json

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False


CASES = (
    ("contiguous_middle", (4, 8, 200, 200), (0, 1)),
    ("strided_middle", (4, 2, 8, 200, 200), (0, 2)),
)


def main():
    failures = 0
    for case_name, shape, axes in CASES:
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            input_cpu = torch.full(shape, 0.5, dtype=dtype)
            expected = torch.sum(torch.square(input_cpu), dim=axes)
            actual = square_sum_v1_validation_lib.square_sum_v1(
                input_cpu.npu(),
                axes,
                False,
                list(expected.shape),
            ).cpu()
            difference = actual.float() - expected.float()
            wrong = torch.nonzero(
                difference.abs() > 0,
                as_tuple=False,
            )
            failures += int(wrong.numel() != 0)
            print(
                json.dumps(
                    {
                        "case": case_name,
                        "shape": shape,
                        "axes": axes,
                        "dtype": str(dtype),
                        "outputs": actual.numel(),
                        "wrong": wrong.shape[0],
                        "max_abs_diff": float(difference.abs().max()),
                        "actual_first8": [
                            float(item) for item in actual.flatten()[:8]
                        ],
                    },
                    ensure_ascii=False,
                )
            )
    if failures:
        raise SystemExit(f"FAILED: {failures}/{len(CASES) * 3}")
    print(f"SUMMARY: {len(CASES) * 3}/{len(CASES) * 3} passed")


if __name__ == "__main__":
    main()
