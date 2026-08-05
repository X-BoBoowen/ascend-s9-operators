import json

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SHAPE = (31, 129, 33, 5)
AXES = (1, 2)
SEED = 2026080516


def tolerances(dtype):
    if dtype == torch.float16:
        return 4e-3, 4e-3
    if dtype == torch.bfloat16:
        return 4e-2, 4e-2
    return 4e-5, 4e-5


def check(label, input_cpu):
    expected = torch.sum(torch.square(input_cpu), dim=AXES)
    actual = square_sum_v1_validation_lib.square_sum_v1(
        input_cpu.npu(),
        AXES,
        False,
        list(expected.shape),
    ).cpu()
    actual_float = actual.float()
    expected_float = expected.float()
    absolute = (actual_float - expected_float).abs()
    rtol, atol = tolerances(input_cpu.dtype)
    allowed = atol + rtol * expected_float.abs()
    wrong = absolute > allowed
    print(
        json.dumps(
            {
                "label": label,
                "shape": SHAPE,
                "axes": AXES,
                "dtype": str(input_cpu.dtype),
                "outputs": actual.numel(),
                "wrong": int(wrong.sum()),
                "max_abs_diff": float(absolute.max()),
                "max_allowed": float(allowed.max()),
                "actual_first8": [
                    float(item) for item in actual.flatten()[:8]
                ],
                "expected_first8": [
                    float(item) for item in expected.flatten()[:8]
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if wrong.any():
        raise AssertionError(f"{label}: {int(wrong.sum())} wrong")


def main():
    for dtype_index, dtype in enumerate(
        (torch.float16, torch.bfloat16, torch.float32)
    ):
        check("constant", torch.full(SHAPE, 0.5, dtype=dtype))
        generator = torch.Generator().manual_seed(SEED + dtype_index)
        check(
            "random",
            (
                torch.rand(
                    SHAPE,
                    dtype=torch.float32,
                    generator=generator,
                )
                * 4.0
                - 2.0
            ).to(dtype),
        )
    print("SUMMARY: 6/6 passed")


if __name__ == "__main__":
    main()
