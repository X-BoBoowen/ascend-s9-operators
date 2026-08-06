import gc
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib
from squaresum_official_profile_case_20260806 import official_style_verify


torch.npu.config.allow_internal_format = False
SEED = 2026080631
DTYPES = (
    ("fp16", torch.float16),
    ("bf16", torch.bfloat16),
    ("fp32", torch.float32),
)
CASES = (
    ("selected_minimum", (32, 2, 64, 1, 64), (0, 2), False),
    ("selected_large", (128, 2, 512, 1, 64), (0, 2), False),
    ("selected_negative_axes", (64, 2, 512, 2, 64), (-5, -3), True),
    ("selected_output1024", (64, 2, 64, 4, 128), (0, 2), False),
    ("control_fast2", (128, 64, 64), (0, 1), False),
    ("control_output2050", (64, 2, 64, 5, 205), (0, 2), True),
)


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def run_case(label, case_index, case, dtype_name, dtype):
    name, shape, axes, keep_dims = case
    generator = torch.Generator().manual_seed(
        SEED + case_index * 31 + product(shape) % 104729
    )
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 0.06
        - 0.03
    ).to(dtype)
    expected = torch.sum(
        torch.square(input_cpu), dim=axes, keepdim=keep_dims
    )
    result = square_sum_v1_validation_lib.square_sum_v1(
        input_cpu.npu(), axes, keep_dims, list(expected.shape)
    )
    torch.npu.synchronize()
    errors, allowed = official_style_verify(result.cpu(), expected)
    print(
        "S03B_CORRECTNESS "
        f"label={label} name={name} dtype={dtype_name} shape={shape} "
        f"axes={axes} keep_dims={int(keep_dims)} inputs={input_cpu.numel()} "
        f"outputs={expected.numel()} errors={errors} allowed={allowed} PASS"
    )
    del result, expected, input_cpu
    gc.collect()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "S03B"
    passed = 0
    for case_index, case in enumerate(CASES):
        for dtype_name, dtype in DTYPES:
            run_case(label, case_index, case, dtype_name, dtype)
            passed += 1
    print(f"S03B_CORRECTNESS_SUMMARY label={label} passed={passed}/{passed}")


if __name__ == "__main__":
    main()
