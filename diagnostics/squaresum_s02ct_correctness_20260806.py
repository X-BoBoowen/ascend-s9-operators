import gc
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib
from squaresum_official_profile_case_20260806 import official_style_verify


torch.npu.config.allow_internal_format = False
SEED = 2026080634
DTYPES = (
    ("fp16", torch.float16),
    ("bf16", torch.bfloat16),
    ("fp32", torch.float32),
)
CASES = (
    ("selected_output9", (9, 64, 512), (1, 2), False),
    ("selected_output10", (10, 64, 512), (-2, -1), True),
    ("selected_output15", (15, 64, 512), (1, 2), False),
    ("selected_output16", (16, 64, 512), (1, 2), True),
    ("selected_rank4", (4, 4, 1000, 32), (3, 2), False),
    ("control_output8", (8, 64, 512), (1, 2), False),
    ("control_output17", (17, 64, 512), (1, 2), False),
    ("control_below_input", (16, 32, 511), (1, 2), False),
    ("control_fast2", (9, 64, 512, 2), (1, 2), True),
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
    assert errors <= allowed
    print(
        "S02CT_CORRECTNESS "
        f"label={label} name={name} dtype={dtype_name} shape={shape} "
        f"axes={axes} keep_dims={int(keep_dims)} inputs={input_cpu.numel()} "
        f"outputs={expected.numel()} errors={errors} allowed={allowed} PASS"
    )
    del result, expected, input_cpu
    gc.collect()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "S02CT"
    passed = 0
    for case_index, case in enumerate(CASES):
        for dtype_name, dtype in DTYPES:
            run_case(label, case_index, case, dtype_name, dtype)
            passed += 1
    print(f"S02CT_CORRECTNESS_SUMMARY label={label} passed={passed}/{passed}")


if __name__ == "__main__":
    main()
