import gc
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib
from squaresum_official_profile_case_20260806 import official_style_verify


torch.npu.config.allow_internal_format = False
SEED = 2026080632
DTYPES = (
    ("fp16", torch.float16),
    ("bf16", torch.bfloat16),
    ("fp32", torch.float32),
)
CASES = (
    ("selected_tail1023", (40, 8, 1023), (0, 2), False),
    ("selected_tail512", (64, 8, 512), (0, 2), True),
    ("selected_tail200", (200, 8, 200), (0, -1), False),
    ("selected_tail1", (200, 200, 8, 1), (3, 1, 0), False),
    ("selected_output9", (80, 9, 512), (0, 2), False),
    ("selected_output16", (80, 16, 512), (0, 2), True),
    ("control_tail1024", (40, 8, 1024), (0, 2), False),
    ("control_output17", (80, 17, 512), (0, 2), True),
    ("control_rounding_cost", (41, 16, 1023), (0, 2), False),
    ("control_rows39", (39, 8, 1023), (0, 2), False),
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
        "S02CS_CORRECTNESS "
        f"label={label} name={name} dtype={dtype_name} shape={shape} "
        f"axes={axes} keep_dims={int(keep_dims)} inputs={input_cpu.numel()} "
        f"outputs={expected.numel()} errors={errors} allowed={allowed} PASS"
    )
    del result, expected, input_cpu
    gc.collect()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "S02CS"
    passed = 0
    for case_index, case in enumerate(CASES):
        for dtype_name, dtype in DTYPES:
            run_case(label, case_index, case, dtype_name, dtype)
            passed += 1
    print(f"S02CS_CORRECTNESS_SUMMARY label={label} passed={passed}/{passed}")


if __name__ == "__main__":
    main()
