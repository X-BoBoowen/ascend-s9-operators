import statistics

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026072902


CASES = (
    ("fp16_key1_boundary", (2, 8192), torch.float16, (1,), False),
    ("fp16_key2_boundary", (2, 8193), torch.float16, (1,), False),
    ("bf16_key1_boundary", (2, 8192), torch.bfloat16, (1,), True),
    ("bf16_key2_boundary", (2, 8193), torch.bfloat16, (1,), True),
    ("fp32_key1_boundary", (2, 8192), torch.float32, (1,), False),
    ("fp32_key2_boundary", (2, 8193), torch.float32, (1,), False),
    (
        "fp16_long_multi_axis",
        (8, 8, 8192),
        torch.float16,
        (1, 2),
        False,
    ),
    (
        "bf16_long_multi_axis",
        (8, 8, 8192),
        torch.bfloat16,
        (1, 2),
        False,
    ),
    (
        "fp32_long_atomic",
        (8, 8, 8192),
        torch.float32,
        (1, 2),
        False,
    ),
    (
        "fp32_long_non_atomic",
        (9, 8, 8192),
        torch.float32,
        (1, 2),
        False,
    ),
)


def tolerances(dtype):
    if dtype == torch.float16:
        return 2e-3, 2e-4
    if dtype == torch.bfloat16:
        return 2e-2, 2e-3
    return 2e-5, 2e-6


def run_case(name, shape, dtype, axis, keep_dims):
    generator = torch.Generator().manual_seed(SEED + len(shape))
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator)
        * 0.02
        - 0.01
    ).to(dtype)
    expected = torch.sum(
        torch.square(input_cpu),
        dim=axis,
        keepdim=keep_dims,
    )
    input_npu = input_cpu.npu()
    outputs = []
    for _ in range(10):
        result = square_sum_v1_validation_lib.square_sum_v1(
            input_npu,
            axis,
            keep_dims,
            list(expected.shape),
        )
        outputs.append(result.cpu())

    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(
        outputs[-1],
        expected,
        rtol=rtol,
        atol=atol,
    )
    errors = [
        torch.max(torch.abs(output.float() - expected.float())).item()
        for output in outputs
    ]
    variation = max(
        torch.max(torch.abs(output.float() - outputs[0].float())).item()
        for output in outputs
    )
    print(
        f"PASS {name}: dtype={dtype}, shape={shape}, axis={axis}, "
        f"keep_dims={keep_dims}, max_abs_error={max(errors):.9g}, "
        f"median_abs_error={statistics.median(errors):.9g}, "
        f"run_variation={variation:.9g}"
    )


for case in CASES:
    run_case(*case)

print(f"ALL TILING-KEY PROBES PASS: {len(CASES)} cases")
