import statistics

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026072903


CASES = (
    (
        "fp16_last_workspace",
        (8, 8, 8192),
        torch.float16,
        (1, 2),
        False,
    ),
    (
        "bf16_last_workspace",
        (8, 8, 8192),
        torch.bfloat16,
        (1, 2),
        True,
    ),
    (
        "fp16_last_output9_fallback",
        (9, 8, 8192),
        torch.float16,
        (1, 2),
        False,
    ),
    (
        "fp16_axis0_workspace",
        (10000, 64),
        torch.float16,
        (0,),
        False,
    ),
    (
        "bf16_axis0_workspace",
        (10000, 64),
        torch.bfloat16,
        (0,),
        True,
    ),
    (
        "fp32_axis0_workspace",
        (10000, 64),
        torch.float32,
        (0,),
        False,
    ),
    (
        "fp16_prefix_workspace",
        (200, 1000, 64),
        torch.float16,
        (0, 1),
        False,
    ),
    (
        "fp32_prefix_workspace",
        (200, 1000, 64),
        torch.float32,
        (0, 1),
        True,
    ),
    (
        "fp32_middle_output1024",
        (2048, 1024),
        torch.float32,
        (0,),
        False,
    ),
    (
        "fp32_middle_output1025_fallback",
        (2048, 1025),
        torch.float32,
        (0,),
        False,
    ),
)


def tolerances(dtype):
    if dtype == torch.float16:
        return 3e-3, 3e-3
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    return 2e-5, 2e-5


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
        output = square_sum_v1_validation_lib.square_sum_v1(
            input_npu,
            axis,
            keep_dims,
            list(expected.shape),
        )
        outputs.append(output.cpu())

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

print(f"ALL WORKSPACE PROBES PASS: {len(CASES)} cases")
