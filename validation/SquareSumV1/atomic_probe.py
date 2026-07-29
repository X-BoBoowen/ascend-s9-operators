import statistics

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026072901


CASES = (
    ((262144,), (0,), False),
    ((8, 65536), (1,), False),
    ((4, 3, 65536), (1, 2), False),
    ((2, 3, 4, 32768), (1, 2, 3), True),
)


def run_case(shape, axis, keep_dims):
    generator = torch.Generator().manual_seed(SEED + len(shape))
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator)
        * 0.02
        - 0.01
    )
    expected = torch.sum(
        torch.square(input_cpu),
        dim=axis,
        keepdim=keep_dims,
    )
    input_npu = input_cpu.npu()
    outputs = []
    for _ in range(30):
        result = square_sum_v1_validation_lib.square_sum_v1(
            input_npu,
            axis,
            keep_dims,
            list(expected.shape),
        )
        outputs.append(result.cpu())

    errors = [
        torch.max(torch.abs(output - expected)).item()
        for output in outputs
    ]
    variation = max(
        torch.max(torch.abs(output - outputs[0])).item()
        for output in outputs
    )
    torch.testing.assert_close(
        outputs[-1],
        expected,
        rtol=2e-5,
        atol=2e-6,
    )
    print(
        f"PASS shape={shape}, axis={axis}, keep_dims={keep_dims}, "
        f"output={tuple(expected.shape)}, "
        f"max_abs_error={max(errors):.9g}, "
        f"median_abs_error={statistics.median(errors):.9g}, "
        f"run_variation={variation:.9g}"
    )


for case in CASES:
    run_case(*case)

print(f"ALL ATOMIC PROBES PASS: {len(CASES)} cases")
