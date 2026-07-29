import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False


CASES = (
    ("contiguous_63", (63,), 0.10009765625, (0,), False),
    ("middle_63", (63, 8), 0.10009765625, (0,), False),
    ("contiguous_10000", (10000,), 0.1005859375, (0,), True),
    ("sparse_7", (7, 4, 1), 0.10205078125, (0, 2), False),
)


for name, shape, value, axis, keep_dims in CASES:
    input_cpu = torch.full(shape, value, dtype=torch.bfloat16)
    expected = torch.sum(
        torch.square(input_cpu),
        dim=axis,
        keepdim=keep_dims,
    )
    actual = square_sum_v1_validation_lib.square_sum_v1(
        input_cpu.npu(),
        axis,
        keep_dims,
        list(expected.shape),
    ).cpu()
    if not torch.equal(actual, expected):
        raise AssertionError(
            f"{name}: expected={expected}, actual={actual}, "
            f"max_abs_error="
            f"{torch.max(torch.abs(actual.float() - expected.float())).item()}"
        )
    print(
        f"PASS {name}: shape={shape}, axis={axis}, "
        f"keep_dims={keep_dims}, output={actual}"
    )


print(f"ALL BF16 SEMANTIC PROBES PASS: {len(CASES)} cases")
