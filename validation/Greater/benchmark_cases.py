import statistics
import sys

import torch
import torch_npu

import greater_validation_lib
from profile_once import build_case


torch.npu.config.allow_internal_format = False


def measure(name, repeats=100, trials=5):
    self_tensor, other_tensor = build_case(name)
    expected = torch.gt(self_tensor.cpu(), other_tensor.cpu())
    for _ in range(20):
        result = greater_validation_lib.greater(
            self_tensor,
            other_tensor,
        )
    torch.npu.synchronize()
    values = []
    for _ in range(trials):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            result = greater_validation_lib.greater(
                self_tensor,
                other_tensor,
            )
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / repeats)
    if not torch.equal(result.cpu(), expected):
        raise AssertionError(name)
    print(
        f"{name}: median_us={statistics.median(values):.3f}, "
        f"trials_us={[round(value, 3) for value in values]}"
    )


def main():
    names = sys.argv[1:] or [
        "public",
        "same_large",
        "same_large_fp32",
        "same_large_bf16",
        "same_large_int8",
        "scalar_other",
        "scalar_self",
        "scalar_other_fp32",
        "scalar_other_bf16",
        "scalar_other_int8",
        "last_broadcast",
        "mixed_broadcast",
        "int32_same",
    ]
    for name in names:
        measure(name)


if __name__ == "__main__":
    main()
