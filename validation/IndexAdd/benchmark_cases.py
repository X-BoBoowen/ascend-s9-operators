import statistics
import sys

import torch
import torch_npu

import index_add_validation_lib
from profile_once import assert_result, build_case


torch.npu.config.allow_internal_format = False


def measure(name, repeats=40, trials=5):
    self_cpu, dim, index_cpu, source_cpu = build_case(name)
    expected = torch.index_add(
        self_cpu, dim, index_cpu, source_cpu
    )
    self_tensor = self_cpu.npu()
    index = index_cpu.npu()
    source = source_cpu.npu()
    for _ in range(10):
        result = index_add_validation_lib.index_add(
            self_tensor, dim, index, source
        )
    torch.npu.synchronize()
    values = []
    for _ in range(trials):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            result = index_add_validation_lib.index_add(
                self_tensor, dim, index, source
            )
        end.record()
        end.synchronize()
        values.append(
            start.elapsed_time(end) * 1000.0 / repeats
        )
    assert_result(name, result.cpu(), expected)
    print(
        f"{name}: median_us={statistics.median(values):.3f}, "
        f"trials_us={[round(value, 3) for value in values]}"
    )


def main():
    names = sys.argv[1:] or [
        "public",
        "fp16_dim0",
        "bf16_dim0",
        "fp32_middle",
        "int32_last",
        "int8_repeat",
        "index_max",
    ]
    for name in names:
        measure(name)


if __name__ == "__main__":
    main()
