import statistics
import sys

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026073014


def measure(name, shape, dtype, axis, repeats=100, trials=5):
    generator = torch.Generator().manual_seed(SEED)
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * 2.0
        - 1.0
    ).to(dtype)
    expected = torch.sum(
        torch.square(input_cpu),
        dim=axis,
        keepdim=False,
    )
    input_npu = input_cpu.npu()

    result = None
    for _ in range(20):
        result = square_sum_v1_validation_lib.square_sum_v1(
            input_npu,
            axis,
            False,
            list(expected.shape),
        )
    torch.npu.synchronize()

    values = []
    for _ in range(trials):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            result = square_sum_v1_validation_lib.square_sum_v1(
                input_npu,
                axis,
                False,
                list(expected.shape),
            )
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / repeats)

    if dtype == torch.float16:
        rtol = atol = 3e-3
    elif dtype == torch.bfloat16:
        rtol = atol = 3e-2
    else:
        rtol = atol = 1e-4
    torch.testing.assert_close(
        result.cpu(),
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    print(
        f"{name}: median_us={statistics.median(values):.3f}, "
        f"trials_us={[round(value, 3) for value in values]}"
    )


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "middle"):
        for reduce_elements in (
            4096,
            8192,
            10000,
            16384,
            32768,
            65536,
            131072,
            200000,
        ):
            measure(
                f"middle_fp32_r{reduce_elements}_o64",
                (reduce_elements, 64),
                torch.float32,
                (0,),
            )
    if mode in ("all", "last"):
        for dtype_name, dtype in (
            ("fp16", torch.float16),
            ("bf16", torch.bfloat16),
        ):
            for reduce_elements in (
                262144,
                1048576,
                4194304,
                12582912,
            ):
                measure(
                    f"last_{dtype_name}_r{reduce_elements}_o1",
                    (reduce_elements,),
                    dtype,
                    (0,),
                )


if __name__ == "__main__":
    main()
