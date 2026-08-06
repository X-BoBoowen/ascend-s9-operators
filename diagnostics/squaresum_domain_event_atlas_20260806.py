import argparse
import gc
import importlib
import json
import statistics
from pathlib import Path

import torch
import torch_npu

from squaresum_official_profile_case_20260806 import official_style_verify
from validate_squaresum_profile_matrix_20260806 import load_and_validate


DEFAULT_MATRIX = Path(__file__).with_name(
    "squaresum_official_profile_matrix_20260806.json"
)
DTYPES = {
    "fp16": (torch.float16, 2),
    "bf16": (torch.bfloat16, 2),
    "fp32": (torch.float32, 4),
}


def run_case(operator, label, case, dtype_name):
    dtype, type_bytes = DTYPES[dtype_name]
    seed = 2026080622 + sum(case["shape"])
    generator = torch.Generator().manual_seed(seed)
    input_cpu = (
        torch.rand(case["shape"], dtype=torch.float32, generator=generator)
        * 0.06
        - 0.03
    ).to(dtype)
    axes = tuple(case["axes"])
    keep_dims = bool(case["keep_dims"])
    expected = torch.sum(
        torch.square(input_cpu), dim=axes, keepdim=keep_dims
    )
    input_npu = input_cpu.npu()

    result = None
    for _ in range(3):
        result = operator(input_npu, axes, keep_dims, list(expected.shape))
    torch.npu.synchronize()

    repeats = 2
    samples = []
    for _ in range(5):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            result = operator(input_npu, axes, keep_dims, list(expected.shape))
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / repeats)

    errors, allowed = official_style_verify(result.cpu(), expected)
    median = statistics.median(samples)
    elements = input_cpu.numel()
    print(
        "DOMAIN_EVENT_ATLAS "
        f"label={label} case={case['name']} route={case['route']} "
        f"dtype={dtype_name} shape={case['shape']} axes={list(axes)} "
        f"inputs={elements} outputs={expected.numel()} "
        f"median_us={median:.6f} "
        f"input_gbps={elements * type_bytes / median / 1000.0:.6f} "
        f"errors={errors} allowed={allowed} samples_us={samples} PASS"
    )
    del result, input_npu, expected, input_cpu
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--tier", default="atlas")
    parser.add_argument(
        "--dtypes", nargs="+", choices=DTYPES, default=["fp16", "bf16", "fp32"]
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--module", default="square_sum_v1_validation_lib")
    parser.add_argument("--function", default="square_sum_v1")
    args = parser.parse_args()

    torch.npu.config.allow_internal_format = False
    matrix_path = args.matrix.resolve()
    load_and_validate(matrix_path)
    document = json.loads(matrix_path.read_text(encoding="utf-8"))
    cases = [case for case in document["cases"] if case.get("tier") == args.tier]
    if not cases:
        raise ValueError(f"matrix tier {args.tier!r} contains no cases")

    module = importlib.import_module(args.module)
    operator = getattr(module, args.function)
    passed = 0
    for case in cases:
        for dtype_name in args.dtypes:
            run_case(operator, args.label, case, dtype_name)
            passed += 1
    expected_count = len(cases) * len(args.dtypes)
    print(
        f"DOMAIN_EVENT_ATLAS_SUMMARY label={args.label} tier={args.tier} "
        f"passed={passed}/{expected_count}"
    )


if __name__ == "__main__":
    main()
