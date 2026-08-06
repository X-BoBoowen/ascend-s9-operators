import argparse
import importlib
import json
from pathlib import Path

import torch
import torch_npu


DEFAULT_MATRIX = Path(__file__).with_name(
    "squaresum_official_profile_matrix_20260806.json"
)
DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def load_case(path, name):
    document = json.loads(path.read_text(encoding="utf-8"))
    matches = [case for case in document["cases"] if case["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one matrix case named {name!r}")
    return matches[0]


def official_style_verify(actual, expected):
    if expected.dtype == torch.float32:
        rtol = 1e-4
        atol = 1e-4
        loss = 1e-4
    else:
        rtol = 1e-2
        atol = 1e-2
        loss = 1e-3

    minimum = torch.tensor(1e-9, dtype=expected.dtype)
    expected = torch.where(expected == 0, minimum, expected)
    actual = torch.where(actual == 0, minimum, actual)
    absolute = torch.abs(actual - expected)
    relative = absolute / torch.maximum(torch.abs(actual), torch.abs(expected))
    close = (absolute <= atol) | (relative <= rtol)
    close = close | (torch.isnan(actual) & torch.isnan(expected))
    errors = int(torch.sum(~close).item())
    allowed = actual.numel() * loss
    if errors > allowed:
        raise AssertionError(
            f"official-style verification failed: errors={errors}, "
            f"allowed={allowed}, elements={actual.numel()}"
        )
    return errors, allowed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--module", default="square_sum_v1_validation_lib")
    parser.add_argument("--function", default="square_sum_v1")
    args = parser.parse_args()

    torch.npu.config.allow_internal_format = False
    case = load_case(args.matrix.resolve(), args.case)
    dtype = DTYPES[args.dtype]
    axes = tuple(case["axes"])
    keep_dims = bool(case["keep_dims"])

    seed = 2026080600 + sum(case["shape"])
    generator = torch.Generator().manual_seed(seed)
    input_cpu = (
        torch.rand(case["shape"], dtype=torch.float32, generator=generator)
        * 0.06
        - 0.03
    ).to(dtype)
    expected = torch.sum(
        torch.square(input_cpu), dim=axes, keepdim=keep_dims
    )
    input_npu = input_cpu.npu()

    module = importlib.import_module(args.module)
    operator = getattr(module, args.function)

    # Match the competition wrapper's task stream: one 4096x4096 FP32 Mul
    # followed by one SquareSumV1 invocation, repeated exactly 30 times.
    a = torch.empty((4096, 4096), device="npu", dtype=torch.float32)
    b = torch.empty((4096, 4096), device="npu", dtype=torch.float32)
    c = torch.empty((4096, 4096), device="npu", dtype=torch.float32)
    result = None
    for _ in range(30):
        torch.mul(a, b, out=c)
        result = operator(input_npu, axes, keep_dims, list(expected.shape))
    torch.npu.synchronize()

    actual = result.cpu()
    errors, allowed = official_style_verify(actual, expected)
    print(
        "OFFICIAL_PROFILE_CASE "
        f"label={args.label} case={args.case} route={case['route']} "
        f"dtype={args.dtype} shape={case['shape']} axes={list(axes)} "
        f"keep_dims={keep_dims} outputs={expected.numel()} "
        f"errors={errors} allowed={allowed} PASS"
    )


if __name__ == "__main__":
    main()
