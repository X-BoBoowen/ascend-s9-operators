import argparse
import json
import statistics

import torch
import torch_npu

import square_sum_v1_validation_lib


torch.npu.config.allow_internal_format = False
SEED = 2026080213
DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Public contract plus one or more boundary representatives for every path
# changed after S02AA. These are differential gates, not guesses about the
# hidden benchmark shapes.
CORRECTNESS_CASES = (
    ((123, 31), (-1,), True, "public"),
    ((1, 65), (1,), False, "last_small_boundary"),
    ((7, 513), (1,), False, "last_small_multi_output"),
    ((31, 4096), (1,), False, "last_small_upper"),
    ((1, 262144), (1,), False, "last_workspace_output1"),
    ((8, 32768), (1,), False, "last_workspace_output8"),
    ((4096, 127), (0,), False, "middle_sequential"),
    ((256, 256, 232), (0, 1), False, "fp32_tree_tile232"),
    ((256, 256, 233), (0, 1), False, "fp32_tree_tile233"),
    ((2, 5, 127, 4, 16), (0, 2, 4), False, "short_width4"),
    ((2, 5, 127, 6, 15), (0, 2, 4), False, "short_width2"),
    ((2, 5, 127, 7, 31), (0, 2, 4), False, "short_width1"),
    ((2, 5, 127, 1, 63), (0, 2, 4), False, "short_singleton"),
    ((2, 5, 31, 7, 127), (0, 2, 4), False, "medium_grouped"),
    ((2, 3, 7, 4, 1025), (0, 2, 4), False, "long_grouped"),
)

PERFORMANCE_CASES = (
    ((1, 65), (1,), False, "last_small_boundary"),
    ((7, 513), (1,), False, "last_small_multi_output"),
    ((31, 4096), (1,), False, "last_small_upper"),
    ((1, 262144), (1,), False, "last_workspace_output1"),
    ((8, 32768), (1,), False, "last_workspace_output8"),
    ((4096, 127), (0,), False, "middle_sequential"),
    ((256, 256, 232), (0, 1), False, "fp32_tree_tile232"),
    ((2, 5, 127, 4, 16), (0, 2, 4), False, "short_width4"),
    ((2, 5, 127, 6, 15), (0, 2, 4), False, "short_width2"),
    ((2, 5, 127, 7, 31), (0, 2, 4), False, "short_width1"),
    ((2, 5, 31, 7, 127), (0, 2, 4), False, "medium_grouped"),
    ((2, 3, 7, 4, 1025), (0, 2, 4), False, "long_grouped"),
)


def dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def tolerance(dtype):
    if dtype == torch.float16:
        return 3e-3, 3e-3
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    return 2e-4, 2e-4


def make_case(index, shape, axes, dtype):
    generator = torch.Generator().manual_seed(SEED + index)
    input_cpu = (
        torch.rand(shape, dtype=torch.float32, generator=generator)
        * 0.06
        - 0.03
    ).to(dtype)
    return input_cpu.npu()


def expected_result(input_npu, axes, keep_dims):
    return torch.sum(
        torch.square(input_npu.cpu()), dim=axes, keepdim=keep_dims
    )


def run_once(input_npu, axes, keep_dims, output_shape):
    return square_sum_v1_validation_lib.square_sum_v1(
        input_npu,
        axes,
        keep_dims,
        list(output_shape),
    )


def assert_result(actual, expected):
    rtol, atol = tolerance(expected.dtype)
    torch.testing.assert_close(
        actual.cpu(), expected, rtol=rtol, atol=atol, equal_nan=True
    )


def selected_cases(cases, paths):
    if not paths:
        return cases
    wanted = set(paths.split(","))
    selected = tuple(case for case in cases if case[3] in wanted)
    if not selected:
        raise ValueError("--paths did not match any case")
    return selected


def correctness(paths):
    cases = selected_cases(CORRECTNESS_CASES, paths)
    passed = 0
    total = len(cases) * len(DTYPES)
    for case_index, (shape, axes, keep_dims, path) in enumerate(
        cases
    ):
        for dtype_index, dtype in enumerate(DTYPES):
            input_npu = make_case(
                case_index * len(DTYPES) + dtype_index,
                shape,
                axes,
                dtype,
            )
            expected = expected_result(input_npu, axes, keep_dims)
            actual = run_once(
                input_npu, axes, keep_dims, expected.shape
            )
            torch.npu.synchronize()
            assert_result(actual, expected)
            passed += 1
            print(
                json.dumps(
                    {
                        "kind": "correctness",
                        "path": path,
                        "shape": shape,
                        "axes": axes,
                        "keep_dims": keep_dims,
                        "dtype": dtype_name(dtype),
                        "status": "pass",
                    },
                    ensure_ascii=False,
                )
            )
    print(f"SUMMARY_CORRECTNESS={passed}/{total}")


def performance(label, paths):
    cases = selected_cases(PERFORMANCE_CASES, paths)
    measured = 0
    total = len(cases) * len(DTYPES)
    for case_index, (shape, axes, keep_dims, path) in enumerate(
        cases
    ):
        elements = 1
        for extent in shape:
            elements *= extent
        repeats = max(5, min(80, 16_000_000 // elements))
        for dtype_index, dtype in enumerate(DTYPES):
            input_npu = make_case(
                case_index * len(DTYPES) + dtype_index,
                shape,
                axes,
                dtype,
            )
            expected = expected_result(input_npu, axes, keep_dims)
            actual = None
            for _ in range(5):
                actual = run_once(
                    input_npu, axes, keep_dims, expected.shape
                )
            torch.npu.synchronize()

            samples = []
            for _ in range(5):
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(repeats):
                    actual = run_once(
                        input_npu, axes, keep_dims, expected.shape
                    )
                end.record()
                end.synchronize()
                samples.append(
                    start.elapsed_time(end) * 1000.0 / repeats
                )
            assert_result(actual, expected)
            measured += 1
            print(
                json.dumps(
                    {
                        "kind": "performance",
                        "label": label,
                        "path": path,
                        "shape": shape,
                        "axes": axes,
                        "keep_dims": keep_dims,
                        "dtype": dtype_name(dtype),
                        "repeats": repeats,
                        "median_us": round(
                            statistics.median(samples), 6
                        ),
                        "samples_us": [
                            round(value, 6) for value in samples
                        ],
                    },
                    ensure_ascii=False,
                )
            )
    print(f"SUMMARY_PERFORMANCE={measured}/{total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("correctness", "performance"))
    parser.add_argument("--label", default="candidate")
    parser.add_argument(
        "--paths",
        default="",
        help="comma-separated path labels; empty runs the full matrix",
    )
    args = parser.parse_args()
    if args.mode == "correctness":
        correctness(args.paths)
    else:
        performance(args.label, args.paths)


if __name__ == "__main__":
    main()
