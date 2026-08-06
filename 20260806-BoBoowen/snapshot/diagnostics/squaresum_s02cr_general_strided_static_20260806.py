import hashlib
import math
from pathlib import Path

from validate_squaresum_profile_matrix_20260806 import (
    classify_route,
    normalize_axes,
    validate_shape,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE = (
    REPO_ROOT
    / "baselines"
    / "squaresum_s02ca_formal_best_20260806"
    / "SquareSumV1"
)
CANDIDATE = (
    REPO_ROOT
    / "candidates"
    / "squaresum_s02cr_general_strided_splitk_20260806"
    / "SquareSumV1"
)
BASELINE_HOST_SHA256 = (
    "0B5C6AEE3B01A63192A6CD4A59CF77A19373B6423274DC73968BAA77D84E9203"
)
BASELINE_KERNEL_SHA256 = (
    "C6EA5927D44DDF905E50B309C08E811E9DF614B7018D47F2BEB6A71115EC4C80"
)

MAX_BLOCKS = 40
OUTPUTS_PER_CORE = 64
OUTPUT_CHUNK = 1024
MIN_INPUT = 1 << 18
MIN_REDUCE = 2048
MAX_OUTPUTS = 8192


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def metadata(shape, axes):
    validate_shape(list(shape))
    normalized = normalize_axes(len(shape), axes)
    reduced = [axis in normalized for axis in range(len(shape))]
    last_reduced = normalized[-1]
    return {
        "shape": tuple(shape),
        "axes": tuple(normalized),
        "route": classify_route(shape, normalized),
        "input": math.prod(shape),
        "reduce": math.prod(shape[axis] for axis in normalized),
        "output": math.prod(
            shape[axis] for axis in range(len(shape)) if not reduced[axis]
        ),
        "inner": math.prod(shape[last_reduced + 1 :]),
        "last_reduce_dim": shape[last_reduced],
    }


def existing_grouped_splitk(info):
    # Exact sufficient conditions for the focused grouped control below.  All
    # positive S02CR cases use inner>16, so they cannot enter this old path.
    if info["route"] != "fast4" or info["inner"] > 16:
        return False
    last_reduce = info["last_reduce_dim"]
    return (
        last_reduce > 1
        and last_reduce <= 64
        and last_reduce & (last_reduce - 1) == 0
        and info["inner"] >= 4
        and info["reduce"] // last_reduce >= MAX_BLOCKS
        and info["input"] >= MIN_INPUT
        and info["output"] <= 512
    )


def select_general_splitk(info):
    if (
        info["route"] != "fast4"
        or existing_grouped_splitk(info)
        or info["input"] < MIN_INPUT
        or info["reduce"] < MIN_REDUCE
        or not 0 < info["output"] <= MAX_OUTPUTS
        or info["inner"] <= 0
        or info["output"] % info["inner"]
    ):
        return False, 0.0

    output_rows = info["output"] // info["inner"]
    chunks_per_row = math.ceil(info["inner"] / OUTPUT_CHUNK)
    output_chunks = output_rows * chunks_per_row
    if info["inner"] <= 16:
        existing_blocks = min(info["output"], MAX_BLOCKS)
    else:
        existing_blocks = min(
            math.ceil(info["output"] / OUTPUTS_PER_CORE), MAX_BLOCKS
        )
    baseline_cost = (
        math.ceil(output_chunks / existing_blocks) * info["reduce"]
    )
    splitk_cost = output_chunks * math.ceil(info["reduce"] / MAX_BLOCKS)
    ratio = baseline_cost / splitk_cost
    return baseline_cost >= 2 * splitk_cost, ratio


def assert_source_contract():
    baseline_host = BASELINE / "op_host" / "square_sum_v1.cpp"
    baseline_kernel = BASELINE / "op_kernel" / "square_sum_v1.cpp"
    candidate_host = CANDIDATE / "op_host" / "square_sum_v1.cpp"
    candidate_kernel = CANDIDATE / "op_kernel" / "square_sum_v1.cpp"
    assert sha256(baseline_host) == BASELINE_HOST_SHA256
    assert sha256(baseline_kernel) == BASELINE_KERNEL_SHA256

    host_text = candidate_host.read_text(encoding="utf-8")
    kernel_text = candidate_kernel.read_text(encoding="utf-8")
    assert host_text.count("reduceMode = 6U;") == 1
    assert host_text.count("generalStridedSplitK") == 6
    assert kernel_text.count("ProcessParallelStrided(") == 2
    assert kernel_text.count("reduceMode_ == 6U") == 3
    assert "BaseInputOffset(outputIndex)" in kernel_text
    assert "ReduceInputOffset(flatReduceIndex)" in kernel_text
    assert "copyParams.blockCount =\n                    static_cast<uint16_t>(\n                        currentReduceRows);" in kernel_text
    assert "rowsUntilBoundary" in kernel_text

    for relative in (
        Path("op_host/CMakeLists.txt"),
        Path("op_host/square_sum_v1_tiling.h"),
        Path("op_kernel/CMakeLists.txt"),
    ):
        assert (BASELINE / relative).read_bytes() == (CANDIDATE / relative).read_bytes()


def main():
    assert_source_contract()
    positive = (
        ("one_chunk_128", (64, 2, 512, 1, 64), (0, 2)),
        ("one_chunk_400", (200, 2, 512, 1, 200), (0, 2)),
        ("two_rows_128", (64, 2, 512, 2, 64), (0, 2)),
        ("wide_six_chunks", (64, 2, 512, 1, 2049), (0, 2)),
        ("minimum_gates", (32, 2, 64, 1, 64), (0, 2)),
        ("output8192", (32, 2, 64, 2, 2048), (0, 2)),
    )
    controls = (
        ("contiguous_fast2", (200, 1000, 64), (0, 1)),
        ("below_input_gate", (8, 2, 64, 2, 64), (0, 2)),
        ("below_reduce_gate", (8, 4, 128, 4, 64), (0, 2)),
        ("existing_grouped_splitk", (64, 8, 64, 2, 4), (0, 2)),
        ("output8196", (32, 2, 64, 2, 2049), (0, 2)),
        ("workspace_bound", (40, 16, 512, 16, 64), (0, 2)),
    )

    passed = 0
    for name, shape, axes in positive:
        info = metadata(shape, axes)
        selected, ratio = select_general_splitk(info)
        assert info["route"] == "fast4"
        assert selected, (name, info, ratio)
        print(
            f"S02CR_ROUTE name={name} selected=1 route={info['route']} "
            f"input={info['input']} output={info['output']} "
            f"reduce={info['reduce']} inner={info['inner']} "
            f"modeled_critical_ratio={ratio:.3f} PASS"
        )
        passed += 1

    for name, shape, axes in controls:
        info = metadata(shape, axes)
        selected, ratio = select_general_splitk(info)
        assert not selected, (name, info, ratio)
        print(
            f"S02CR_CONTROL name={name} selected=0 route={info['route']} "
            f"input={info['input']} output={info['output']} "
            f"reduce={info['reduce']} inner={info['inner']} PASS"
        )
        passed += 1

    print(
        f"S02CR_STATIC_SUMMARY passed={passed}/{passed} "
        f"candidate_host_sha256={sha256(CANDIDATE / 'op_host' / 'square_sum_v1.cpp')} "
        f"candidate_kernel_sha256={sha256(CANDIDATE / 'op_kernel' / 'square_sum_v1.cpp')}"
    )


if __name__ == "__main__":
    main()
