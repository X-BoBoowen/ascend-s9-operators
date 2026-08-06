import hashlib
import math
from pathlib import Path

from validate_squaresum_profile_matrix_20260806 import (
    classify_route,
    normalize_axes,
    validate_shape,
)


ROOT = Path(__file__).resolve().parent.parent
BASELINE = (
    ROOT
    / "baselines"
    / "squaresum_s02ca_formal_best_20260806"
    / "SquareSumV1"
)
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02cs_short_tail_splitk_20260806"
    / "SquareSumV1"
)
HOST = Path("op_host/square_sum_v1.cpp")
KERNEL = Path("op_kernel/square_sum_v1.cpp")
BASELINE_HOST_SHA256 = (
    "0B5C6AEE3B01A63192A6CD4A59CF77A19373B6423274DC73968BAA77D84E9203"
)
BASELINE_KERNEL_SHA256 = (
    "C6EA5927D44DDF905E50B309C08E811E9DF614B7018D47F2BEB6A71115EC4C80"
)

MIN_INPUT = 1 << 18
MIN_REDUCE = 1 << 15
MIN_ROWS = 40
MAX_TAIL = 1023
MAX_OUTPUTS = 8
BLOCKS = 40


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def metadata(shape, axes):
    validate_shape(list(shape))
    normalized = normalize_axes(len(shape), axes)
    reduced = [axis in normalized for axis in range(len(shape))]
    input_elements = math.prod(shape)
    reduce_elements = math.prod(shape[axis] for axis in normalized)
    output_elements = input_elements // reduce_elements
    trailing = 1
    for axis in range(len(shape) - 1, -1, -1):
        if not reduced[axis]:
            break
        trailing *= shape[axis]
    return {
        "route": classify_route(shape, normalized),
        "input": input_elements,
        "reduce": reduce_elements,
        "output": output_elements,
        "tail": trailing,
        "rows": reduce_elements // trailing,
    }


def selected(info):
    return (
        info["route"] == "fast3"
        and info["input"] >= MIN_INPUT
        and info["reduce"] >= MIN_REDUCE
        and 0 < info["output"] <= MAX_OUTPUTS
        and 0 < info["tail"] <= MAX_TAIL
        and info["rows"] >= MIN_ROWS
    )


def modeled_critical_ratio(info):
    baseline_rows = info["rows"]
    split_rows = math.ceil(info["rows"] / BLOCKS)
    return baseline_rows / (info["output"] * split_rows)


def source_contract():
    assert sha256(BASELINE / HOST) == BASELINE_HOST_SHA256
    assert sha256(BASELINE / KERNEL) == BASELINE_KERNEL_SHA256
    assert sha256(CANDIDATE / KERNEL) == BASELINE_KERNEL_SHA256

    baseline_files = {
        path.relative_to(BASELINE)
        for path in BASELINE.rglob("*")
        if path.is_file()
    }
    candidate_files = {
        path.relative_to(CANDIDATE)
        for path in CANDIDATE.rglob("*")
        if path.is_file()
    }
    assert baseline_files == candidate_files
    for relative in baseline_files - {HOST}:
        assert (BASELINE / relative).read_bytes() == (
            CANDIDATE / relative
        ).read_bytes()

    host = (CANDIDATE / HOST).read_text(encoding="utf-8")
    assert host.count("noncontiguousShortTailSplitK") == 2
    assert "NONCONTIGUOUS_SHORT_SPLITK_MIN_ROWS = 40U" in host
    assert "NONCONTIGUOUS_SHORT_SPLITK_MAX_OUTPUTS = 8U" in host
    assert "trailingReduceElements < NONCONTIGUOUS_SPLITK_MIN_TAIL" in host
    assert "reduceMode = 3U;" in host
    assert "useTreeFinalize = true;" in host
    for forbidden in ("Case1", "Case2", "Case3", "Case4", "Case5"):
        assert forbidden not in host


def main():
    source_contract()
    positives = (
        ("boundary_40x1023", (40, 8, 1023), (0, 2)),
        ("tail512", (64, 8, 512), (0, 2)),
        ("tail200_negative", (200, 8, 200), (0, -1)),
        ("tail1_rank4", (200, 200, 8, 1), (0, 1, 3)),
        ("keepdims_shape", (64, 4, 2, 512), (0, 3)),
    )
    controls = (
        ("old_tail1024", (40, 8, 1024), (0, 2)),
        ("rows39", (39, 8, 1023), (0, 2)),
        ("output9", (40, 9, 1023), (0, 2)),
        ("below_reduce", (40, 8, 512), (0, 2)),
        ("contiguous_fast1", (4, 8192), (1,)),
        ("strided_fast4", (40, 2, 512, 4), (0, 2)),
    )

    passed = 0
    for name, shape, axes in positives:
        info = metadata(shape, axes)
        assert selected(info), (name, info)
        ratio = modeled_critical_ratio(info)
        assert ratio >= 4.0, (name, info, ratio)
        print(
            f"S02CS_ROUTE name={name} selected=1 {info} "
            f"modeled_critical_ratio={ratio:.3f} PASS"
        )
        passed += 1

    for name, shape, axes in controls:
        info = metadata(shape, axes)
        assert not selected(info), (name, info)
        print(f"S02CS_CONTROL name={name} selected=0 {info} PASS")
        passed += 1

    print(
        f"S02CS_STATIC_SUMMARY passed={passed}/{passed} "
        f"candidate_host_sha256={sha256(CANDIDATE / HOST)} "
        f"candidate_kernel_sha256={sha256(CANDIDATE / KERNEL)}"
    )


if __name__ == "__main__":
    main()
