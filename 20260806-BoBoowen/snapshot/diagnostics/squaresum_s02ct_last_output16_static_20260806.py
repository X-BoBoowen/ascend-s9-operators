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
    / "squaresum_s02ct_last_output16_splitk_20260806"
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
MIN_REDUCE = 2048
OLD_MAX_OUTPUTS = 8
NEW_MAX_OUTPUTS = 16
BLOCKS = 40


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def metadata(shape, axes):
    validate_shape(list(shape))
    normalized = normalize_axes(len(shape), axes)
    input_elements = math.prod(shape)
    reduce_elements = math.prod(shape[axis] for axis in normalized)
    return {
        "route": classify_route(shape, normalized),
        "input": input_elements,
        "reduce": reduce_elements,
        "output": input_elements // reduce_elements,
    }


def new_route(info):
    return (
        info["route"] == "fast1"
        and info["input"] >= MIN_INPUT
        and info["reduce"] >= MIN_REDUCE
        and OLD_MAX_OUTPUTS < info["output"] <= NEW_MAX_OUTPUTS
    )


def critical_ratio(info):
    return info["reduce"] / (
        info["output"] * math.ceil(info["reduce"] / BLOCKS)
    )


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
    baseline_host = (BASELINE / HOST).read_text(encoding="utf-8")
    candidate_host = (CANDIDATE / HOST).read_text(encoding="utf-8")
    assert "WORKSPACE_LAST_MAX_OUTPUTS = 8;" in baseline_host
    assert "WORKSPACE_LAST_MAX_OUTPUTS = 16;" in candidate_host
    assert candidate_host == baseline_host.replace(
        "WORKSPACE_LAST_MAX_OUTPUTS = 8;",
        "WORKSPACE_LAST_MAX_OUTPUTS = 16;",
        1,
    )
    for forbidden in ("Case1", "Case2", "Case3", "Case4", "Case5"):
        assert forbidden not in candidate_host


def main():
    source_contract()
    positives = (
        ("output9", (9, 64, 512), (1, 2)),
        ("output10_negative", (10, 64, 512), (-2, -1)),
        ("output15", (15, 64, 512), (1, 2)),
        ("output16", (16, 64, 512), (1, 2)),
        ("rank4_output16", (4, 4, 1000, 32), (2, 3)),
    )
    controls = (
        ("old_output8", (8, 64, 512), (1, 2)),
        ("output17", (17, 64, 512), (1, 2)),
        ("below_input", (16, 32, 511), (1, 2)),
        ("fast2", (9, 64, 512, 2), (1, 2)),
        ("small_public", (123, 31), (-1,)),
    )
    passed = 0
    for name, shape, axes in positives:
        info = metadata(shape, axes)
        assert new_route(info), (name, info)
        ratio = critical_ratio(info)
        assert ratio >= 2.49, (name, info, ratio)
        print(
            f"S02CT_ROUTE name={name} selected=1 {info} "
            f"modeled_critical_ratio={ratio:.3f} PASS"
        )
        passed += 1
    for name, shape, axes in controls:
        info = metadata(shape, axes)
        assert not new_route(info), (name, info)
        print(f"S02CT_CONTROL name={name} selected=0 {info} PASS")
        passed += 1
    print(
        f"S02CT_STATIC_SUMMARY passed={passed}/{passed} "
        f"candidate_host_sha256={sha256(CANDIDATE / HOST)} "
        f"candidate_kernel_sha256={sha256(CANDIDATE / KERNEL)}"
    )


if __name__ == "__main__":
    main()
