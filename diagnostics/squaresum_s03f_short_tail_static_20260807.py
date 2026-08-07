import hashlib
import math
from pathlib import Path

from validate_squaresum_profile_matrix_20260806 import (
    classify_route,
    normalize_axes,
    validate_shape,
)


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "candidates/squaresum_s03e_s02f_splitk_singleton_20260807/SquareSumV1"
CANDIDATE = ROOT / "candidates/squaresum_s03f_s03e_short_tail_splitk_20260807/SquareSumV1"
HOST = Path("op_host/square_sum_v1.cpp")
KERNEL = Path("op_kernel/square_sum_v1.cpp")
HOST_SHA256 = "035EF0C90BC52C1F915EF5794A9BDD5EE62E15938C0D1DDA5E38A7F2DECF86C8"
KERNEL_SHA256 = "13585F20D40FF6D9B05CF813BBCDAEAF30F26368A7F57139F73A7AEDD19E0470"
MIN_INPUT = 1 << 18
MIN_REDUCE = 1 << 15
MIN_ROWS = 40
MAX_TAIL = 1023
MAX_OUTPUTS = 16
BLOCKS = 40


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def metadata(shape, axes):
    validate_shape(list(shape))
    normalized = normalize_axes(len(shape), axes)
    reduced = [axis in normalized for axis in range(len(shape))]
    input_elements = math.prod(shape)
    reduce_elements = math.prod(shape[axis] for axis in normalized)
    trailing = 1
    for axis in range(len(shape) - 1, -1, -1):
        if not reduced[axis]:
            break
        trailing *= shape[axis]
    return {
        "route": classify_route(shape, normalized),
        "input": input_elements,
        "reduce": reduce_elements,
        "output": input_elements // reduce_elements,
        "tail": trailing,
        "rows": reduce_elements // trailing,
    }


def selected(info):
    if not (
        info["route"] == "fast3"
        and info["input"] >= MIN_INPUT
        and info["reduce"] >= MIN_REDUCE
        and 0 < info["output"] <= MAX_OUTPUTS
        and 0 < info["tail"] <= MAX_TAIL
        and info["rows"] >= MIN_ROWS
    ):
        return False
    splitk_cost = info["output"] * math.ceil(info["rows"] / BLOCKS)
    return info["rows"] >= 2 * splitk_cost


def source_contract():
    assert sha256(CANDIDATE / HOST) == HOST_SHA256
    assert sha256(CANDIDATE / KERNEL) == KERNEL_SHA256
    base_files = {path.relative_to(BASE) for path in BASE.rglob("*") if path.is_file()}
    candidate_files = {
        path.relative_to(CANDIDATE)
        for path in CANDIDATE.rglob("*")
        if path.is_file()
    }
    assert base_files == candidate_files
    for relative in base_files - {HOST}:
        assert (BASE / relative).read_bytes() == (CANDIDATE / relative).read_bytes()
    host = (CANDIDATE / HOST).read_text(encoding="utf-8")
    assert host.count("noncontiguousShortTailSplitK") == 2
    assert "SHORT_TAIL_SPLITK_MIN_ROWS = 40U" in host
    assert "SHORT_TAIL_SPLITK_MAX_TAIL = 1023U" in host
    assert "splitKNaturalRows >= doubledShortTailSplitKCost" in host


def main():
    source_contract()
    positives = (
        ((40, 8, 1023), (0, 2)),
        ((64, 8, 512), (0, 2)),
        ((200, 8, 200), (0, -1)),
        ((200, 200, 8, 1), (0, 1, 3)),
        ((80, 16, 512), (0, 2)),
    )
    controls = (
        ((40, 8, 1024), (0, 2)),
        ((39, 8, 1023), (0, 2)),
        ((80, 17, 512), (0, 2)),
        ((41, 16, 1023), (0, 2)),
        ((4, 8192), (1,)),
    )
    passed = 0
    for shape, axes in positives:
        info = metadata(shape, axes)
        assert selected(info), (shape, axes, info)
        print(f"S03F_ROUTE shape={shape} axes={axes} info={info} PASS")
        passed += 1
    for shape, axes in controls:
        info = metadata(shape, axes)
        assert not selected(info), (shape, axes, info)
        print(f"S03F_CONTROL shape={shape} axes={axes} info={info} PASS")
        passed += 1
    print(f"S03F_STATIC_SUMMARY passed={passed}/{passed} host_sha256={sha256(CANDIDATE / HOST)}")


if __name__ == "__main__":
    main()
