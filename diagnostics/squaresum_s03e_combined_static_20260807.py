import hashlib
import math
from pathlib import Path

from squaresum_s03d_singleton_gap_static_20260806 import (
    assert_coordinate_equivalence,
)
from validate_squaresum_profile_matrix_20260806 import (
    classify_route,
    normalize_axes,
    validate_shape,
)


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "baselines/squaresum_s02f_global_best_20260806/SquareSumV1"
S03B = ROOT / "candidates/squaresum_s03b_s02f_fast4_splitk_20260806/SquareSumV1"
CANDIDATE = ROOT / "candidates/squaresum_s03e_s02f_splitk_singleton_20260807/SquareSumV1"
HOST_SHA256 = "BD01AB1FD427BBCFD2D95CB55ADCF76E21CAC423BA9EB417AF8F581446211D10"
KERNEL_SHA256 = "13585F20D40FF6D9B05CF813BBCDAEAF30F26368A7F57139F73A7AEDD19E0470"
MIN_INPUT = 1 << 18
MIN_REDUCE = 2048
MAX_OUTPUTS = 1024


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def metadata(shape, axes):
    validate_shape(list(shape))
    normalized = normalize_axes(len(shape), axes)
    reduced = set(normalized)
    return {
        "route": classify_route(shape, normalized),
        "input": math.prod(shape),
        "reduce": math.prod(shape[axis] for axis in normalized),
        "output": math.prod(
            shape[axis] for axis in range(len(shape)) if axis not in reduced
        ),
        "inner": math.prod(shape[normalized[-1] + 1 :]),
    }


def splitk_selected(info):
    return (
        info["route"] == "fast4"
        and info["input"] >= MIN_INPUT
        and info["reduce"] >= MIN_REDUCE
        and 0 < info["output"] <= MAX_OUTPUTS
        and info["inner"] > 0
        and info["output"] % info["inner"] == 0
    )


def assert_source_contract():
    host = CANDIDATE / "op_host/square_sum_v1.cpp"
    kernel = CANDIDATE / "op_kernel/square_sum_v1.cpp"
    assert sha256(host) == HOST_SHA256
    assert sha256(kernel) == KERNEL_SHA256
    assert kernel.read_bytes() == (S03B / "op_kernel/square_sum_v1.cpp").read_bytes()
    for relative in (
        Path("op_host/CMakeLists.txt"),
        Path("op_host/square_sum_v1_tiling.h"),
        Path("op_kernel/CMakeLists.txt"),
    ):
        assert (BASE / relative).read_bytes() == (CANDIDATE / relative).read_bytes()

    host_text = host.read_text(encoding="utf-8")
    kernel_text = kernel.read_text(encoding="utf-8")
    assert host_text.count("hasSingletonGap") == 3
    assert host_text.count("reduceElements < 8192U") == 1
    assert host_text.count("generalStridedSplitK") == 2
    assert host_text.count("reduceMode = 3U;") == 1
    assert kernel_text.count("ProcessParallelStrided(") == 2
    assert kernel_text.count("reduceMode_ == 3U") == 3


def main():
    assert_source_contract()
    splitk_cases = (
        ((32, 2, 64, 1, 64), (0, 2)),
        ((128, 2, 512, 1, 64), (0, 2)),
        ((64, 2, 64, 4, 128), (0, 2)),
    )
    singleton_cases = (
        ((128, 1, 64), (0, 2), "fast1"),
        ((128, 1, 64, 64), (2, 0), "fast2"),
        ((3, 128, 1, 64, 16), (1, 3), "fast2"),
    )
    passed = 0
    for shape, axes in splitk_cases:
        info = metadata(shape, axes)
        assert splitk_selected(info), (shape, axes, info)
        print(f"S03E_SPLITK shape={shape} axes={axes} PASS")
        passed += 1
    for shape, axes, expected in singleton_cases:
        normalized = normalize_axes(len(shape), axes)
        actual = classify_route(shape, normalized)
        assert actual == expected, (shape, axes, actual, expected)
        assert_coordinate_equivalence(shape, axes)
        print(f"S03E_SINGLETON shape={shape} axes={axes} route={actual} PASS")
        passed += 1
    print(
        f"S03E_STATIC_SUMMARY passed={passed}/{passed} "
        f"host_sha256={sha256(CANDIDATE / 'op_host/square_sum_v1.cpp')} "
        f"kernel_sha256={sha256(CANDIDATE / 'op_kernel/square_sum_v1.cpp')}"
    )


if __name__ == "__main__":
    main()
