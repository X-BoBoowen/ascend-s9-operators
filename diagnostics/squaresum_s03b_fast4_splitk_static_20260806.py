import hashlib
import math
from pathlib import Path

from validate_squaresum_profile_matrix_20260806 import (
    classify_route,
    normalize_axes,
    validate_shape,
)


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "baselines/squaresum_s02f_global_best_20260806/SquareSumV1"
CANDIDATE = ROOT / "candidates/squaresum_s03b_s02f_fast4_splitk_20260806/SquareSumV1"
BASE_HOST_SHA256 = "6C4731323E66D3E7A02044FA214386D7C10167A0BA145C6AA8D3902535F5F367"
BASE_KERNEL_SHA256 = "B668B6A2217130E1878063713B3D03F663C7626B7C175FB6C8E503119002255A"
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


def selected(info):
    return (
        info["route"] == "fast4"
        and info["input"] >= MIN_INPUT
        and info["reduce"] >= MIN_REDUCE
        and 0 < info["output"] <= MAX_OUTPUTS
        and info["inner"] > 0
        and info["output"] % info["inner"] == 0
    )


def assert_source_contract():
    base_host = BASE / "op_host/square_sum_v1.cpp"
    base_kernel = BASE / "op_kernel/square_sum_v1.cpp"
    host = CANDIDATE / "op_host/square_sum_v1.cpp"
    kernel = CANDIDATE / "op_kernel/square_sum_v1.cpp"
    assert sha256(base_host) == BASE_HOST_SHA256
    assert sha256(base_kernel) == BASE_KERNEL_SHA256
    for relative in (
        Path("op_host/CMakeLists.txt"),
        Path("op_host/square_sum_v1_tiling.h"),
        Path("op_kernel/CMakeLists.txt"),
    ):
        assert (BASE / relative).read_bytes() == (CANDIDATE / relative).read_bytes()

    host_text = host.read_text(encoding="utf-8")
    kernel_text = kernel.read_text(encoding="utf-8")
    assert host_text.count("generalStridedSplitK") == 2
    assert host_text.count("reduceMode = 3U;") == 1
    assert "outputElements <= GENERAL_STRIDED_MAX_OUTPUTS" in host_text
    assert kernel_text.count("ProcessParallelStrided(") == 2
    assert kernel_text.count("reduceMode_ == 3U") == 3
    assert "ReduceInputOffset(flatReduceIndex)" in kernel_text
    assert "rowsUntilBoundary" in kernel_text
    assert "HighestPowerOfTwo(" in kernel_text


def main():
    assert_source_contract()
    positive = (
        ("minimum_gate", (32, 2, 64, 1, 64), (0, 2)),
        ("large_reduce_small_output", (128, 2, 512, 1, 64), (0, 2)),
        ("negative_axes", (64, 2, 512, 2, 64), (-5, -3)),
        ("output_1024", (64, 2, 64, 4, 128), (0, 2)),
    )
    controls = (
        ("contiguous_fast2", (128, 64, 64), (0, 1)),
        ("below_input", (8, 2, 64, 2, 64), (0, 2)),
        ("below_reduce", (8, 4, 128, 4, 64), (0, 2)),
        ("output_1025", (64, 2, 64, 5, 205), (0, 2)),
    )
    passed = 0
    for name, shape, axes in positive:
        info = metadata(shape, axes)
        assert selected(info), (name, info)
        print(f"S03B_ROUTE name={name} selected=1 info={info} PASS")
        passed += 1
    for name, shape, axes in controls:
        info = metadata(shape, axes)
        assert not selected(info), (name, info)
        print(f"S03B_CONTROL name={name} selected=0 info={info} PASS")
        passed += 1
    print(
        f"S03B_STATIC_SUMMARY passed={passed}/{passed} "
        f"candidate_host_sha256={sha256(CANDIDATE / 'op_host/square_sum_v1.cpp')} "
        f"candidate_kernel_sha256={sha256(CANDIDATE / 'op_kernel/square_sum_v1.cpp')}"
    )


if __name__ == "__main__":
    main()
