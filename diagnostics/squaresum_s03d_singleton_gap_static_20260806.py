import hashlib
import itertools
import math
from pathlib import Path

from validate_squaresum_profile_matrix_20260806 import (
    classify_route,
    normalize_axes,
)


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "baselines/squaresum_s02f_global_best_20260806/SquareSumV1"
CANDIDATE = ROOT / "candidates/squaresum_s03d_s02f_singleton_gap_20260806/SquareSumV1"
BASE_HOST_SHA256 = "6C4731323E66D3E7A02044FA214386D7C10167A0BA145C6AA8D3902535F5F367"
BASE_KERNEL_SHA256 = "B668B6A2217130E1878063713B3D03F663C7626B7C175FB6C8E503119002255A"
CANDIDATE_HOST_SHA256 = "C698F38623B8BFB0B7FDBED197B18B84541DA6CE5713F83E9CEA5DA54460F523"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def strides(shape):
    result = [0] * len(shape)
    running = 1
    for axis in range(len(shape) - 1, -1, -1):
        result[axis] = running
        running *= shape[axis]
    return result


def flat_offset(index, dims, physical_strides):
    result = 0
    for axis in range(len(dims) - 1, -1, -1):
        coordinate = index % dims[axis]
        index //= dims[axis]
        result += coordinate * physical_strides[axis]
    return result


def assert_coordinate_equivalence(shape, axes):
    normalized = normalize_axes(len(shape), axes)
    reduced = set(normalized)
    physical = strides(shape)
    output_dims = [shape[i] for i in range(len(shape)) if i not in reduced]
    output_strides = [physical[i] for i in range(len(shape)) if i not in reduced]
    reduce_dims = [shape[i] for i in normalized]
    reduce_strides = [physical[i] for i in normalized]
    outputs = math.prod(output_dims)
    reductions = math.prod(reduce_dims)
    inner = math.prod(shape[normalized[-1] + 1 :])
    route = "fast1" if normalized[-1] == len(shape) - 1 else "fast2"
    output_samples = sorted({0, outputs // 2, outputs - 1})
    reduce_samples = sorted({0, reductions // 2, reductions - 1})
    for output_index, reduce_index in itertools.product(
        output_samples, reduce_samples
    ):
        generic = flat_offset(
            output_index, output_dims, output_strides
        ) + flat_offset(reduce_index, reduce_dims, reduce_strides)
        if route == "fast1":
            contiguous = output_index * reductions + reduce_index
        else:
            outer = output_index // inner
            inner_index = output_index % inner
            contiguous = (
                (outer * reductions + reduce_index) * inner + inner_index
            )
        assert generic == contiguous, (
            shape,
            axes,
            output_index,
            reduce_index,
            generic,
            contiguous,
        )


def assert_source_contract():
    base_host = BASE / "op_host/square_sum_v1.cpp"
    base_kernel = BASE / "op_kernel/square_sum_v1.cpp"
    candidate_host = CANDIDATE / "op_host/square_sum_v1.cpp"
    assert sha256(base_host) == BASE_HOST_SHA256
    assert sha256(base_kernel) == BASE_KERNEL_SHA256
    assert sha256(candidate_host) == CANDIDATE_HOST_SHA256
    assert (CANDIDATE / "op_kernel/square_sum_v1.cpp").read_bytes() == base_kernel.read_bytes()
    for relative in (
        Path("op_host/CMakeLists.txt"),
        Path("op_host/square_sum_v1_tiling.h"),
        Path("op_kernel/CMakeLists.txt"),
    ):
        assert (BASE / relative).read_bytes() == (CANDIDATE / relative).read_bytes()
    text = candidate_host.read_text(encoding="utf-8")
    assert text.count("hasSingletonGap") == 3
    assert text.count("reduceElements < 8192U") == 1
    assert "inputDims[axis] != 1U" in text


def main():
    assert_source_contract()
    route_cases = (
        ("selected_fast1", (128, 1, 64), (0, 2), "fast1"),
        ("selected_fast2", (128, 1, 64, 64), (2, 0), "fast2"),
        ("selected_outer", (3, 128, 1, 64, 16), (1, 3), "fast2"),
        ("control_below_threshold", (64, 1, 64, 64), (0, 2), "fast4"),
        ("control_real_gap", (128, 2, 64, 64), (0, 2), "fast4"),
    )
    passed = 0
    for name, shape, axes, expected in route_cases:
        actual = classify_route(shape, normalize_axes(len(shape), axes))
        assert actual == expected, (name, actual, expected)
        print(f"S03D_ROUTE name={name} route={actual} PASS")
        passed += 1

    coordinate_cases = (
        ((2, 1, 3), (0, 2)),
        ((2, 1, 3, 4), (0, 2)),
        ((5, 2, 1, 3, 4), (1, 3)),
        ((5, 2, 1, 3), (1, 3)),
    )
    for shape, axes in coordinate_cases:
        assert_coordinate_equivalence(shape, axes)
        print(f"S03D_COORD shape={shape} axes={axes} PASS")
        passed += 1
    print(
        f"S03D_STATIC_SUMMARY passed={passed}/{passed} "
        f"candidate_host_sha256={sha256(CANDIDATE / 'op_host/square_sum_v1.cpp')}"
    )


if __name__ == "__main__":
    main()
