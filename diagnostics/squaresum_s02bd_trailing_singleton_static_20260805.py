import hashlib
import itertools
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BB = ROOT / (
    "candidates/squaresum_s02bb_compact_middle8_20260805/"
    "SquareSumV1"
)
BD = ROOT / (
    "candidates/squaresum_s02bd_trailing_singleton_last_20260805/"
    "SquareSumV1"
)
HOST = BD / "op_host/square_sum_v1.cpp"
KERNEL = BD / "op_kernel/square_sum_v1.cpp"
MAX_RANK = 5


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def strides(shape):
    result = [0] * len(shape)
    running = 1
    for axis in range(len(shape) - 1, -1, -1):
        result[axis] = running
        running *= shape[axis]
    return tuple(result)


def normalize_axes(rank, raw_axes):
    if not raw_axes:
        return tuple(range(rank))
    normalized = set()
    for raw_axis in raw_axes:
        axis = raw_axis + rank if raw_axis < 0 else raw_axis
        if axis < 0 or axis >= rank:
            raise ValueError((rank, raw_axes))
        normalized.add(axis)
    return tuple(sorted(normalized))


def metadata(shape, raw_axes, promote, keep_dims=False):
    rank = len(shape)
    axes = normalize_axes(rank, raw_axes)
    reduced = tuple(axis in axes for axis in range(rank))
    input_strides = strides(shape)
    input_elements = product(shape)
    reduce_elements = product(shape[axis] for axis in axes)
    output_axes = tuple(axis for axis in range(rank) if not reduced[axis])
    output_elements = product(shape[axis] for axis in output_axes)
    output_dims = []
    output_strides = []
    for axis in range(rank):
        if reduced[axis]:
            if keep_dims:
                output_dims.append(1)
                output_strides.append(0)
        else:
            output_dims.append(shape[axis])
            output_strides.append(input_strides[axis])
    reduce_dims = tuple(shape[axis] for axis in axes)
    reduce_strides = tuple(input_strides[axis] for axis in axes)

    first_reduced = axes[0]
    last_reduced = axes[-1]
    contiguous_group = all(
        reduced[axis] or shape[axis] == 1
        for axis in range(first_reduced, last_reduced + 1)
    )
    has_singleton_gap = any(
        not reduced[axis]
        for axis in range(first_reduced, last_reduced + 1)
    )
    if (
        contiguous_group
        and has_singleton_gap
        and last_reduced != rank - 1
        and reduce_elements < 8192
    ):
        contiguous_group = False

    inner_elements = 1
    if contiguous_group:
        fast_path = 1 if last_reduced == rank - 1 else 2
        inner_elements = product(shape[last_reduced + 1 :])
        if (
            promote
            and fast_path == 2
            and inner_elements == 1
            and reduce_elements > 0
        ):
            fast_path = 1
    elif reduced[-1]:
        fast_path = 3
    else:
        fast_path = 4
        trailing = 0
        while trailing < rank and not reduced[rank - 1 - trailing]:
            trailing += 1
        inner_elements = product(shape[rank - trailing :])

    return {
        "axes": axes,
        "fast_path": fast_path,
        "inner_elements": inner_elements,
        "input_elements": input_elements,
        "output_elements": output_elements,
        "output_dims": tuple(output_dims),
        "output_strides": tuple(output_strides),
        "reduce_elements": reduce_elements,
        "output_axes": output_axes,
        "input_strides": input_strides,
        "reduce_dims": reduce_dims,
        "reduce_strides": reduce_strides,
    }


def coordinates(dims):
    if not dims:
        yield ()
        return
    yield from itertools.product(*(range(dim) for dim in dims))


def base_offsets(shape, info):
    for coordinate in coordinates(info["output_dims"]):
        yield sum(
            value * stride
            for value, stride in zip(coordinate, info["output_strides"])
        )


def reduce_offsets(info):
    for coordinate in coordinates(info["reduce_dims"]):
        yield sum(
            value * stride
            for value, stride in zip(coordinate, info["reduce_strides"])
        )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def source_guards():
    host = HOST.read_text(encoding="utf-8")
    bb_host = (BB / "op_host/square_sum_v1.cpp").read_text(
        encoding="utf-8"
    )
    guard = """if (fastPath == 2U &&
                innerElements == 1U &&
                reduceElements > 0U) {
                fastPath = 1U;
            }"""
    assert host.count(guard) == 1
    insertion_point = """            }
        } else if (reduced[rank - 1]) {"""
    expected_host = bb_host.replace(
        insertion_point,
        f"            }}\n            {guard}\n"
        "        } else if (reduced[rank - 1]) {",
        1,
    )
    assert host == expected_host
    bb_files = {
        path.relative_to(BB) for path in BB.rglob("*") if path.is_file()
    }
    bd_files = {
        path.relative_to(BD) for path in BD.rglob("*") if path.is_file()
    }
    assert bb_files == bd_files
    for relative in bb_files - {Path("op_host/square_sum_v1.cpp")}:
        assert sha256(BB / relative) == sha256(BD / relative)
    assert sha256(KERNEL) == sha256(BB / "op_kernel/square_sum_v1.cpp")


def exhaustive_address_audit():
    layouts = 0
    promoted = 0
    promoted_keep_dims = 0
    zero_reduce_retained = 0
    output_histogram = {}
    extents = (0, 1, 2, 3)

    for rank in range(1, MAX_RANK + 1):
        for shape in itertools.product(extents, repeat=rank):
            for mask in range(1, 1 << rank):
                axes = tuple(
                    axis for axis in range(rank) if mask & (1 << axis)
                )
                for keep_dims in (False, True):
                    before = metadata(shape, axes, False, keep_dims)
                    after = metadata(shape, axes, True, keep_dims)
                    layouts += 1
                    if before["fast_path"] == after["fast_path"]:
                        if (
                            not keep_dims
                            and before["fast_path"] == 2
                            and before["inner_elements"] == 1
                            and before["reduce_elements"] == 0
                        ):
                            zero_reduce_retained += 1
                        continue

                    promoted += 1
                    promoted_keep_dims += int(keep_dims)
                    assert before["fast_path"] == 2
                    assert after["fast_path"] == 1
                    assert after["inner_elements"] == 1
                    assert after["reduce_elements"] > 0
                    output_histogram[after["output_elements"]] = (
                        output_histogram.get(after["output_elements"], 0) + 1
                    )

                    reduced = set(after["axes"])
                    last_reduced = after["axes"][-1]
                    assert last_reduced < rank - 1
                    assert product(shape[last_reduced + 1 :]) == 1
                    first_reduced = after["axes"][0]
                    assert all(
                        axis in reduced or shape[axis] == 1
                        for axis in range(first_reduced, last_reduced + 1)
                    )

                    relative = tuple(reduce_offsets(after))
                    expected_relative = tuple(range(after["reduce_elements"]))
                    assert relative == expected_relative
                    visited = []
                    for base in base_offsets(shape, after):
                        segment = tuple(base + offset for offset in relative)
                        assert segment == tuple(
                            range(base, base + after["reduce_elements"])
                        )
                        visited.extend(segment)
                    assert len(visited) == after["input_elements"]
                    assert visited == list(range(after["input_elements"]))
                    assert sorted(visited) == list(
                        range(after["input_elements"])
                    )

    assert promoted > 0
    assert zero_reduce_retained > 0
    return {
        "layouts": layouts,
        "promoted": promoted,
        "promoted_keep_dims": promoted_keep_dims,
        "zero_reduce_retained": zero_reduce_retained,
        "output_histogram": output_histogram,
    }


def axis_form_audit():
    cases = (
        ((3, 1, 5, 1), (0, 2), (-4, -2, -2, 0)),
        ((2, 3, 1, 7, 1), (1, 3), (3, -4, 1)),
        ((17, 1, 1), (0,), (-3, 0, 0)),
    )
    checked = 0
    for shape, canonical, alternate in cases:
        left = metadata(shape, canonical, True)
        right = metadata(shape, alternate, True)
        assert left == right
        checked += 1
    return checked


def large_layout_audit():
    cases = (
        ((1_000_000, 1), (0,), 1),
        ((2, 262_145, 1), (1,), 1),
        ((2, 3, 32_769, 1), (2,), 1),
        ((3, 131, 1, 251, 1), (1, 3), 1),
        ((262_145, 1, 1), (0,), 1),
        ((131_072, 2), (0,), 2),
        ((65_536, 4), (0,), 2),
        ((0, 1), (0,), 2),
    )
    promoted = 0
    controls = 0
    for shape, axes, expected in cases:
        before = metadata(shape, axes, False)
        after = metadata(shape, axes, True)
        assert after["fast_path"] == expected
        if before["fast_path"] != after["fast_path"]:
            promoted += 1
        else:
            controls += 1
    assert promoted == 5
    assert controls == 3
    return len(cases), promoted, controls


def main():
    source_guards()
    result = exhaustive_address_audit()
    axis_forms = axis_form_audit()
    large, large_promoted, large_controls = large_layout_audit()
    print(f"SOURCE_GUARDS=PASS")
    print(f"LAYOUTS={result['layouts']}")
    print(f"PROMOTED={result['promoted']}")
    print(f"PROMOTED_KEEP_DIMS_EQUIVALENTS={result['promoted_keep_dims']}")
    print(f"ZERO_REDUCE_RETAINED={result['zero_reduce_retained']}")
    print(
        "PROMOTED_OUTPUT_CARDINALITIES="
        f"{len(result['output_histogram'])}"
    )
    print(f"AXIS_FORM_CASES={axis_forms}")
    print(f"LARGE_CASES={large}")
    print(f"LARGE_PROMOTED={large_promoted}")
    print(f"LARGE_CONTROLS={large_controls}")
    print("SUMMARY: S02BD trailing-singleton static audit passed")


if __name__ == "__main__":
    main()
