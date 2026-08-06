import argparse
import json
import math
from pathlib import Path


DEFAULT_MATRIX = Path(__file__).with_name(
    "squaresum_official_profile_matrix_20260806.json"
)
TRAILING_DIMENSION_LIMITS = (200, 1000, 10000, 10000)


def normalize_axes(rank, axes):
    normalized = []
    for axis in axes:
        value = axis + rank if axis < 0 else axis
        if value < 0 or value >= rank:
            raise ValueError(f"axis {axis} is invalid for rank {rank}")
        if value in normalized:
            raise ValueError(f"axis {axis} is duplicated")
        normalized.append(value)
    if not normalized:
        raise ValueError("the competition matrix must use at least one axis")
    return sorted(normalized)


def validate_shape(shape):
    if not shape or any(not isinstance(value, int) or value < 1 for value in shape):
        raise ValueError(f"shape must contain positive integers: {shape}")

    checked = shape[-4:]
    limits = TRAILING_DIMENSION_LIMITS[-len(checked) :]
    names = ("N4", "N3", "N2", "N")[-len(checked) :]
    for name, value, limit in zip(names, checked, limits):
        if value > limit:
            raise ValueError(
                f"{name}={value} exceeds the specification limit {limit}"
            )


def classify_route(shape, normalized_axes):
    rank = len(shape)
    reduced = [axis in normalized_axes for axis in range(rank)]
    first_reduced = normalized_axes[0]
    last_reduced = normalized_axes[-1]
    reduce_elements = math.prod(shape[axis] for axis in normalized_axes)

    contiguous_group = True
    has_singleton_gap = False
    for axis in range(first_reduced, last_reduced + 1):
        if not reduced[axis] and shape[axis] != 1:
            contiguous_group = False
            break
        if not reduced[axis]:
            has_singleton_gap = True

    if (
        contiguous_group
        and has_singleton_gap
        and last_reduced != rank - 1
        and reduce_elements < 8192
    ):
        contiguous_group = False

    if contiguous_group:
        route = "fast1" if last_reduced == rank - 1 else "fast2"
        inner_elements = math.prod(shape[last_reduced + 1 :])
        if route == "fast2" and inner_elements == 1 and reduce_elements > 0:
            route = "fast1"
        return route
    if reduced[-1]:
        return "fast3"
    return "fast4"


def load_and_validate(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("matrix must contain a non-empty cases list")

    names = set()
    results = []
    for case in cases:
        name = case.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("every case must have a non-empty name")
        if name in names:
            raise ValueError(f"duplicate case name: {name}")
        names.add(name)

        shape = case.get("shape")
        axes = case.get("axes")
        validate_shape(shape)
        normalized_axes = normalize_axes(len(shape), axes)
        actual_route = classify_route(shape, normalized_axes)
        expected_route = case.get("route")
        if expected_route != actual_route:
            raise ValueError(
                f"{name}: expected {expected_route}, current host logic selects "
                f"{actual_route}"
            )
        results.append(
            {
                "name": name,
                "tier": case.get("tier", "unspecified"),
                "route": actual_route,
                "shape": shape,
                "axes": normalized_axes,
                "elements": math.prod(shape),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    args = parser.parse_args()
    results = load_and_validate(args.matrix.resolve())
    for result in results:
        print(
            "MATRIX_CASE "
            f"name={result['name']} tier={result['tier']} "
            f"route={result['route']} shape={result['shape']} "
            f"axes={result['axes']} elements={result['elements']} PASS"
        )
    print(f"MATRIX_SUMMARY passed={len(results)}/{len(results)}")


if __name__ == "__main__":
    main()
