from itertools import combinations, product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02ak_grouped_short_narrow_20260802"
    / "SquareSumV1"
)
HOST = CANDIDATE / "op_host" / "square_sum_v1.cpp"
CHUNK = 8192


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def strides(dims):
    result = [0] * len(dims)
    running = 1
    for axis in range(len(dims) - 1, -1, -1):
        result[axis] = running
        running *= dims[axis]
    return result


def metadata(dims, reduced_axes, keep_dims):
    input_strides = strides(dims)
    reduced = set(reduced_axes)
    output_dims = []
    output_strides = []
    reduce_dims = []
    reduce_strides = []
    for axis, extent in enumerate(dims):
        if axis in reduced:
            reduce_dims.append(extent)
            reduce_strides.append(input_strides[axis])
            if keep_dims:
                output_dims.append(1)
                output_strides.append(0)
        else:
            output_dims.append(extent)
            output_strides.append(input_strides[axis])

    first_reduced = min(reduced)
    last_reduced = max(reduced)
    contiguous = all(
        axis in reduced
        for axis in range(first_reduced, last_reduced + 1)
    )
    if contiguous:
        fast_path = 1 if last_reduced == len(dims) - 1 else 2
    elif len(dims) - 1 in reduced:
        fast_path = 3
    else:
        fast_path = 4

    trailing = 1
    if fast_path == 3:
        for axis in range(len(dims) - 1, -1, -1):
            if axis not in reduced:
                break
            trailing *= dims[axis]
    return {
        "input_strides": input_strides,
        "output_dims": output_dims,
        "output_strides": output_strides,
        "reduce_dims": reduce_dims,
        "reduce_strides": reduce_strides,
        "fast_path": fast_path,
        "trailing": trailing,
        "output_elements": product_value(output_dims),
        "reduce_elements": product_value(reduce_dims),
    }


def product_value(values):
    result = 1
    for value in values:
        result *= value
    return result


def base_input_offset(output_index, output_dims, output_strides):
    result = 0
    for axis in range(len(output_dims) - 1, -1, -1):
        coordinate = output_index % output_dims[axis]
        output_index //= output_dims[axis]
        result += coordinate * output_strides[axis]
    return result


def reduce_input_offset(reduce_index, reduce_dims, reduce_strides):
    result = 0
    for axis in range(len(reduce_dims) - 1, -1, -1):
        coordinate = reduce_index % reduce_dims[axis]
        reduce_index //= reduce_dims[axis]
        result += coordinate * reduce_strides[axis]
    return result


def select_short(meta, element_bytes):
    if meta["fast_path"] != 3:
        return 0, None
    expected_stride = 1
    first_trailing_axis = len(meta["reduce_dims"])
    for axis in range(len(meta["reduce_dims"]) - 1, -1, -1):
        if meta["reduce_strides"][axis] != expected_stride:
            break
        expected_stride *= meta["reduce_dims"][axis]
        first_trailing_axis = axis
    if (
        first_trailing_axis == 0
        or first_trailing_axis >= len(meta["reduce_dims"])
        or expected_stride != meta["trailing"]
    ):
        return 0, None

    physical_output_axis = len(meta["output_dims"])
    for axis in range(len(meta["output_dims"]) - 1, -1, -1):
        if meta["output_strides"][axis] != 0:
            physical_output_axis = axis
            break
    if physical_output_axis == len(meta["output_dims"]):
        return 0, None
    output_dim = meta["output_dims"][physical_output_axis]
    trailing = meta["trailing"]
    if not (0 < output_dim < 8 and 0 < trailing <= 64):
        return 0, None
    if meta["output_strides"][physical_output_axis] != trailing:
        return 0, None

    batch_axis = first_trailing_axis - 1
    batch_dim = meta["reduce_dims"][batch_axis]
    elements_per_block = 32 // element_bytes
    padded_tail = ceil_div(trailing, elements_per_block) * elements_per_block
    legacy_rows = CHUNK // padded_tail
    for width in (4, 2, 1):
        if meta["output_elements"] % width or output_dim % width:
            continue
        vector_elements = trailing * width
        if meta["reduce_strides"][batch_axis] < vector_elements:
            continue
        source_gap_bytes = (
            meta["reduce_strides"][batch_axis] - vector_elements
        ) * element_bytes
        if source_gap_bytes > 0xFFFFFFFF:
            continue
        aligned_vector = ceil_div(
            vector_elements, elements_per_block
        ) * elements_per_block
        padded_vector = max(64, aligned_vector)
        if trailing % elements_per_block == 0:
            vector_rows = min(31, CHUNK // vector_elements)
        else:
            vector_rows = CHUNK // padded_vector
        if vector_rows == 0 or legacy_rows == 0:
            continue
        new_dma = ceil_div(batch_dim, vector_rows)
        old_dma = width * ceil_div(batch_dim, legacy_rows)
        if new_dma <= old_dma:
            return width, (first_trailing_axis, batch_axis)
    return 0, None


def verify_selected_layout(meta, width, axes):
    first_trailing_axis, batch_axis = axes
    trailing = meta["trailing"]
    output_elements = meta["output_elements"]
    assert output_elements % width == 0
    task_count = output_elements // width
    tasks = {0, task_count - 1, task_count // 2}
    for task in tasks:
        output_start = task * width
        base = base_input_offset(
            output_start,
            meta["output_dims"],
            meta["output_strides"],
        )
        for lane in range(width):
            lane_base = base_input_offset(
                output_start + lane,
                meta["output_dims"],
                meta["output_strides"],
            )
            assert lane_base == base + lane * trailing

    batch_dim = meta["reduce_dims"][batch_axis]
    outer_groups = meta["reduce_elements"] // (
        batch_dim * trailing
    )
    assert outer_groups > 0
    sample_groups = {0, outer_groups - 1}
    sample_batches = {0, batch_dim - 1}
    for group in sample_groups:
        group_start = group * batch_dim * trailing
        for batch in sample_batches:
            reduce_index = group_start + batch * trailing
            offset = reduce_input_offset(
                reduce_index,
                meta["reduce_dims"],
                meta["reduce_strides"],
            )
            if batch + 1 < batch_dim:
                next_offset = reduce_input_offset(
                    reduce_index + trailing,
                    meta["reduce_dims"],
                    meta["reduce_strides"],
                )
                assert (
                    next_offset - offset
                    == meta["reduce_strides"][batch_axis]
                )
    assert first_trailing_axis > 0


def audit_source():
    source = HOST.read_text(encoding="utf-8")
    assert source.count("{") == source.count("}")
    assert source.count("(") == source.count(")")
    assert "groupedShortVectorWidth" in source
    assert "groupedFixedVectorWidth" in source


def main():
    audit_source()
    dims_values = (1, 2, 4, 7, 15, 16)
    configurations = 0
    fast3 = 0
    selected = {1: 0, 2: 0, 4: 0}
    keep_dims_selected = 0
    singleton_selected = 0
    for rank in range(2, 6):
        axis_sets = tuple(
            axes
            for count in range(1, rank + 1)
            for axes in combinations(range(rank), count)
        )
        for dims in product(dims_values, repeat=rank):
            for reduced_axes in axis_sets:
                for keep_dims in (False, True):
                    configurations += 1
                    meta = metadata(dims, reduced_axes, keep_dims)
                    if meta["fast_path"] == 3:
                        fast3 += 1
                    for element_bytes in (2, 4):
                        width, selected_axes = select_short(
                            meta, element_bytes
                        )
                        if width == 0:
                            continue
                        selected[width] += 1
                        if keep_dims:
                            keep_dims_selected += 1
                        if 1 in dims:
                            singleton_selected += 1
                        verify_selected_layout(
                            meta, width, selected_axes
                        )
    assert fast3 > 0
    assert all(value > 0 for value in selected.values())
    assert keep_dims_selected > 0
    assert singleton_selected > 0
    print(f"HOST={HOST}")
    print(f"CONFIGURATIONS={configurations}")
    print(f"FAST3_CONFIGURATIONS={fast3}")
    print(f"SELECTED_WIDTH4={selected[4]}")
    print(f"SELECTED_WIDTH2={selected[2]}")
    print(f"SELECTED_WIDTH1={selected[1]}")
    print(f"KEEP_DIMS_SELECTED={keep_dims_selected}")
    print(f"SINGLETON_SELECTED={singleton_selected}")
    print("SUMMARY: S02AK rank/axis routing audit passed")


if __name__ == "__main__":
    main()
