import numpy as np


SEED = 2026080627
BLOCKS = 40
REDUCE_LENGTHS = (2048, 4097, 32768, 100000)
DTYPES = ("fp16", "bf16", "fp32")
DISTRIBUTIONS = ("uniform", "small", "log_uniform")


def bf16_round(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def strict_squared(source, dtype_name):
    if dtype_name == "fp16":
        values = source.astype(np.float16)
        return np.multiply(values, values, dtype=np.float16).astype(np.float32)
    if dtype_name == "bf16":
        values = bf16_round(source)
        return bf16_round(values * values)
    values = source.astype(np.float32)
    return np.multiply(values, values, dtype=np.float32)


def cast_output(values, dtype_name):
    if dtype_name == "fp16":
        return values.astype(np.float16).astype(np.float32)
    if dtype_name == "bf16":
        return bf16_round(values)
    return values.astype(np.float32)


def splitk_sum(values):
    rows, reduce_length = values.shape
    partials = np.zeros((64, rows), dtype=np.float32)
    base = reduce_length // BLOCKS
    extra = reduce_length % BLOCKS
    for block in range(BLOCKS):
        count = base + int(block < extra)
        start = block * base + min(block, extra)
        partials[block] = np.sum(
            values[:, start : start + count], axis=1, dtype=np.float32
        )
    active = 64
    while active > 1:
        half = active // 2
        partials[:half] = np.add(
            partials[:half], partials[half:active], dtype=np.float32
        )
        active = half
    return partials[0]


def official_close(actual, expected, dtype_name):
    if dtype_name == "fp32":
        rtol = atol = 1e-4
        loss = 1e-4
    else:
        rtol = atol = 1e-2
        loss = 1e-3
    minimum = np.float32(1e-9)
    actual = np.where(actual == 0, minimum, actual)
    expected = np.where(expected == 0, minimum, expected)
    absolute = np.abs(actual - expected)
    relative = absolute / np.maximum(np.abs(actual), np.abs(expected))
    close = (absolute <= atol) | (relative <= rtol)
    close |= np.isnan(actual) & np.isnan(expected)
    errors = int(np.count_nonzero(~close))
    allowed = actual.size * loss
    return errors, allowed, float(np.max(relative, initial=0.0))


def make_source(rng, rows, reduce_length, distribution):
    shape = (rows, reduce_length)
    if distribution == "uniform":
        return rng.uniform(-0.1, 0.1, shape).astype(np.float32)
    if distribution == "small":
        return rng.uniform(-1e-3, 1e-3, shape).astype(np.float32)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), shape)
    exponents = rng.uniform(-8.0, -2.0, shape).astype(np.float32)
    return signs * np.exp2(exponents).astype(np.float32)


def contiguous_strides(shape):
    strides = [0] * len(shape)
    running = 1
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = running
        running *= shape[axis]
    return strides


def flat_offset(index, dims, strides):
    offset = 0
    for axis in range(len(dims) - 1, -1, -1):
        coordinate = index % dims[axis]
        index //= dims[axis]
        offset += coordinate * strides[axis]
    return offset


def emulate_strided_splitk(source, axes, dtype_name):
    shape = source.shape
    normalized_axes = sorted(axis % len(shape) for axis in axes)
    input_strides = contiguous_strides(shape)
    output_dims = [
        shape[axis] for axis in range(len(shape)) if axis not in normalized_axes
    ]
    output_strides = [
        input_strides[axis]
        for axis in range(len(shape))
        if axis not in normalized_axes
    ]
    reduce_dims = [shape[axis] for axis in normalized_axes]
    reduce_strides = [input_strides[axis] for axis in normalized_axes]
    output_elements = int(np.prod(output_dims, dtype=np.int64))
    reduce_elements = int(np.prod(reduce_dims, dtype=np.int64))
    squared = strict_squared(source, dtype_name).reshape(-1)
    partials = np.zeros((64, output_elements), dtype=np.float32)
    base_count = reduce_elements // BLOCKS
    extra = reduce_elements % BLOCKS
    for block in range(BLOCKS):
        count = base_count + int(block < extra)
        start = block * base_count + min(block, extra)
        for output_index in range(output_elements):
            base_offset = flat_offset(
                output_index, output_dims, output_strides
            )
            total = np.float32(0.0)
            for reduce_index in range(start, start + count):
                source_offset = base_offset + flat_offset(
                    reduce_index, reduce_dims, reduce_strides
                )
                total = np.add(total, squared[source_offset], dtype=np.float32)
            partials[block, output_index] = total
    active = 64
    while active > 1:
        half = active // 2
        partials[:half] = np.add(
            partials[:half], partials[half:active], dtype=np.float32
        )
        active = half
    return cast_output(partials[0], dtype_name)


def run_structured_layouts(rng):
    layouts = (
        ("rank5_sparse", (3, 2, 5, 2, 7), (0, 2), False),
        ("unsorted_keepdims", (4, 3, 5, 1, 9), (2, 0), True),
        ("negative_axes", (2, 3, 4, 2, 5), (-5, -3), False),
    )
    passed = 0
    for name, shape, axes, keep_dims in layouts:
        source = rng.uniform(-0.1, 0.1, shape).astype(np.float32)
        for dtype_name in DTYPES:
            squared = strict_squared(source, dtype_name)
            reference = cast_output(
                np.sum(
                    squared,
                    axis=tuple(axis % len(shape) for axis in axes),
                    keepdims=keep_dims,
                    dtype=np.float32,
                ).reshape(-1),
                dtype_name,
            )
            candidate = emulate_strided_splitk(source, axes, dtype_name)
            errors, allowed, max_relative = official_close(
                candidate, reference, dtype_name
            )
            if errors > allowed:
                raise AssertionError(
                    f"layout={name} dtype={dtype_name} errors={errors} "
                    f"allowed={allowed} max_relative={max_relative}"
                )
            print(
                "S02CR_LAYOUT "
                f"name={name} dtype={dtype_name} shape={shape} axes={axes} "
                f"keep_dims={int(keep_dims)} errors={errors} "
                f"allowed={allowed} max_relative={max_relative:.8f} PASS"
            )
            passed += 1
    return passed


def main():
    rng = np.random.default_rng(SEED)
    passed = run_structured_layouts(rng)
    for reduce_length in REDUCE_LENGTHS:
        rows = max(8, min(257, 2_000_000 // reduce_length))
        for distribution in DISTRIBUTIONS:
            source = make_source(rng, rows, reduce_length, distribution)
            for dtype_name in DTYPES:
                squared = strict_squared(source, dtype_name)
                reference = cast_output(
                    np.sum(squared, axis=1, dtype=np.float32), dtype_name
                )
                candidate = cast_output(splitk_sum(squared), dtype_name)
                errors, allowed, max_relative = official_close(
                    candidate, reference, dtype_name
                )
                if errors > allowed:
                    raise AssertionError(
                        f"dtype={dtype_name} reduce={reduce_length} "
                        f"distribution={distribution} errors={errors} "
                        f"allowed={allowed} max_relative={max_relative}"
                    )
                print(
                    "S02CR_SEMANTICS "
                    f"dtype={dtype_name} reduce={reduce_length} rows={rows} "
                    f"distribution={distribution} errors={errors} "
                    f"allowed={allowed} max_relative={max_relative:.8f} PASS"
                )
                passed += 1
    print(f"S02CR_SEMANTICS_SUMMARY passed={passed}/{passed}")


if __name__ == "__main__":
    main()
