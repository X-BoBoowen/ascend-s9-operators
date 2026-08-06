import numpy as np

from squaresum_s02cr_splitk_semantics_model_20260806 import (
    cast_output,
    official_close,
    strict_squared,
)


SEED = 2026080631
BLOCKS = 40
DTYPES = ("fp16", "bf16", "fp32")


def emulate_short_tail_splitk(source, axes, dtype_name):
    rank = source.ndim
    reduced_axes = sorted({axis % rank for axis in axes})
    kept_axes = [axis for axis in range(rank) if axis not in reduced_axes]
    squared = strict_squared(source, dtype_name)
    matrix = np.transpose(squared, kept_axes + reduced_axes).reshape(
        -1, int(np.prod([source.shape[axis] for axis in reduced_axes]))
    )

    trailing = 1
    for axis in range(rank - 1, -1, -1):
        if axis not in reduced_axes:
            break
        trailing *= source.shape[axis]
    natural_rows = matrix.shape[1] // trailing
    partials = np.zeros((64, matrix.shape[0]), dtype=np.float32)
    base = natural_rows // BLOCKS
    extra = natural_rows % BLOCKS
    for block in range(BLOCKS):
        count = base + int(block < extra)
        first_row = block * base + min(block, extra)
        begin = first_row * trailing
        limit = (first_row + count) * trailing
        partials[block] = np.sum(
            matrix[:, begin:limit], axis=1, dtype=np.float32
        )

    active = 64
    while active > 1:
        half = active // 2
        partials[:half] = np.add(
            partials[:half], partials[half:active], dtype=np.float32
        )
        active = half
    return cast_output(partials[0], dtype_name)


def make_source(rng, shape, distribution):
    if distribution == "uniform":
        return rng.uniform(-0.1, 0.1, shape).astype(np.float32)
    if distribution == "small":
        return rng.uniform(-1e-3, 1e-3, shape).astype(np.float32)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), shape)
    exponents = rng.uniform(-8.0, -2.0, shape).astype(np.float32)
    return signs * np.exp2(exponents).astype(np.float32)


def main():
    rng = np.random.default_rng(SEED)
    layouts = (
        ("tail1023", (40, 8, 1023), (0, 2), "uniform"),
        ("tail512", (64, 8, 512), (0, 2), "small"),
        ("tail200_negative", (200, 8, 200), (0, -1), "log_uniform"),
        ("tail1_rank4", (200, 200, 8, 1), (0, 1, 3), "uniform"),
        ("rank4_sparse", (64, 4, 2, 512), (0, 3), "small"),
    )
    passed = 0
    for name, shape, axes, distribution in layouts:
        source = make_source(rng, shape, distribution)
        normalized = tuple(axis % len(shape) for axis in axes)
        for dtype_name in DTYPES:
            squared = strict_squared(source, dtype_name)
            reference = cast_output(
                np.sum(
                    squared,
                    axis=normalized,
                    dtype=np.float32,
                ).reshape(-1),
                dtype_name,
            )
            candidate = emulate_short_tail_splitk(
                source, axes, dtype_name
            )
            errors, allowed, max_relative = official_close(
                candidate, reference, dtype_name
            )
            assert errors <= allowed, (
                name,
                dtype_name,
                errors,
                allowed,
                max_relative,
            )
            print(
                f"S02CS_SEMANTICS name={name} dtype={dtype_name} "
                f"shape={shape} axes={axes} errors={errors} "
                f"allowed={allowed} max_relative={max_relative:.8f} PASS"
            )
            passed += 1
    print(f"S02CS_SEMANTICS_SUMMARY passed={passed}/{passed}")


if __name__ == "__main__":
    main()
