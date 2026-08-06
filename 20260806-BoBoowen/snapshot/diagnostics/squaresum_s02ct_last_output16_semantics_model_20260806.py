import numpy as np

from squaresum_s02cr_splitk_semantics_model_20260806 import (
    cast_output,
    official_close,
    splitk_sum,
    strict_squared,
)


SEED = 2026080633
DTYPES = ("fp16", "bf16", "fp32")


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
    cases = (
        ("output9", (9, 64, 512), (1, 2), "uniform"),
        ("output10", (10, 64, 512), (-2, -1), "small"),
        ("output15", (15, 64, 512), (1, 2), "log_uniform"),
        ("output16", (16, 64, 512), (1, 2), "uniform"),
        ("rank4_output16", (4, 4, 1000, 32), (2, 3), "small"),
    )
    passed = 0
    for name, shape, axes, distribution in cases:
        source = make_source(rng, shape, distribution)
        normalized = tuple(axis % len(shape) for axis in axes)
        output_elements = int(
            np.prod(shape[: min(normalized)], dtype=np.int64)
        )
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
            matrix = squared.reshape(output_elements, -1)
            candidate = cast_output(splitk_sum(matrix), dtype_name)
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
                f"S02CT_SEMANTICS name={name} dtype={dtype_name} "
                f"shape={shape} axes={axes} errors={errors} "
                f"allowed={allowed} max_relative={max_relative:.8f} PASS"
            )
            passed += 1
    print(f"S02CT_SEMANTICS_SUMMARY passed={passed}/{passed}")


if __name__ == "__main__":
    main()
