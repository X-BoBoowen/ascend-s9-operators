import numpy as np


SEED = 2026080621
REDUCE_LENGTHS = (1, 2, 3, 7, 31, 127, 513, 4096, 10000)
ROWS_BY_REDUCE = {
    1: 65536,
    2: 32768,
    3: 21845,
    7: 9362,
    31: 4096,
    127: 2048,
    513: 1024,
    4096: 256,
    10000: 128,
}


def round_float32_to_bfloat16_as_float32(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    rounded = (bits + rounding_bias) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def official_close(actual, expected):
    minimum = np.float32(1e-9)
    actual = np.where(actual == 0, minimum, actual)
    expected = np.where(expected == 0, minimum, expected)
    absolute = np.abs(actual - expected)
    relative = absolute / np.maximum(np.abs(actual), np.abs(expected))
    close = (absolute <= 1e-2) | (relative <= 1e-2)
    close |= np.isnan(actual) & np.isnan(expected)
    errors = int(np.count_nonzero(~close))
    allowed = actual.size * 1e-3
    return errors, allowed, float(np.max(relative, initial=0.0))


def run_distribution(rng, reduce_length, distribution):
    rows = ROWS_BY_REDUCE[reduce_length]
    if distribution == "uniform":
        source = rng.uniform(-10.0, 10.0, (rows, reduce_length)).astype(
            np.float32
        )
    elif distribution == "small":
        source = rng.uniform(-0.03, 0.03, (rows, reduce_length)).astype(
            np.float32
        )
    else:
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), (rows, reduce_length))
        exponents = rng.uniform(-12.0, 12.0, (rows, reduce_length)).astype(
            np.float32
        )
        source = signs * np.exp2(exponents).astype(np.float32)

    bf16_input = round_float32_to_bfloat16_as_float32(source)
    reference_square = round_float32_to_bfloat16_as_float32(
        bf16_input * bf16_input
    )
    reference = round_float32_to_bfloat16_as_float32(
        np.sum(reference_square, axis=1, dtype=np.float32)
    )
    candidate = round_float32_to_bfloat16_as_float32(
        np.sum(bf16_input * bf16_input, axis=1, dtype=np.float32)
    )
    errors, allowed, max_relative = official_close(candidate, reference)
    if errors > allowed:
        raise AssertionError(
            f"reduce={reduce_length} distribution={distribution}: "
            f"errors={errors} allowed={allowed} max_relative={max_relative}"
        )
    print(
        "S02CQ_SEMANTICS "
        f"reduce={reduce_length} distribution={distribution} rows={rows} "
        f"errors={errors} allowed={allowed} max_relative={max_relative:.8f} PASS"
    )


def main():
    rng = np.random.default_rng(SEED)
    passed = 0
    for reduce_length in REDUCE_LENGTHS:
        for distribution in ("uniform", "small", "log_uniform"):
            run_distribution(rng, reduce_length, distribution)
            passed += 1
    print(f"S02CQ_SEMANTICS_SUMMARY passed={passed}/{passed}")


if __name__ == "__main__":
    main()
