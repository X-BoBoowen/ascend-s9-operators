from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02ad_fp32_tree_capacity_20260802"
    / "SquareSumV1"
)
KERNEL = CANDIDATE / "op_kernel" / "square_sum_v1.cpp"
UB_BYTES = 192 * 1024
BLOCKS = 40
REDUCTION_ROWS = 64
TILE_OUTPUTS = 1024
LONG_CHUNK = 16384
SAFE_TILE_OUTPUTS = 232
SAFE_FLOAT_ELEMENTS = REDUCTION_ROWS * SAFE_TILE_OUTPUTS


def audit_source():
    source = KERNEL.read_text(encoding="utf-8")
    assert source.count("{") == source.count("}")
    assert "constexpr uint32_t FP32_LONG_TREE_OUTPUTS = 232;" in source
    assert (
        "constexpr uint32_t FP32_LONG_TREE_FLOAT_ELEMENTS =\n"
        "    64U * FP32_LONG_TREE_OUTPUTS;"
        in source
    )
    assert (
        "? (TREE_FINALIZE\n"
        "                    ? FP32_LONG_TREE_FLOAT_ELEMENTS *\n"
        "                        sizeof(float)\n"
        "                    : 32U)"
        in source
    )
    assert (
        "FP32_LONG_TREE_FLOAT_ELEMENTS /\n"
        "                reductionRows"
        in source
    )


def ub_audit():
    output_buffer = TILE_OUTPUTS * 4
    input_buffer = LONG_CHUNK * 4
    float_buffer = SAFE_FLOAT_ELEMENTS * 4
    reduce_buffer = LONG_CHUNK * 4
    sum_buffer = 32
    total = (
        output_buffer
        + input_buffer
        + float_buffer
        + reduce_buffer
        + sum_buffer
    )
    assert total <= UB_BYTES
    assert SAFE_TILE_OUTPUTS % 8 == 0
    assert SAFE_FLOAT_ELEMENTS == 14848

    old_float_buffer = 32
    old_total = (
        output_buffer
        + input_buffer
        + old_float_buffer
        + reduce_buffer
        + sum_buffer
    )
    old_access_bytes = REDUCTION_ROWS * 256 * 4
    assert old_access_bytes > old_float_buffer
    return total, UB_BYTES - total, old_total, old_access_bytes


def tree_reduce(values):
    active_rows = values.shape[0]
    while active_rows > 1:
        half_rows = active_rows // 2
        values[:half_rows] = np.add(
            values[:half_rows],
            values[half_rows:active_rows],
            dtype=np.float32,
        )
        active_rows = half_rows
    return values[0].copy()


def tiled_finalize(partials, tile_outputs):
    output_elements = partials.shape[1]
    partial_stride = ((output_elements + 7) // 8) * 8
    output = np.empty(output_elements, dtype=np.float32)
    dmas = 0
    for start in range(0, output_elements, tile_outputs):
        current = min(tile_outputs, output_elements - start)
        padded = ((current + 7) // 8) * 8
        assert REDUCTION_ROWS * padded <= SAFE_FLOAT_ELEMENTS
        assert partial_stride >= current
        source_stride_bytes = (partial_stride - current) * 4
        assert 0 <= source_stride_bytes <= 0xFFFFFFFF
        local = np.zeros((REDUCTION_ROWS, padded), dtype=np.float32)
        local[:BLOCKS, :current] = partials[:, start : start + current]
        output[start : start + current] = tree_reduce(local)[:current]
        dmas += 1
    return output, dmas


def numeric_audit():
    generator = np.random.default_rng(2026080205)
    samples = 0
    max_relative_error = 0.0
    max_absolute_error = 0.0
    max_dma_ratio = 0.0
    for output_elements in (
        1,
        7,
        8,
        231,
        232,
        233,
        463,
        464,
        465,
        1023,
        1024,
    ):
        for scale in (1e-4, 1.0, 1e4):
            for _ in range(20):
                partials = generator.random(
                    (BLOCKS, output_elements), dtype=np.float32
                ) * np.float32(scale)
                result, safe_dmas = tiled_finalize(
                    partials, SAFE_TILE_OUTPUTS
                )
                reference = partials.astype(np.float64).sum(axis=0)
                absolute = np.abs(result.astype(np.float64) - reference)
                relative = absolute / np.maximum(np.abs(reference), 1e-30)
                max_absolute_error = max(
                    max_absolute_error, float(absolute.max())
                )
                max_relative_error = max(
                    max_relative_error, float(relative.max())
                )
                np.testing.assert_allclose(
                    result, reference, rtol=2e-6, atol=2e-6
                )
                old_dmas = (output_elements + 255) // 256
                max_dma_ratio = max(
                    max_dma_ratio, safe_dmas / old_dmas
                )
                samples += 1
    assert max_dma_ratio == 2.0
    return samples, max_relative_error, max_absolute_error, max_dma_ratio


def main():
    audit_source()
    total, headroom, old_total, old_access_bytes = ub_audit()
    samples, max_relative, max_absolute, max_dma_ratio = numeric_audit()
    print(f"KERNEL={KERNEL}")
    print(f"UB_BUDGET_BYTES={UB_BYTES}")
    print(f"NEW_TOTAL_UB_BYTES={total}")
    print(f"NEW_UB_HEADROOM_BYTES={headroom}")
    print(f"OLD_ALLOCATED_TOTAL_UB_BYTES={old_total}")
    print(f"OLD_FLOAT_BUFFER_BYTES=32")
    print(f"OLD_MAX_FLOAT_ACCESS_BYTES={old_access_bytes}")
    print(f"SAFE_TILE_OUTPUTS={SAFE_TILE_OUTPUTS}")
    print(f"NUMERIC_SAMPLES={samples}")
    print(f"MAX_RELATIVE_ERROR={max_relative:.9e}")
    print(f"MAX_ABSOLUTE_ERROR={max_absolute:.9e}")
    print(f"MAX_DMA_RATIO_VS_UNSAFE_256={max_dma_ratio:.6f}")
    print("SUMMARY: S02AD FP32 tree UB-capacity audit passed")


if __name__ == "__main__":
    main()
