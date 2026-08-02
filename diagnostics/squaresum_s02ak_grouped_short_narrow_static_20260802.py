from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02ak_grouped_short_narrow_20260802"
    / "SquareSumV1"
)
HOST = CANDIDATE / "op_host" / "square_sum_v1.cpp"
KERNEL = CANDIDATE / "op_kernel" / "square_sum_v1.cpp"
CHUNK = 8192


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def select_width(element_bytes, output_dim, output_elements, trailing, batch):
    elements_per_block = 32 // element_bytes
    padded_tail = ceil_div(trailing, elements_per_block) * elements_per_block
    legacy_rows = CHUNK // padded_tail
    for width in (4, 2, 1):
        if output_elements % width or output_dim % width:
            continue
        vector_elements = width * trailing
        aligned_vector = ceil_div(vector_elements, elements_per_block) * elements_per_block
        padded_vector = max(64, aligned_vector)
        if trailing % elements_per_block == 0:
            vector_rows = min(31, CHUNK // vector_elements)
        else:
            vector_rows = CHUNK // padded_vector
        if vector_rows == 0 or legacy_rows == 0:
            continue
        new_dma = ceil_div(batch, vector_rows)
        old_dma = width * ceil_div(batch, legacy_rows)
        if new_dma <= old_dma:
            return width, old_dma, new_dma
    return 0, 0, 0


def audit_source():
    host = HOST.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")
    assert host.count("{") == host.count("}")
    assert host.count("(") == host.count(")")
    assert kernel.count("{") == kernel.count("}")
    assert "uint32_t groupedShortVectorWidth = 0U;" in host
    assert "trailingReduceElements <= 64U" in host
    assert "vectorDma <= legacyDma" in host
    assert "groupedFixedVectorWidth" in host
    start = kernel.index(
        "__aicore__ inline void ProcessGroupedSuffixVector8()"
    )
    end = kernel.index(
        "__aicore__ inline void ProcessGroupedSuffix()", start
    )
    body = kernel[start:end]
    assert (
        "constexpr uint32_t VECTOR_OUTPUTS =\n"
        "            GROUPED_VECTOR_WIDTH;"
        in body
    )
    assert "constexpr uint32_t VECTOR_OUTPUTS = 8U;" not in body
    assert "ReduceRowsInto(" in body


def exhaustive_audit():
    configs = 0
    selected = {1: 0, 2: 0, 4: 0}
    fallback = 0
    old_dma_total = 0
    new_dma_total = 0
    removed_scalar_syncs = 0
    for element_bytes in (2, 4):
        for output_dim in range(1, 8):
            for output_outer in (1, 2, 7, 31, 127):
                output_elements = output_dim * output_outer
                for trailing in range(1, 65):
                    for batch in (1, 2, 7, 31, 32, 63, 127, 255, 1024):
                        configs += 1
                        width, old_dma, new_dma = select_width(
                            element_bytes,
                            output_dim,
                            output_elements,
                            trailing,
                            batch,
                        )
                        if width == 0:
                            fallback += 1
                            continue
                        selected[width] += 1
                        tasks = output_elements // width
                        old_total = tasks * old_dma
                        new_total = tasks * new_dma
                        assert new_total <= old_total
                        old_dma_total += old_total
                        new_dma_total += new_total
                        removed_scalar_syncs += old_total
    assert sum(selected.values()) > 0
    assert new_dma_total <= old_dma_total
    return (
        configs,
        selected,
        fallback,
        old_dma_total,
        new_dma_total,
        removed_scalar_syncs,
    )


def numeric_audit():
    generator = np.random.default_rng(2026080212)
    samples = 0
    max_relative_error = 0.0
    for output_dim in range(1, 8):
        for trailing in (1, 7, 8, 15, 16, 31, 32, 63, 64):
            for batch in (1, 3, 31, 127):
                width, _, _ = select_width(
                    4, output_dim, output_dim, trailing, batch
                )
                if width == 0:
                    continue
                values = generator.uniform(
                    -2.0, 2.0, size=(batch, output_dim, trailing)
                ).astype(np.float32)
                reference = np.square(values, dtype=np.float32).astype(
                    np.float64
                ).sum(axis=(0, 2))
                result = np.empty(output_dim, dtype=np.float32)
                for start in range(0, output_dim, width):
                    partial = np.square(
                        values[:, start : start + width, :],
                        dtype=np.float32,
                    ).sum(axis=2, dtype=np.float32)
                    result[start : start + width] = partial.sum(
                        axis=0, dtype=np.float32
                    )
                absolute = np.abs(result.astype(np.float64) - reference)
                relative = absolute / np.maximum(np.abs(reference), 1e-30)
                max_relative_error = max(
                    max_relative_error, float(relative.max())
                )
                np.testing.assert_allclose(
                    result, reference, rtol=2e-5, atol=2e-5
                )
                samples += 1
    return samples, max_relative_error


def main():
    audit_source()
    (
        configs,
        selected,
        fallback,
        old_dma,
        new_dma,
        removed_syncs,
    ) = exhaustive_audit()
    samples, max_relative = numeric_audit()
    print(f"HOST={HOST}")
    print(f"KERNEL={KERNEL}")
    print(f"CONFIGS={configs}")
    print(f"SELECTED_WIDTH4={selected[4]}")
    print(f"SELECTED_WIDTH2={selected[2]}")
    print(f"SELECTED_WIDTH1={selected[1]}")
    print(f"FALLBACK={fallback}")
    print(f"OLD_SELECTED_DMA={old_dma}")
    print(f"NEW_SELECTED_DMA={new_dma}")
    print(f"MAX_DMA_RATIO={new_dma / old_dma:.6f}")
    print(f"REMOVED_SCALAR_SYNCS={removed_syncs}")
    print(f"NUMERIC_SAMPLES={samples}")
    print(f"MAX_RELATIVE_ERROR={max_relative:.9e}")
    print("SUMMARY: S02AK grouped short-narrow model passed")


if __name__ == "__main__":
    main()
