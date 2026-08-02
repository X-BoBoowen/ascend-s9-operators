from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02aj_sequential_first_block_direct_20260802"
    / "SquareSumV1"
)
KERNEL = CANDIDATE / "op_kernel" / "square_sum_v1.cpp"
BLOCKS = 40


def audit_source():
    source = KERNEL.read_text(encoding="utf-8")
    assert source.count("{") == source.count("}")
    start = source.index(
        "__aicore__ inline void FinalizeParallelReductionSequential()"
    )
    end = source.index(
        "__aicore__ inline void FinalizeParallelReductionTree()", start
    )
    body = source[start:end]
    assert "Duplicate(\n                accumulateLocal" not in body
    assert "if (block == 0U)" in body
    assert "AscendC::Adds(" in body
    assert "AscendC::Add(" in body


def numeric_audit():
    generator = np.random.default_rng(2026080211)
    samples = 0
    for outputs in (1, 7, 8, 127, 1024):
        for scale in (1e-4, 1.0, 1e4):
            for _ in range(50):
                partials = generator.random(
                    (BLOCKS, outputs), dtype=np.float32
                ) * np.float32(scale)
                old = np.zeros(outputs, dtype=np.float32)
                for partial in partials:
                    old = np.add(old, partial, dtype=np.float32)
                new = partials[0].copy()
                for partial in partials[1:]:
                    new = np.add(new, partial, dtype=np.float32)
                np.testing.assert_array_equal(new, old)
                samples += 1
    return samples


def main():
    audit_source()
    final_tiles = sum((outputs + 1023) // 1024 for outputs in range(1, 8193))
    samples = numeric_audit()
    print(f"KERNEL={KERNEL}")
    print(f"FINAL_TILES={final_tiles}")
    print(f"REMOVED_ZERO_FILLS={final_tiles}")
    print(f"NUMERIC_BITWISE_SAMPLES={samples}")
    print("SUMMARY: S02AJ sequential first-block direct passed")


if __name__ == "__main__":
    main()
