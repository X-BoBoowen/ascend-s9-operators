from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "candidates/squaresum_s02bi_fp3_splitk_20260805/SquareSumV1"
CANDIDATE = ROOT / "candidates/squaresum_s02bj_compact_small_middle_20260806/SquareSumV1"
KERNEL = CANDIDATE / "op_kernel/square_sum_v1.cpp"
CHUNK = 8192
BLOCK_DIM = 40


def highest_power_of_two(value):
    result = 1
    while result * 2 <= value:
        result *= 2
    return result


def audit_model(type_bytes, inner, reduce_elements):
    elements_per_block = 32 // type_bytes
    rows_per_tile = highest_power_of_two(CHUNK // inner)
    base = reduce_elements // BLOCK_DIM
    extra = reduce_elements % BLOCK_DIM
    old_padded_inner = (
        (inner + elements_per_block - 1) // elements_per_block
        * elements_per_block
    )
    old_moved = 0
    new_moved = 0
    saw_tail_padding = False
    for block in range(BLOCK_DIM):
        remaining = base + (1 if block < extra else 0)
        while remaining:
            rows = min(remaining, rows_per_tile)
            reduction_rows = 1
            while reduction_rows < rows:
                reduction_rows *= 2
            input_count = rows * inner
            copy_padded = (
                (input_count + elements_per_block - 1)
                // elements_per_block
                * elements_per_block
            )
            assert 0 < copy_padded <= CHUNK
            assert 0 <= copy_padded - input_count < elements_per_block
            assert reduction_rows * inner <= CHUNK
            assert copy_padded - input_count <= 15
            old_moved += rows * old_padded_inner
            new_moved += copy_padded
            saw_tail_padding |= copy_padded != input_count
            remaining -= rows
    assert new_moved <= old_moved
    return old_moved, new_moved, saw_tail_padding


def main():
    kernel = KERNEL.read_text(encoding="utf-8")
    required = (
        "compactFullSmallInner",
        "innerElements_ <= 8U",
        "compactCopyPadded",
        "vectorInputCount",
        "compactCopyPadded != inputCount",
        "? inputCount * sizeof(T)",
    )
    for token in required:
        assert token in kernel, token
    forbidden = ("Case1", "Case2", "Case3", "Case4", "Case5")
    for token in forbidden:
        assert token not in kernel, token

    changed = []
    for path in sorted(CANDIDATE.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(CANDIDATE)
        baseline = BASELINE / relative
        assert baseline.is_file(), relative
        if path.read_bytes() != baseline.read_bytes():
            changed.append(relative.as_posix())
    assert changed == ["op_kernel/square_sum_v1.cpp"], changed

    checks = 0
    padded_tails = 0
    ratios = []
    for type_bytes in (2, 4):
        for inner in range(1, 9):
            for reduce_elements in (
                2048, 2049, 8191, 8192, 32771, 65539,
                131075, 262147,
            ):
                old, new, padded = audit_model(
                    type_bytes, inner, reduce_elements
                )
                ratios.append(old / new)
                padded_tails += int(padded)
                checks += 1
    assert padded_tails > 0
    print(
        "S02BJ_STATIC_PASS "
        f"checks={checks} padded_tail_grids={padded_tails} "
        f"min_ratio={min(ratios):.6f} max_ratio={max(ratios):.6f} "
        f"changed={changed}"
    )


if __name__ == "__main__":
    main()
