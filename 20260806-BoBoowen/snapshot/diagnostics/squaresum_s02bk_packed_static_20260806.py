from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "candidates/squaresum_s02bi_fp3_splitk_20260805/SquareSumV1"
CANDIDATE = ROOT / "candidates/squaresum_s02bk_packed_small_middle_20260806/SquareSumV1"
HOST = CANDIDATE / "op_host/square_sum_v1.cpp"
KERNEL = CANDIDATE / "op_kernel/square_sum_v1.cpp"
CHUNK = 8192
BLOCK_DIM = 40


def phase_layout(inner):
    phase_rows = 1
    while phase_rows * inner % 8:
        phase_rows *= 2
    return phase_rows, phase_rows * inner


def audit_partition(inner, reduce_elements):
    phase_rows, phase_span = phase_layout(inner)
    max_groups = CHUNK // phase_span
    assert 1 <= phase_rows <= 8
    assert phase_span % 8 == 0
    assert phase_span <= 56
    assert max_groups > 0
    base = reduce_elements // BLOCK_DIM
    extra = reduce_elements % BLOCK_DIM
    for block in range(BLOCK_DIM):
        reduce_count = base + (block < extra)
        offset = 0
        while reduce_count - offset >= phase_rows:
            groups = min(
                (reduce_count - offset) // phase_rows,
                max_groups,
            )
            assert groups > 0
            input_count = groups * phase_span
            assert input_count <= CHUNK
            group_offset = 0
            remaining = groups
            while remaining:
                current = 1 << (remaining.bit_length() - 1)
                assert group_offset * phase_span % 8 == 0
                assert current * phase_span <= CHUNK
                group_offset += current
                remaining -= current
            offset += groups * phase_rows
        tail_rows = reduce_count - offset
        assert 0 <= tail_rows < phase_rows
        assert tail_rows * inner < phase_span
    return phase_rows, phase_span


def main():
    host = HOST.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")
    for token in (
        "packedScratchStride",
        "scratchElements",
        "SafeAdd(\n                partialElements",
    ):
        assert token in host, token
    for token in (
        "FLOAT_ELEMENTS_PER_BLOCK",
        "ProcessParallelMiddlePackedPhases",
        "AccumulatePackedGroups",
        "packedScratchStride_",
        "MTE3_MTE2",
        "phaseSpan",
        "scratchPadding",
    ):
        assert token in kernel, token
    for token in ("Case1", "Case2", "Case3", "Case4", "Case5"):
        assert token not in host
        assert token not in kernel

    changed = []
    for path in sorted(CANDIDATE.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(CANDIDATE)
        baseline = BASELINE / relative
        assert baseline.is_file(), relative
        if path.read_bytes() != baseline.read_bytes():
            changed.append(relative.as_posix())
    assert changed == [
        "op_host/square_sum_v1.cpp",
        "op_kernel/square_sum_v1.cpp",
    ], changed

    checks = 0
    layouts = set()
    for inner in range(1, 8):
        for reduce_elements in (
            2048, 2049, 8191, 8192, 32771, 65539,
            131075, 262147,
        ):
            layouts.add(audit_partition(inner, reduce_elements))
            checks += 1
    scratch_elements = sum(span * BLOCK_DIM for _, span in layouts)
    print(
        "S02BK_STATIC_PASS "
        f"checks={checks} layouts={sorted(layouts)} "
        f"layout_scratch_sum={scratch_elements} changed={changed}"
    )


if __name__ == "__main__":
    main()
