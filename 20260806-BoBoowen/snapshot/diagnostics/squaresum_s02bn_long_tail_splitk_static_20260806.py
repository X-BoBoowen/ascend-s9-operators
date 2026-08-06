from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "candidates"
    / "squaresum_s02bm_split_narrow_workspace_20260806"
    / "SquareSumV1"
)
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02bn_long_tail_splitk_20260806"
    / "SquareSumV1"
)
CHUNK = 4096
MAX_BLOCKS = 40


def files(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def enabled(
    inputs,
    reduce,
    outputs,
    tail,
    first_trailing_axis=1,
    reduce_rank=2,
    expected_trailing_stride=None,
):
    if expected_trailing_stride is None:
        expected_trailing_stride = tail
    natural_rows = 0 if tail == 0 else reduce // tail
    chunks_per_row = 0 if tail == 0 else (tail - 1) // CHUNK + 1
    work_units = natural_rows * chunks_per_row
    return (
        inputs >= 1 << 18
        and reduce >= 1 << 15
        and 0 < outputs <= 16
        and tail > 16384
        and 0 < first_trailing_axis < reduce_rank
        and expected_trailing_stride == tail
        and work_units >= MAX_BLOCKS
    )


def main():
    base_files = files(BASE)
    candidate_files = files(CANDIDATE)
    assert set(base_files) == set(candidate_files)
    changed = [
        name
        for name in sorted(base_files)
        if base_files[name] != candidate_files[name]
    ]
    assert changed == [
        "op_host/square_sum_v1.cpp",
        "op_kernel/square_sum_v1.cpp",
    ], changed

    host = (CANDIDATE / "op_host" / "square_sum_v1.cpp").read_text(
        encoding="utf-8"
    )
    kernel = (CANDIDATE / "op_kernel" / "square_sum_v1.cpp").read_text(
        encoding="utf-8"
    )
    for token in (
        "NONCONTIGUOUS_LONG_SPLITK_CHUNK = 4096U",
        "NONCONTIGUOUS_LONG_SPLITK_MAX_OUTPUTS = 16U",
        "noncontiguousLongTailSplitK",
        "longTailWorkUnits >= MAX_BLOCK_DIM",
        "reduceMode = 4U",
        "noncontiguousSplitK || noncontiguousLongTailSplitK",
    ):
        assert token in host, token
    for token in (
        "LONG_TAIL_SPLIT_CHUNK = 4096",
        "reduceMode_ == 4U",
        "ProcessGroupedLongTailSplitK",
        "GetSplitKRange(totalUnits, firstUnit, limitUnit)",
        "ReduceInputOffset(reduceIndex)",
        "FinalizeParallelReductionTree",
    ):
        assert token in kernel, token

    assert enabled(4 * 8 * 65536, 4 * 65536, 8, 65536)
    assert enabled(4 * 9 * 65536, 4 * 65536, 9, 65536)
    assert enabled(2 * 16 * 131072, 2 * 131072, 16, 131072)
    assert enabled(8 * 8 * 16385, 8 * 16385, 8, 16385)
    assert not enabled(16 * 8 * 16384, 16 * 16384, 8, 16384)
    assert not enabled(4 * 17 * 65536, 4 * 65536, 17, 65536)
    assert not enabled(7 * 8 * 20000, 7 * 20000, 8, 20000)
    assert not enabled((1 << 18) - 1, 8 * 16385, 1, 16385)
    assert not enabled(4 * 8 * 65536, (1 << 15) - 1, 8, 65536)
    assert not enabled(
        4 * 8 * 65536,
        4 * 65536,
        8,
        65536,
        expected_trailing_stride=65535,
    )

    print(
        "S02BN_STATIC_PASS changed=host+kernel "
        "routing_checks=10 implementation_tokens=12"
    )


if __name__ == "__main__":
    main()
