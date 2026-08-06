from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "candidates"
    / "squaresum_s02br_strided_grouped_rows_key5_20260806"
    / "SquareSumV1"
)
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02bs_strided_grouped_scalar_rows_20260806"
    / "SquareSumV1"
)


def files(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


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
    assert "innerElements == 1U" in host
    assert "lastReduceDim <= 64U" in host
    method = kernel.rsplit(
        "__aicore__ inline void ProcessStridedGroupedRows()", 1
    )[1].split("__aicore__ inline bool CanUseStridedInnerBulk()", 1)[0]
    assert method.count("AscendC::WholeReduceSum<float>(") == 3
    assert "ReduceRowsInPlace(" not in method
    assert "sumBuffer_.Get<float>()" in method
    assert "TILING_KEY_IS(14)" not in kernel

    print("S02BS_STATIC_PASS changed=host,kernel aligned_whole_reduce=3")


if __name__ == "__main__":
    main()
