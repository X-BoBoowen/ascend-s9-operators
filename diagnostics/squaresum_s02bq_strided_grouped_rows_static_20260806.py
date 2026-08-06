from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "candidates"
    / "squaresum_s02bp_strided_narrow_full_cores_20260806"
    / "SquareSumV1"
)
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02bq_strided_grouped_rows_20260806"
    / "SquareSumV1"
)


def files(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def grouped_route(group_dim, inner, last_reduce_dim, outer_rows=1):
    if inner <= 0 or inner > 16:
        return False
    if last_reduce_dim <= 1 or last_reduce_dim & (last_reduce_dim - 1):
        return False
    row_elements = last_reduce_dim * inner
    padded_row = (row_elements + 7) // 8 * 8
    if 8 * padded_row > 8192:
        return False
    tasks = outer_rows * ((group_dim + 7) // 8)
    return group_dim >= 8 and tasks >= 32


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
    assert "stridedGroupedRows" in host
    assert "stridedGroupedTasks >=\n                        SMALL_FAST_BLOCK_DIM" in host
    assert "? 14U" in host
    assert "ProcessStridedGroupedRows();" in kernel
    assert "TILING_KEY_IS(14)" in kernel
    assert "copyParams.blockCount =\n                    static_cast<uint16_t>(activeRows);" in kernel
    assert "copyParams.srcStride = 0U;" in kernel

    assert grouped_route(256, 1, 64)
    assert grouped_route(64, 1, 64, outer_rows=4)
    assert not grouped_route(64, 1, 64)
    assert not grouped_route(256, 17, 64)
    assert not grouped_route(256, 1, 63)
    assert not grouped_route(256, 16, 1024)

    print("S02BQ_STATIC_PASS changed=host,kernel route_checks=6")


if __name__ == "__main__":
    main()
