from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "candidates"
    / "squaresum_s02bs_strided_grouped_scalar_rows_20260806"
    / "SquareSumV1"
)
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02bt_strided_grouped_padded_rows_20260806"
    / "SquareSumV1"
)


def files(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def select_width(group_dim, outer_rows, last_reduce, inner, type_bytes):
    block_elements = 32 // type_bytes
    padded_inner = (inner + block_elements - 1) // block_elements * block_elements
    for width in (8, 4, 2, 1):
        if group_dim < width:
            continue
        if width * last_reduce * padded_inner > 8192:
            continue
        tasks = outer_rows * ((group_dim + width - 1) // width)
        min_tasks = 32 if inner <= 8 else 4
        if tasks >= min_tasks:
            return width, tasks
    return 0, 0


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
    assert "stridedGroupedWidth == 8U" in host
    assert "stridedGroupedWidth == 4U" in host
    assert "stridedGroupedWidth == 2U" in host
    assert "ProcessStridedGroupedPaddedRows();" in kernel
    assert "activeRows * lastReduceDim" in kernel
    assert "ReduceAndAccumulateGroupedRows(" in kernel
    assert "TILING_KEY_IS(14)" not in kernel

    assert select_width(256, 1, 64, 1, 2) == (8, 32)
    assert select_width(128, 1, 64, 2, 2) == (4, 32)
    assert select_width(64, 1, 64, 4, 2) == (2, 32)
    assert select_width(32, 1, 64, 8, 2) == (1, 32)
    assert select_width(16, 1, 64, 16, 2) == (4, 4)
    assert select_width(16, 2, 64, 16, 2) == (8, 4)

    print("S02BT_STATIC_PASS changed=host,kernel adaptive_width_checks=6")


if __name__ == "__main__":
    main()
