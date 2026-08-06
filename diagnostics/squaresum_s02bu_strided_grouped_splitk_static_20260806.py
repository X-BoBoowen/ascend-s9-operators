from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "candidates"
    / "squaresum_s02bt_strided_grouped_padded_rows_20260806"
    / "SquareSumV1"
)
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02bu_strided_grouped_splitk_20260806"
    / "SquareSumV1"
)


def files(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def selects_splitk(
    group_dim,
    inner,
    last_reduce,
    outer_reduce_groups,
    input_elements,
    output_elements,
    type_bytes,
):
    block_elements = 32 // type_bytes
    padded_inner = (
        (inner + block_elements - 1) // block_elements * block_elements
    )
    return (
        2 <= inner <= 16
        and group_dim >= 8
        and 2 <= last_reduce <= 64
        and last_reduce & (last_reduce - 1) == 0
        and outer_reduce_groups >= 40
        and input_elements >= 1 << 18
        and output_elements <= 512
        and 8 * last_reduce * padded_inner <= 8192
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
    assert "stridedGroupedSplitK" in host
    assert "reduceMode = 5U" in host
    assert "STRIDED_SPLITK_MIN_INPUT" in host
    assert "STRIDED_SPLITK_MAX_OUTPUTS" in host
    assert "ProcessStridedGroupedSplitK();" in kernel
    assert "reduceMode_ == 5U" in kernel
    assert "FinalizeParallelReductionTree();" in kernel
    assert "partialWorkspaceOffset + outputStart" in kernel
    kernel_compact = " ".join(kernel.split())
    assert (
        "reduceMode_ == 4U || reduceMode_ == 5U "
        "? AscendC::GetBlockNum()" in kernel_compact
    )

    assert selects_splitk(128, 2, 64, 64, 1 << 20, 256, 2)
    assert selects_splitk(64, 4, 64, 40, 1 << 20, 256, 4)
    assert selects_splitk(32, 8, 64, 41, 1 << 20, 256, 2)
    assert selects_splitk(16, 16, 64, 80, 1 << 20, 256, 4)
    assert not selects_splitk(128, 1, 64, 64, 1 << 20, 128, 2)
    assert not selects_splitk(128, 2, 64, 39, 1 << 20, 256, 2)
    assert not selects_splitk(128, 2, 64, 64, 1 << 17, 256, 2)
    assert not selects_splitk(512, 2, 64, 64, 1 << 20, 1024, 2)

    print("S02BU_STATIC_PASS changed=host,kernel route_checks=8")


if __name__ == "__main__":
    main()
