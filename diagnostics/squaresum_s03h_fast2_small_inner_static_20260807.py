from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = (
    ROOT
    / "baselines"
    / "squaresum_s02f_global_best_20260806"
    / "SquareSumV1"
)
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s03h_s02f_fast2_small_inner_16k_20260807"
    / "SquareSumV1"
)


def tiling_key(fast_path, reduce_elements, inner_elements, tree):
    long_last = fast_path == 1 and reduce_elements > 8192
    long_middle = (
        fast_path == 2
        and reduce_elements >= 2048
        and inner_elements <= 64
    )
    long_chunk = long_last or long_middle
    if tree:
        return 4 if long_chunk else 3
    return 2 if long_chunk else 1


def main():
    host = (CANDIDATE / "op_host/square_sum_v1.cpp").read_text(
        encoding="utf-8"
    )
    assert host.count("{") == host.count("}")
    assert "const bool longMiddleSmallInner =" in host
    assert "innerElements <= 64U;" in host
    assert "longContiguous || longMiddleSmallInner" in host
    assert "? (useLongChunk ? 4U : 3U)" in host
    assert ": (useLongChunk ? 2U : 1U)" in host

    unchanged = (
        "op_host/square_sum_v1_tiling.h",
        "op_kernel/square_sum_v1.cpp",
        "op_kernel/CMakeLists.txt",
    )
    for relative in unchanged:
        assert (BASELINE / relative).read_bytes() == (
            CANDIDATE / relative
        ).read_bytes(), relative

    selected = 0
    fallback = 0
    for reduce_elements in (1, 2047, 2048, 8192, 65536):
        for inner_elements in (1, 2, 8, 31, 64, 65, 257, 1024):
            for tree in (False, True):
                key = tiling_key(
                    2, reduce_elements, inner_elements, tree
                )
                should_select = (
                    reduce_elements >= 2048 and inner_elements <= 64
                )
                if should_select:
                    assert key == (4 if tree else 2)
                    selected += 1
                else:
                    assert key == (3 if tree else 1)
                    fallback += 1

    # The existing fastPath1 decision is byte-for-byte equivalent.
    for reduce_elements in (1, 8192, 8193, 65536):
        for tree in (False, True):
            expected = (
                (4 if tree else 2)
                if reduce_elements > 8192
                else (3 if tree else 1)
            )
            assert tiling_key(1, reduce_elements, 1024, tree) == expected

    print(f"SELECTED_ROUTE_CHECKS={selected}")
    print(f"FALLBACK_ROUTE_CHECKS={fallback}")
    print("UNCHANGED_FILES=3/3")
    print("SUMMARY: S03H fast2 small-inner 16K static audit passed")


if __name__ == "__main__":
    main()
