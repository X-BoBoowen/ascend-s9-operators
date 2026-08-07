from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = (
    ROOT
    / "baselines"
    / "squaresum_s02f_global_best_20260806"
    / "SquareSumV1"
)
REFERENCE = (
    ROOT
    / "candidates"
    / "squaresum_s02ay_workspace_task_boundary_20260805"
    / "SquareSumV1"
)
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s03i_s02f_fast2_small_inner_tree_20260807"
    / "SquareSumV1"
)


def extract_function(source, marker):
    marker_index = source.index(marker)
    start = source.rfind("\n", 0, marker_index) + 1
    brace = source.index("{", marker_index)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated function: {marker}")


def compact(value):
    return "".join(value.split())


def main():
    host = (CANDIDATE / "op_host/square_sum_v1.cpp").read_text(
        encoding="utf-8"
    )
    kernel = (CANDIDATE / "op_kernel/square_sum_v1.cpp").read_text(
        encoding="utf-8"
    )
    reference_kernel = (
        REFERENCE / "op_kernel/square_sum_v1.cpp"
    ).read_text(encoding="utf-8")
    assert host.count("{") == host.count("}")
    assert kernel.count("{") == kernel.count("}")
    assert "const bool longMiddleSmallInner =" in host
    assert "innerElements <= 64U;" in host
    assert "longContiguous || longMiddleSmallInner" in host

    for marker in (
        "__aicore__ inline void ProcessParallelMiddle",
        "__aicore__ inline void ProcessMiddleContiguous",
    ):
        assert compact(extract_function(kernel, marker)) == compact(
            extract_function(reference_kernel, marker)
        ), marker

    # All packaging/tiling declarations stay on the S02F contract.
    for relative in (
        "op_host/square_sum_v1_tiling.h",
        "op_kernel/CMakeLists.txt",
    ):
        assert (BASELINE / relative).read_bytes() == (
            CANDIDATE / relative
        ).read_bytes(), relative

    # Both legacy five-argument callers and the new initialize-on-first-tile
    # middle path must remain available.
    assert kernel.count("__aicore__ inline void ReduceRowsInto(") == 2
    assert "const bool initialize" in kernel
    assert kernel.count("reduceOffset == 0U") == 6
    assert kernel.count("paddedRowElements != 0U") == 9

    print("REFERENCE_MIDDLE_FUNCTIONS=2/2")
    print("S02F_CONTRACT_FILES=2/2")
    print("INITIALIZE_CALLS=6")
    print("TAIL_PADDING_BRANCHES=9")
    print("SUMMARY: S03I selective middle-tree static audit passed")


if __name__ == "__main__":
    main()
