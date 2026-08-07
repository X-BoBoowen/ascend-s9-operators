from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "submission-src/SquareSumV1/op_host/square_sum_v1.cpp"
KERNEL = ROOT / "submission-src/SquareSumV1/op_kernel/square_sum_v1.cpp"


def route(input_elements, reduce_elements, inner_elements, output_rows):
    output_elements = inner_elements * output_rows
    return (
        input_elements >= (1 << 20)
        and reduce_elements >= 2048
        and 0 < inner_elements <= 8
        and output_elements % inner_elements == 0
        and output_rows >= 40
    )


def main():
    host = HOST.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")

    positives = (
        (1 << 20, 2048, 1, 40),
        (1 << 20, 8192, 2, 64),
        (4 << 20, 8192, 8, 64),
        (8 << 20, 4097, 4, 73),
    )
    negatives = (
        ((1 << 20) - 1, 8192, 2, 64),
        (1 << 20, 2047, 2, 256),
        (1 << 20, 8192, 9, 64),
        (1 << 20, 8192, 8, 39),
        (1 << 20, 8192, 0, 64),
    )
    assert all(route(*case) for case in positives)
    assert not any(route(*case) for case in negatives)

    required_host = (
        "const bool middleFullRows =",
        "!middleFullRows &&",
        "? middleOutputRows",
        "? 6U",
    )
    required_kernel = (
        "bool MIDDLE_FULL_ROWS = false>",
        "if constexpr (MIDDLE_FULL_ROWS)",
        "ProcessMiddleFullRows();",
        "ReduceRowsIntoEight(",
        "TILING_KEY_IS(6)",
        "LONG_CHUNK,\n            false,\n            false,\n            true> op;",
    )
    for marker in required_host:
        assert marker in host, marker
    for marker in required_kernel:
        assert marker in kernel, marker
    assert kernel.count("ProcessMiddleFullRows()") == 2
    assert kernel.count("ReduceRowsIntoEight(") == 4

    print(f"POSITIVE_ROUTE_CASES={len(positives)}/{len(positives)}")
    print(f"NEGATIVE_ROUTE_CASES={len(negatives)}/{len(negatives)}")
    print("S03K_STATIC_AUDIT_PASS")


if __name__ == "__main__":
    main()
