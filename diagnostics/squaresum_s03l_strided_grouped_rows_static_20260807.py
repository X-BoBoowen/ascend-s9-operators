from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "submission-src/SquareSumV1/op_host/square_sum_v1.cpp"
KERNEL = ROOT / "submission-src/SquareSumV1/op_kernel/square_sum_v1.cpp"
CMAKE = ROOT / "submission-src/SquareSumV1/op_kernel/CMakeLists.txt"


def routes(last_reduce, inner, grouped_dim, outer_rows, input_elements, reduce):
    if (
        input_elements < (1 << 20)
        or reduce < 2048
        or last_reduce <= 1
        or last_reduce & (last_reduce - 1)
        or inner <= 0
        or inner % 8
    ):
        return False
    row_elements = last_reduce * inner
    if row_elements > 8192:
        return False
    for width in (8, 4, 2):
        tasks = outer_rows * ((grouped_dim + width - 1) // width)
        if (
            grouped_dim >= width
            and width * row_elements <= 8192
            and width * inner <= 1024
            and tasks >= 32
        ):
            return True
    return False


POSITIVES = (
    (128, 16, 32, 8, 33554432, 8192),
    (64, 16, 64, 4, 16777216, 4096),
    (128, 8, 32, 8, 8388608, 4096),
    (128, 16, 16, 16, 16777216, 4096),
    (64, 16, 33, 7, 15138816, 4096),
)
NEGATIVES = (
    (127, 16, 32, 8, 33292288, 8128),
    (128, 15, 32, 8, 31457280, 8192),
    (128, 16, 8, 4, 4194304, 8192),
    (128, 16, 32, 8, 1048575, 8192),
    (128, 16, 32, 8, 33554432, 2047),
)


def main():
    assert all(routes(*case) for case in POSITIVES)
    assert not any(routes(*case) for case in NEGATIVES)
    host = HOST.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")
    cmake = CMAKE.read_text(encoding="utf-8")
    for marker in (
        "stridedGroupedRows",
        "STRIDED_GROUPED_MIN_TASKS",
        "? 7U",
    ):
        assert marker in host, marker
    for marker in (
        "STRIDED_GROUPED_ROWS",
        "ProcessStridedGroupedRows",
        "TILING_KEY_IS(7)",
    ):
        assert marker in kernel, marker
    assert "--tiling_key=1,2,3,4,5,6,7" in cmake
    print(f"POSITIVE_ROUTE_CASES={len(POSITIVES)}/{len(POSITIVES)}")
    print(f"NEGATIVE_ROUTE_CASES={len(NEGATIVES)}/{len(NEGATIVES)}")
    print("S03L_STATIC_AUDIT_PASS")


if __name__ == "__main__":
    main()
