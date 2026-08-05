import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BF = ROOT / (
    "candidates/squaresum_s02bf_empty_tiling_key_20260805/"
    "SquareSumV1"
)
BG = ROOT / (
    "candidates/squaresum_s02bg_fp32_middle8_compact_20260805/"
    "SquareSumV1"
)
KERNEL = Path("op_kernel/square_sum_v1.cpp")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def source_scope():
    before = (BF / KERNEL).read_text(encoding="utf-8")
    after = (BG / KERNEL).read_text(encoding="utf-8")
    exclusion = """            const bool compactFullInner8 =
                !std::is_same<T, float>::value &&
                innerElements_ == 8U &&"""
    promotion = """            const bool compactFullInner8 =
                innerElements_ == 8U &&"""
    assert before.count(exclusion) == 1
    assert after == before.replace(exclusion, promotion, 1)

    before_files = {
        path.relative_to(BF) for path in BF.rglob("*") if path.is_file()
    }
    after_files = {
        path.relative_to(BG) for path in BG.rglob("*") if path.is_file()
    }
    assert before_files == after_files
    for relative in before_files - {KERNEL}:
        assert sha256(BF / relative) == sha256(BG / relative)


def alignment_and_capacity():
    checks = 0
    for chunk in (8192, 16384):
        width = 8
        rows_per_tile = chunk // width
        for type_bytes in (2, 4):
            input_elements = rows_per_tile * width
            block_len = rows_per_tile * width * type_bytes
            assert input_elements == chunk
            assert block_len % 32 == 0
            assert 0 < block_len <= 0xFFFFFFFF

            active_rows = rows_per_tile
            while active_rows > 1:
                half_rows = active_rows // 2
                source_offset_bytes = half_rows * width * 4
                assert source_offset_bytes % 32 == 0
                active_rows //= 2
                checks += 1
    return checks


def main():
    source_scope()
    checks = alignment_and_capacity()
    print("SOURCE_SCOPE=PASS")
    print("PROMOTED_DTYPE=FP32")
    print("PROMOTED_INNER_WIDTH=8")
    print(f"TREE_ALIGNMENT_CHECKS={checks}")
    print("SUMMARY: S02BG FP32 middle8 static audit passed")


if __name__ == "__main__":
    main()
