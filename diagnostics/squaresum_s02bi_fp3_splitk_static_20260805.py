import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BG = ROOT / (
    "candidates/squaresum_s02bg_fp32_middle8_compact_20260805/"
    "SquareSumV1"
)
BI = ROOT / (
    "candidates/squaresum_s02bi_fp3_splitk_20260805/"
    "SquareSumV1"
)
HOST = Path("op_host/square_sum_v1.cpp")
KERNEL = Path("op_kernel/square_sum_v1.cpp")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def source_scope():
    before_files = {
        path.relative_to(BG) for path in BG.rglob("*") if path.is_file()
    }
    after_files = {
        path.relative_to(BI) for path in BI.rglob("*") if path.is_file()
    }
    assert before_files == after_files
    for relative in before_files - {HOST, KERNEL}:
        assert sha256(BG / relative) == sha256(BI / relative)


def source_contract():
    host = (BI / HOST).read_text(encoding="utf-8")
    kernel = (BI / KERNEL).read_text(encoding="utf-8")
    assert "NONCONTIGUOUS_SPLITK_MIN_ROWS = 16U" in host
    assert "NONCONTIGUOUS_SPLITK_MIN_TAIL = 1024U" in host
    assert "NONCONTIGUOUS_SPLITK_MAX_TAIL = 16384U" in host
    assert "NONCONTIGUOUS_SPLITK_MAX_OUTPUTS = 8U" in host
    assert "expectedTrailingStride == trailingReduceElements" in host
    assert "? 14U" in host
    assert "reduceMode = 3U" in host
    assert "reduceMode_ == 2U || reduceMode_ == 3U" in kernel
    assert "GetSplitKRange" in kernel
    assert "batchRow % batchDim" in kernel
    assert "rowsUntilGroupEnd" in kernel
    assert "TILING_KEY_IS(14)" in kernel
    assert "ProcessStridedInnerBulk();" in kernel
    for forbidden in ("Case1", "Case2", "Case3", "Case4", "Case5"):
        assert forbidden not in host
        assert forbidden not in kernel


def gate_model():
    def enabled(inputs, reduce, outputs, rows, tail, grouped=True):
        return (
            grouped
            and inputs >= 1 << 18
            and reduce >= 1 << 15
            and 0 < outputs <= 8
            and rows >= 16
            and 1024 <= tail <= 16384
        )

    assert enabled(524288, 65536, 8, 16, 4096)
    assert enabled(4194304, 524288, 8, 128, 4096)
    assert enabled(327680, 40960, 8, 40, 1024)
    assert not enabled(262144, 32768, 8, 128, 256)
    assert not enabled(262144, 32768, 8, 8, 4096)
    assert not enabled(589824, 65536, 9, 16, 4096)
    assert not enabled(524288, 65536, 8, 16, 4096, grouped=False)


def main():
    source_scope()
    source_contract()
    gate_model()
    print("SOURCE_SCOPE=PASS")
    print("SPLITK_FAST_PATH=3")
    print("SPLITK_TILING_KEY=14")
    print("SPLITK_ROWS_MIN=16")
    print("SPLITK_TAIL_RANGE=1024..16384")
    print("SUMMARY: S02BI fastPath3 split-K static audit passed")


if __name__ == "__main__":
    main()
