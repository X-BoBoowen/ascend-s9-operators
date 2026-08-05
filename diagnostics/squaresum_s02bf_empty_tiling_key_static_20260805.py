import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BE = ROOT / (
    "candidates/squaresum_s02be_empty_reduce_zero_20260805/"
    "SquareSumV1"
)
BF = ROOT / (
    "candidates/squaresum_s02bf_empty_tiling_key_20260805/"
    "SquareSumV1"
)
KERNEL = Path("op_kernel/square_sum_v1.cpp")
HOST = Path("op_host/square_sum_v1.cpp")
CMAKE = Path("op_kernel/CMakeLists.txt")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    before_files = {
        path.relative_to(BE) for path in BE.rglob("*") if path.is_file()
    }
    after_files = {
        path.relative_to(BF) for path in BF.rglob("*") if path.is_file()
    }
    assert before_files == after_files
    for relative in before_files - {KERNEL, HOST, CMAKE}:
        assert sha256(BE / relative) == sha256(BF / relative)

    cmake = (BF / CMAKE).read_text(encoding="utf-8")
    assert "--tiling_key=1,2,3,4,5,6,7,8,9,10,11,12,13" in cmake

    kernel = (BF / KERNEL).read_text(encoding="utf-8")
    assert "bool EMPTY_REDUCTION = false>" in kernel
    assert "if constexpr (EMPTY_REDUCTION)" in kernel
    assert "if (reduceElements_ == 0U)" not in kernel
    assert "TILING_KEY_IS(13)" in kernel
    assert "8U,\n            true> op;" in kernel
    assert kernel.count("ProcessEmptyReduction();") == 1

    host = (BF / HOST).read_text(encoding="utf-8")
    assert "reduceElements == 0U\n            ? 13U" in host
    assert host.count("context->SetTilingKey(") == 1

    print("SOURCE_SCOPE=PASS")
    print("EMPTY_TILING_KEY=13")
    print("NONEMPTY_RUNTIME_EMPTY_BRANCHES=0")
    print("SUMMARY: S02BF empty tiling-key static audit passed")


if __name__ == "__main__":
    main()
