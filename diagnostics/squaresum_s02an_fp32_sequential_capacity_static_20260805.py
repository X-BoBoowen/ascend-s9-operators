from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02an_fp32_sequential_capacity_20260805"
    / "SquareSumV1"
)
HOST = CANDIDATE / "op_host" / "square_sum_v1.cpp"
KERNEL = CANDIDATE / "op_kernel" / "square_sum_v1.cpp"

UB_BYTES = 192 * 1024
TILE_OUTPUTS = 1024
LONG_CHUNK = 16384
FP32_LONG_TREE_OUTPUTS = 232
BLOCKS = 64


def audit_source():
    host = HOST.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")
    assert host.count("{") == host.count("}")
    assert kernel.count("{") == kernel.count("}")
    assert "WORKSPACE_MIDDLE_MAX_OUTPUTS = 1024" in host
    assert "TREE_MIDDLE_REDUCE_THRESHOLD =\n        1U << 16U" in host
    assert "FP32_LONG_TREE_OUTPUTS = 232" in kernel
    assert "TILE_OUTPUTS * sizeof(float))" in kernel
    assert "outputBuffer_.Get<float>()" in kernel


def audit_capacity():
    output_bytes = TILE_OUTPUTS * 4
    input_bytes = LONG_CHUNK * 4
    sequential_float_bytes = TILE_OUTPUTS * 4
    tree_float_bytes = BLOCKS * FP32_LONG_TREE_OUTPUTS * 4
    reduce_work_bytes = LONG_CHUNK * 4
    sum_bytes = 32

    sequential_total = (
        output_bytes
        + input_bytes
        + sequential_float_bytes
        + reduce_work_bytes
        + sum_bytes
    )
    tree_total = (
        output_bytes
        + input_bytes
        + tree_float_bytes
        + reduce_work_bytes
        + sum_bytes
    )
    assert sequential_float_bytes == 4096
    assert sequential_total == 139296
    assert tree_total == 194592
    assert sequential_total <= UB_BYTES
    assert tree_total <= UB_BYTES
    return sequential_total, tree_total


def main():
    audit_source()
    sequential_total, tree_total = audit_capacity()
    print(f"HOST={HOST}")
    print(f"KERNEL={KERNEL}")
    print("SEQUENTIAL_FLOAT_BYTES=4096")
    print(f"SEQUENTIAL_UB_BYTES={sequential_total}/{UB_BYTES}")
    print(f"TREE_UB_BYTES={tree_total}/{UB_BYTES}")
    print("SUMMARY: S02AN FP32 sequential capacity passed")


if __name__ == "__main__":
    main()
