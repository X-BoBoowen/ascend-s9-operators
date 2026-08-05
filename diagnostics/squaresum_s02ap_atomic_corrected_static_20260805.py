from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02ap_atomic_corrected_20260805"
    / "SquareSumV1"
)
HOST = CANDIDATE / "op_host" / "square_sum_v1.cpp"
KERNEL = CANDIDATE / "op_kernel" / "square_sum_v1.cpp"


def route(dtype, fast_path, inputs, reduce, outputs):
    atomic = (
        dtype == "float32"
        and fast_path == 1
        and inputs >= (1 << 18)
        and reduce >= 2048
        and outputs <= 8
    )
    workspace = (
        inputs >= (1 << 18)
        and reduce >= 2048
        and (
            (fast_path == 1 and dtype != "float32" and outputs <= 8)
            or (fast_path == 2 and outputs <= 1024)
        )
    )
    return atomic, workspace


def main():
    host = HOST.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")
    assert host.count("{") == host.count("}")
    assert kernel.count("{") == kernel.count("}")
    assert "ATOMIC_REDUCE_INPUT_THRESHOLD = 1U << 18U" in host
    assert "ATOMIC_REDUCE_MAX_OUTPUTS = 8" in host
    assert "inputDesc->GetDataType() != ge::DT_FLOAT" in host
    assert "FP32_LONG_TREE_MAX_OUTPUTS = 232U" in host
    assert "TILE_OUTPUTS * sizeof(float))" in kernel
    assert "outputBuffer_.Get<float>()" in kernel

    assert route("float32", 1, 1 << 18, 2048, 8) == (True, False)
    assert route("float16", 1, 1 << 18, 2048, 8) == (False, True)
    assert route("bfloat16", 1, 1 << 18, 2048, 8) == (False, True)
    assert route("float32", 2, 1 << 18, 2048, 127) == (False, True)
    assert route("float32", 1, (1 << 18) - 1, 2048, 8) == (False, False)

    print(f"HOST={HOST}")
    print(f"KERNEL={KERNEL}")
    print("SUMMARY: S02AP corrected atomic control passed")


if __name__ == "__main__":
    main()
