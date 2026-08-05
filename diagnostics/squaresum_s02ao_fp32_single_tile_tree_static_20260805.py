from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02ao_fp32_single_tile_tree_20260805"
    / "SquareSumV1"
)
HOST = CANDIDATE / "op_host" / "square_sum_v1.cpp"
KERNEL = CANDIDATE / "op_kernel" / "square_sum_v1.cpp"
MAX_OUTPUTS = 232


def select_tree(dtype, long_chunk, output_elements, workspace, reduce):
    fits = dtype != "float32" or not long_chunk or output_elements <= MAX_OUTPUTS
    return workspace and fits and reduce >= (1 << 16)


def main():
    host = HOST.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")
    assert host.count("{") == host.count("}")
    assert kernel.count("{") == kernel.count("}")
    assert "FP32_LONG_TREE_MAX_OUTPUTS = 232U" in host
    assert "fp32LongTreeFitsOneTile" in host
    assert "outputElements <= FP32_LONG_TREE_MAX_OUTPUTS" in host
    assert "TILE_OUTPUTS * sizeof(float))" in kernel
    assert "outputBuffer_.Get<float>()" in kernel

    assert select_tree("float32", True, 232, True, 1 << 16)
    assert not select_tree("float32", True, 233, True, 1 << 16)
    assert not select_tree("float32", True, 1024, True, 1 << 16)
    assert select_tree("float16", True, 233, True, 1 << 16)
    assert select_tree("bfloat16", True, 233, True, 1 << 16)
    assert select_tree("float32", False, 233, True, 1 << 16)
    assert not select_tree("float32", True, 232, False, 1 << 16)
    assert not select_tree("float32", True, 232, True, (1 << 16) - 1)

    print(f"HOST={HOST}")
    print(f"KERNEL={KERNEL}")
    print("FP32_LONG_TREE_MAX_OUTPUTS=232")
    print("SUMMARY: S02AO single-tile FP32 tree gate passed")


if __name__ == "__main__":
    main()
