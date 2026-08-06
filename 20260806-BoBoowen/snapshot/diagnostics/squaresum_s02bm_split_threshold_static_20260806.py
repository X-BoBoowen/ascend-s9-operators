from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "candidates"
    / "squaresum_s02bl_lower_narrow_workspace_20260806"
    / "SquareSumV1"
)
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02bm_split_narrow_workspace_20260806"
    / "SquareSumV1"
)


def files(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def threshold(fast_path, inner):
    if fast_path == 2 and inner < 8:
        return 1 << 12
    if fast_path == 2 and inner == 8:
        return 1 << 15
    return 1 << 18


def enabled(fast_path, inner, inputs, reduce, outputs):
    max_outputs = 1024 if fast_path == 2 else 8
    return (
        inputs >= threshold(fast_path, inner)
        and reduce >= 2048
        and fast_path in (1, 2)
        and outputs <= max_outputs
    )


def main():
    base_files = files(BASE)
    candidate_files = files(CANDIDATE)
    assert set(base_files) == set(candidate_files)
    changed = [
        name for name in sorted(base_files)
        if base_files[name] != candidate_files[name]
    ]
    assert changed == ["op_host/square_sum_v1.cpp"], changed

    host = (CANDIDATE / "op_host" / "square_sum_v1.cpp").read_text(
        encoding="utf-8"
    )
    required = (
        "PACKED_MIDDLE_WORKSPACE_INPUT_THRESHOLD",
        "ALIGNED8_MIDDLE_WORKSPACE_INPUT_THRESHOLD",
        "fastPath == 2U && innerElements < 8U",
        "fastPath == 2U && innerElements == 8U",
        "inputElements >= workspaceInputThreshold",
    )
    for token in required:
        assert token in host, token

    assert not enabled(2, 2, 4094, 2047, 2)
    assert enabled(2, 2, 4096, 2048, 2)
    assert enabled(2, 4, 8192, 2048, 4)
    assert enabled(2, 7, 14336, 2048, 7)
    assert not enabled(2, 8, 16384, 2048, 8)
    assert enabled(2, 8, 32768, 4096, 8)
    assert not enabled(2, 16, 262128, 16383, 16)
    assert not enabled(1, 1, 262143, 262143, 1)
    assert enabled(1, 1, 262144, 262144, 1)

    print("S02BM_STATIC_PASS changed=op_host/square_sum_v1.cpp checks=12")


if __name__ == "__main__":
    main()
