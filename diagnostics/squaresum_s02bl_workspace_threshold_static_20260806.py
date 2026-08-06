from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "candidates"
    / "squaresum_s02bk_packed_small_middle_20260806"
    / "SquareSumV1"
)
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02bl_lower_narrow_workspace_20260806"
    / "SquareSumV1"
)


def files(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def workspace_enabled(fast_path, inner, inputs, reduce, outputs):
    threshold = 1 << 15 if fast_path == 2 and inner <= 8 else 1 << 18
    max_outputs = 1024 if fast_path == 2 else 8
    return (
        inputs >= threshold
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
        "NARROW_MIDDLE_WORKSPACE_INPUT_THRESHOLD",
        "fastPath == 2U && innerElements <= 8U",
        "inputElements >= workspaceInputThreshold",
        "1U << 15U",
    )
    for token in required:
        assert token in host, token

    assert not workspace_enabled(2, 8, 32760, 4095, 8)
    assert workspace_enabled(2, 8, 32768, 4096, 8)
    assert workspace_enabled(2, 2, 32768, 16384, 2)
    assert workspace_enabled(2, 4, 32768, 8192, 4)
    assert workspace_enabled(2, 8, 262136, 32767, 8)
    assert not workspace_enabled(2, 16, 32768, 2048, 16)
    assert not workspace_enabled(2, 128, 262016, 2047, 128)
    assert not workspace_enabled(1, 1, 262143, 262143, 1)
    assert workspace_enabled(1, 1, 262144, 262144, 1)

    print("S02BL_STATIC_PASS changed=op_host/square_sum_v1.cpp checks=12")


if __name__ == "__main__":
    main()
