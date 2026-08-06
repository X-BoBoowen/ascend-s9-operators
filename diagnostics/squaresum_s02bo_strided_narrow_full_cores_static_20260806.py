from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "candidates"
    / "squaresum_s02bn_long_tail_splitk_20260806"
    / "SquareSumV1"
)
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02bo_strided_narrow_full_cores_20260806"
    / "SquareSumV1"
)
INPUT_THRESHOLD = 1 << 20
MAX_BLOCKS = 40
OUTPUTS_PER_CORE = 64


def files(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def blocks(fast_path, inputs, outputs, inner, candidate):
    if fast_path in (1, 3) or (
        candidate
        and fast_path == 4
        and inputs >= INPUT_THRESHOLD
        and inner <= 32
    ):
        return min(outputs, MAX_BLOCKS)
    return min((outputs + OUTPUTS_PER_CORE - 1) // OUTPUTS_PER_CORE, MAX_BLOCKS)


def main():
    base_files = files(BASE)
    candidate_files = files(CANDIDATE)
    assert set(base_files) == set(candidate_files)
    changed = [
        name
        for name in sorted(base_files)
        if base_files[name] != candidate_files[name]
    ]
    assert changed == ["op_host/square_sum_v1.cpp"], changed

    host = (CANDIDATE / "op_host" / "square_sum_v1.cpp").read_text(
        encoding="utf-8"
    )
    for token in (
        "fastPath == 4 &&",
        "inputElements >= FULL_CORE_INPUT_THRESHOLD",
        "innerElements <= 32U",
    ):
        assert token in host, token

    targets = (
        (1 << 20, 256, 1),
        (1 << 20, 256, 2),
        (1 << 20, 256, 4),
        (1 << 20, 256, 8),
        (1 << 20, 256, 16),
        (1 << 20, 256, 32),
    )
    for inputs, outputs, inner in targets:
        assert blocks(4, inputs, outputs, inner, False) == 4
        assert blocks(4, inputs, outputs, inner, True) == 40

    assert blocks(4, INPUT_THRESHOLD - 1, 256, 1, True) == 4
    assert blocks(4, INPUT_THRESHOLD, 256, 33, True) == 4
    assert blocks(4, INPUT_THRESHOLD, 256, 64, True) == 4
    assert blocks(2, INPUT_THRESHOLD, 256, 1, True) == 4
    assert blocks(4, INPUT_THRESHOLD, 1, 1, True) == 1

    print("S02BO_STATIC_PASS changed=host routing_checks=11")


if __name__ == "__main__":
    main()
