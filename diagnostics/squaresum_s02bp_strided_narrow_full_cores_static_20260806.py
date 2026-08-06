from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "candidates"
    / "squaresum_s02bo_strided_narrow_full_cores_20260806"
    / "SquareSumV1"
)
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02bp_strided_narrow_full_cores_20260806"
    / "SquareSumV1"
)
MAX_BLOCKS = 40
SMALL_BLOCKS = 32
INPUT_THRESHOLD = 1 << 20
OUTPUTS_PER_CORE = 64


def files(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def blocks(fast_path, inputs, outputs, inner, candidate):
    if fast_path in (1, 3) or (
        candidate and fast_path == 4 and inner <= 16
    ):
        block_limit = MAX_BLOCKS if inputs >= INPUT_THRESHOLD else SMALL_BLOCKS
        return min(outputs, block_limit)
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
    route = "(fastPath == 4 &&\n         innerElements <= 16U)"
    assert route in host
    assert "inputElements >= FULL_CORE_INPUT_THRESHOLD &&\n         innerElements" not in host

    for inner in (1, 2, 4, 8, 16):
        assert blocks(4, INPUT_THRESHOLD, 256, inner, False) == 4
        assert blocks(4, INPUT_THRESHOLD, 256, inner, True) == 40
        assert blocks(4, INPUT_THRESHOLD // 4, 64, inner, True) == 32
    for inner in (17, 32, 33, 64):
        assert blocks(4, INPUT_THRESHOLD, 256, inner, True) == 4
    assert blocks(2, INPUT_THRESHOLD, 256, 1, True) == 4
    assert blocks(4, INPUT_THRESHOLD, 1, 1, True) == 1

    print("S02BP_STATIC_PASS changed=host routing_checks=16")


if __name__ == "__main__":
    main()
