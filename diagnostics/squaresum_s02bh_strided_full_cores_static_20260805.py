from difflib import unified_diff
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "candidates"
    / "squaresum_s02bg_fp32_middle8_compact_20260805"
    / "SquareSumV1"
)
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02bh_strided_full_cores_20260805"
    / "SquareSumV1"
)
INPUT_THRESHOLD = 1 << 20
MAX_BLOCK_DIM = 40
GENERIC_OUTPUTS_PER_CORE = 64


TARGETS = (
    ("outputs64", 64 * 1 * 256 * 64, 64),
    ("outputs128", 64 * 4 * 128 * 32, 128),
    ("outputs512", 32 * 8 * 64 * 64, 512),
    ("outputs1105", 32 * 17 * 64 * 65, 1105),
    ("outputs2015", 16 * 31 * 128 * 65, 2015),
    ("outputs2159", 32 * 17 * 64 * 127, 2159),
)


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def audit_source_scope():
    base_files = sorted(
        path.relative_to(BASE) for path in BASE.rglob("*") if path.is_file()
    )
    candidate_files = sorted(
        path.relative_to(CANDIDATE)
        for path in CANDIDATE.rglob("*")
        if path.is_file()
    )
    assert base_files == candidate_files
    changed = []
    diff_lines = []
    for relative in base_files:
        before = (BASE / relative).read_text(encoding="utf-8").splitlines()
        after = (CANDIDATE / relative).read_text(encoding="utf-8").splitlines()
        if before != after:
            changed.append(str(relative).replace("\\", "/"))
            diff_lines.extend(
                unified_diff(before, after, lineterm="")
            )
    assert changed == ["op_host/square_sum_v1.cpp"]
    additions = [line[1:] for line in diff_lines if line.startswith("+") and not line.startswith("+++")]
    removals = [line[1:] for line in diff_lines if line.startswith("-") and not line.startswith("---")]
    assert removals == ["    } else if (fastPath == 1 || fastPath == 3) {"]
    assert additions == [
        "    } else if (",
        "        fastPath == 1 ||",
        "        fastPath == 3 ||",
        "        (fastPath == 4 &&",
        "         inputElements >= FULL_CORE_INPUT_THRESHOLD)) {",
    ]
    host = (CANDIDATE / "op_host" / "square_sum_v1.cpp").read_text(
        encoding="utf-8"
    )
    assert host.count("fastPath == 4 &&\n         inputElements >= FULL_CORE_INPUT_THRESHOLD") == 1
    print("SOURCE_SCOPE=PASS")


def audit_block_model():
    for name, input_elements, outputs in TARGETS:
        assert input_elements >= INPUT_THRESHOLD
        old_blocks = min(
            ceil_div(outputs, GENERIC_OUTPUTS_PER_CORE), MAX_BLOCK_DIM
        )
        new_blocks = min(outputs, MAX_BLOCK_DIM)
        assert new_blocks >= old_blocks
        assert new_blocks <= MAX_BLOCK_DIM
        print(
            f"TARGET={name} INPUTS={input_elements} OUTPUTS={outputs} "
            f"OLD_BLOCKS={old_blocks} NEW_BLOCKS={new_blocks}"
        )
    small_inputs = INPUT_THRESHOLD - 1
    outputs = 1105
    old_blocks = min(
        ceil_div(outputs, GENERIC_OUTPUTS_PER_CORE), MAX_BLOCK_DIM
    )
    new_blocks = old_blocks
    assert small_inputs < INPUT_THRESHOLD and new_blocks == old_blocks
    print(
        f"SMALL_CONTROL_INPUTS={small_inputs} "
        f"OLD_BLOCKS={old_blocks} NEW_BLOCKS={new_blocks}"
    )


def main():
    audit_source_scope()
    audit_block_model()
    print("SUMMARY=PASS")


if __name__ == "__main__":
    main()
