import difflib
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "baselines/squaresum_s02f_global_best_20260806/SquareSumV1"
CANDIDATE = ROOT / "candidates/squaresum_s03a_s02f_bf16_fused_20260806/SquareSumV1"
BASE_HOST_SHA256 = "6C4731323E66D3E7A02044FA214386D7C10167A0BA145C6AA8D3902535F5F367"
BASE_KERNEL_SHA256 = "B668B6A2217130E1878063713B3D03F663C7626B7C175FB6C8E503119002255A"
CANDIDATE_KERNEL_SHA256 = "DF77CC2711687E5C8611F38A8BF8F80CB40CBC8FC1E4D692CCB196B8CADADCC3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    base_host = BASE / "op_host/square_sum_v1.cpp"
    base_kernel = BASE / "op_kernel/square_sum_v1.cpp"
    candidate_kernel = CANDIDATE / "op_kernel/square_sum_v1.cpp"
    assert sha256(base_host) == BASE_HOST_SHA256
    assert sha256(base_kernel) == BASE_KERNEL_SHA256
    assert sha256(candidate_kernel) == CANDIDATE_KERNEL_SHA256

    for relative in (
        Path("op_host/CMakeLists.txt"),
        Path("op_host/square_sum_v1_tiling.h"),
        Path("op_host/square_sum_v1.cpp"),
        Path("op_kernel/CMakeLists.txt"),
    ):
        assert (BASE / relative).read_bytes() == (CANDIDATE / relative).read_bytes()

    before = base_kernel.read_text(encoding="utf-8")
    after = candidate_kernel.read_text(encoding="utf-8")
    diff = list(
        difflib.ndiff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
        )
    )
    additions = sum(line.startswith("+ ") for line in diff)
    deletions = sum(line.startswith("- ") for line in diff)
    assert additions == 0
    assert deletions == 80

    roundtrip = re.compile(
        r"AscendC::Cast\(\s*inputLocal,\s*(?:valueLocal|floatLocal),"
        r"\s*AscendC::RoundMode::CAST_RINT,\s*(?:inputCount|inputPadded|padded)\s*\);"
        r"\s*AscendC::Cast\(\s*(?:valueLocal|floatLocal),\s*inputLocal,"
        r"\s*AscendC::RoundMode::CAST_NONE,\s*(?:inputCount|inputPadded|padded)\s*\);",
        re.MULTILINE,
    )
    assert len(roundtrip.findall(before)) == 8
    assert len(roundtrip.findall(after)) == 0
    assert before.count("AscendC::RoundMode::CAST_RINT") == 15
    assert after.count("AscendC::RoundMode::CAST_RINT") == 7

    scalar_marker = "SquareInInputType<bfloat16_t>"
    scalar_end = "template <typename T>\n__aicore__ inline T OutputFromFloat"
    before_scalar = before[
        before.index(scalar_marker) : before.index(scalar_end)
    ]
    after_scalar = after[
        after.index(scalar_marker) : after.index(scalar_end)
    ]
    assert before_scalar == after_scalar

    print(
        "S03A_STATIC_SUMMARY passed=12/12 "
        f"deleted_lines={deletions} added_lines={additions} "
        f"base_kernel_sha256={sha256(base_kernel)} "
        f"candidate_kernel_sha256={sha256(candidate_kernel)}"
    )


if __name__ == "__main__":
    main()
