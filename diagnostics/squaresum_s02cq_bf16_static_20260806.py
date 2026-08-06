import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02cq_bf16_fused_accum_20260806"
    / "SquareSumV1"
)
EXPECTED_HOST_SHA256 = (
    "0B5C6AEE3B01A63192A6CD4A59CF77A19373B6423274DC73968BAA77D84E9203"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    host = CANDIDATE / "op_host" / "square_sum_v1.cpp"
    kernel = CANDIDATE / "op_kernel" / "square_sum_v1.cpp"
    text = kernel.read_text(encoding="utf-8")

    actual_host_hash = sha256(host)
    if actual_host_hash != EXPECTED_HOST_SHA256:
        raise AssertionError(
            f"host changed: expected {EXPECTED_HOST_SHA256}, got {actual_host_hash}"
        )

    intermediate_roundtrip = re.compile(
        r"AscendC::Cast\(\s*(?:inputLocal|source),\s*"
        r"(?:valueLocal|floatLocal),\s*"
        r"AscendC::RoundMode::CAST_RINT,",
        re.MULTILINE,
    )
    matches = intermediate_roundtrip.findall(text)
    if matches:
        raise AssertionError(
            f"found {len(matches)} BF16 square-to-input intermediate casts"
        )

    final_rounds = text.count("AscendC::RoundMode::CAST_RINT")
    if final_rounds != 13:
        raise AssertionError(
            f"expected 13 final output CAST_RINT calls, found {final_rounds}"
        )

    if "__aicore__ inline void SquareIntoFloat" not in text:
        raise AssertionError("SquareIntoFloat helper is missing")

    print(
        "S02CQ_STATIC "
        f"host_sha256={actual_host_hash} "
        f"kernel_sha256={sha256(kernel)} "
        f"intermediate_roundtrips=0 final_cast_rint={final_rounds} PASS"
    )


if __name__ == "__main__":
    main()
