import hashlib
import math
from pathlib import Path

from validate_squaresum_profile_matrix_20260806 import (
    classify_route,
    normalize_axes,
    validate_shape,
)


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "candidates/squaresum_s03e_s02f_splitk_singleton_20260807/SquareSumV1"
CANDIDATE = ROOT / "candidates/squaresum_s03g_s03e_last_output16_20260807/SquareSumV1"
HOST = Path("op_host/square_sum_v1.cpp")
HOST_SHA256 = "FFCEA8B60BC21D6C487F048AC0E369D0174CE7C2A4EBA5EB3040F94F6C96B286"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def metadata(shape, axes, dtype):
    validate_shape(list(shape))
    normalized = normalize_axes(len(shape), axes)
    input_elements = math.prod(shape)
    reduce_elements = math.prod(shape[axis] for axis in normalized)
    return {
        "route": classify_route(shape, normalized),
        "input": input_elements,
        "reduce": reduce_elements,
        "output": input_elements // reduce_elements,
        "dtype": dtype,
    }


def selected(info):
    return (
        info["route"] == "fast1"
        and info["dtype"] in ("fp16", "bf16")
        and info["input"] >= 1 << 18
        and info["reduce"] >= 2048
        and 8 < info["output"] <= 16
    )


def source_contract():
    assert sha256(CANDIDATE / HOST) == HOST_SHA256
    base_host = (BASE / HOST).read_text(encoding="utf-8")
    candidate_host = (CANDIDATE / HOST).read_text(encoding="utf-8")
    assert candidate_host == base_host.replace(
        "WORKSPACE_LAST_MAX_OUTPUTS = 8;",
        "WORKSPACE_LAST_MAX_OUTPUTS = 16;",
        1,
    )
    for relative in (
        Path("op_host/CMakeLists.txt"),
        Path("op_host/square_sum_v1_tiling.h"),
        Path("op_kernel/CMakeLists.txt"),
        Path("op_kernel/square_sum_v1.cpp"),
    ):
        assert (BASE / relative).read_bytes() == (CANDIDATE / relative).read_bytes()


def main():
    source_contract()
    positives = (
        ((9, 64, 512), (1, 2), "fp16"),
        ((16, 64, 512), (-2, -1), "bf16"),
        ((4, 4, 1000, 32), (2, 3), "fp16"),
    )
    controls = (
        ((8, 64, 512), (1, 2), "fp16"),
        ((17, 64, 512), (1, 2), "bf16"),
        ((16, 64, 512), (1, 2), "fp32"),
        ((16, 32, 511), (1, 2), "fp16"),
        ((9, 64, 512, 2), (1, 2), "fp16"),
    )
    passed = 0
    for shape, axes, dtype in positives:
        info = metadata(shape, axes, dtype)
        assert selected(info), (shape, axes, dtype, info)
        print(f"S03G_ROUTE shape={shape} axes={axes} dtype={dtype} info={info} PASS")
        passed += 1
    for shape, axes, dtype in controls:
        info = metadata(shape, axes, dtype)
        assert not selected(info), (shape, axes, dtype, info)
        print(f"S03G_CONTROL shape={shape} axes={axes} dtype={dtype} info={info} PASS")
        passed += 1
    print(f"S03G_STATIC_SUMMARY passed={passed}/{passed} host_sha256={sha256(CANDIDATE / HOST)}")


if __name__ == "__main__":
    main()
