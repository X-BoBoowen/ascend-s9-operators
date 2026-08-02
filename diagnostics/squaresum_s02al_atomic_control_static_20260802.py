from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02ak_grouped_short_narrow_20260802"
    / "SquareSumV1"
)
ATOMIC_CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02al_atomic_control_20260802"
    / "SquareSumV1"
)


def main():
    workspace_host = (
        WORKSPACE_CANDIDATE / "op_host" / "square_sum_v1.cpp"
    ).read_text(encoding="utf-8")
    atomic_host = (
        ATOMIC_CANDIDATE / "op_host" / "square_sum_v1.cpp"
    ).read_text(encoding="utf-8")
    workspace_kernel = (
        WORKSPACE_CANDIDATE / "op_kernel" / "square_sum_v1.cpp"
    ).read_bytes()
    atomic_kernel = (
        ATOMIC_CANDIDATE / "op_kernel" / "square_sum_v1.cpp"
    ).read_bytes()
    assert workspace_kernel == atomic_kernel
    assert "const bool atomicReduce = false;" in workspace_host
    assert "ATOMIC_REDUCE_INPUT_THRESHOLD" not in workspace_host
    assert "ATOMIC_REDUCE_INPUT_THRESHOLD" in atomic_host
    assert (
        "inputDesc->GetDataType() == ge::DT_FLOAT &&\n"
        "        fastPath == 1"
        in atomic_host
    )
    assert (
        "inputDesc->GetDataType() != ge::DT_FLOAT &&\n"
        "          outputElements <= WORKSPACE_LAST_MAX_OUTPUTS"
        in atomic_host
    )
    assert atomic_host.count("{") == atomic_host.count("}")
    assert atomic_host.count("(") == atomic_host.count(")")

    eligible = 0
    for output_elements in range(1, 9):
        for reduce_elements in range(2048, 1 << 20):
            if output_elements * reduce_elements >= 1 << 18:
                eligible += 1
    assert eligible == 7_676_136
    print(f"WORKSPACE_CANDIDATE={WORKSPACE_CANDIDATE}")
    print(f"ATOMIC_CANDIDATE={ATOMIC_CANDIDATE}")
    print(f"FP32_AB_CONFIGS={eligible}")
    print("KERNELS_BYTE_IDENTICAL=true")
    print("SUMMARY: S02AL atomic-control isolation passed")


if __name__ == "__main__":
    main()
