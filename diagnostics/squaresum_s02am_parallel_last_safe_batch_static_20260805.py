from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "candidates"
    / "squaresum_s02am_parallel_last_safe_batch_20260805"
    / "SquareSumV1"
)
HOST = CANDIDATE / "op_host" / "square_sum_v1.cpp"
KERNEL = CANDIDATE / "op_kernel" / "square_sum_v1.cpp"
MAX_OUTPUTS = 8
TILE_OUTPUTS = 1024


def function_body(source, signature, next_signature):
    start = source.index(signature)
    end = source.index(next_signature, start)
    return source[start:end]


def audit_source():
    host = HOST.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")
    assert host.count("{") == host.count("}")
    assert kernel.count("{") == kernel.count("}")
    assert "WORKSPACE_LAST_MAX_OUTPUTS = 8" in host
    assert "constexpr uint32_t TILE_OUTPUTS = 1024" in kernel

    body = function_body(
        kernel,
        "void ProcessParallelLast(",
        "void ProcessParallelMiddle(",
    )
    assert "outputBuffer_.Get<float>()" in body
    assert "floatBuffer_.Get<float>()" not in body
    assert body.count("AscendC::DataCopyPad(") == 1
    assert "outputElements_ * sizeof(float)" in body

    required_bytes = MAX_OUTPUTS * np.dtype(np.float32).itemsize
    for element_bytes in (2, 4):
        output_buffer_bytes = TILE_OUTPUTS * element_bytes
        assert output_buffer_bytes >= required_bytes


def audit_clobber_model():
    for outputs in range(1, MAX_OUTPUTS + 1):
        expected = np.arange(1, outputs + 1, dtype=np.float32)

        unsafe = np.zeros(TILE_OUTPUTS, dtype=np.float32)
        for output in range(outputs):
            unsafe[:64] = -1.0
            unsafe[output] = expected[output]

        safe = np.zeros(TILE_OUTPUTS, dtype=np.float32)
        conversion = np.zeros(TILE_OUTPUTS, dtype=np.float32)
        for output in range(outputs):
            conversion[:64] = -1.0
            safe[output] = expected[output]

        np.testing.assert_array_equal(safe[:outputs], expected)
        if outputs > 1:
            assert not np.array_equal(unsafe[:outputs], expected)


def main():
    audit_source()
    audit_clobber_model()
    print(f"HOST={HOST}")
    print(f"KERNEL={KERNEL}")
    print("MAX_PARTIAL_BYTES=32")
    print("MIN_OUTPUT_BUFFER_BYTES=2048")
    print("SUMMARY: S02AM safe parallel-last batch buffer passed")


if __name__ == "__main__":
    main()
