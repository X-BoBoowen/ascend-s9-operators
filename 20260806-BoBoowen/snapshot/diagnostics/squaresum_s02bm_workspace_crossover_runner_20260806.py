import sys

from squaresum_s02bl_workspace_threshold_runner_20260806 import (
    DTYPES,
    measure,
)


CASES = (
    ("i2_r512", (1, 512, 2), (1,), False, False),
    ("i2_r1024", (1, 1024, 2), (1,), False, False),
    ("i2_r2047", (1, 2047, 2), (1,), False, False),
    ("i2_r2048", (1, 2048, 2), (1,), False, True),
    ("i2_r4096", (1, 4096, 2), (1,), False, True),
    ("i2_r8192", (1, 8192, 2), (1,), False, True),
    ("i4_r1024", (1, 1024, 4), (1,), False, False),
    ("i4_r2047", (1, 2047, 4), (1,), False, False),
    ("i4_r2048", (1, 2048, 4), (1,), False, True),
    ("i4_r4096", (1, 4096, 4), (1,), False, True),
    ("i4_r8192", (1, 8192, 4), (1,), False, True),
    ("i8_r1024", (1, 1024, 8), (1,), False, False),
    ("i8_r2047", (1, 2047, 8), (1,), False, False),
    ("i8_r2048", (1, 2048, 8), (1,), False, True),
    ("i8_r4096", (1, 4096, 8), (1,), False, True),
    ("i8_r8192", (1, 8192, 8), (1,), False, True),
)


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    passed = 0
    for case in CASES:
        for dtype_name, dtype, type_bytes in DTYPES:
            measure(label, case, dtype_name, dtype, type_bytes)
            passed += 1
    print(f"SUMMARY label={label} passed={passed}/{len(CASES) * len(DTYPES)}")


if __name__ == "__main__":
    main()
