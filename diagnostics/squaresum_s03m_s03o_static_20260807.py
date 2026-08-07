import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RouteDecision:
    enabled: bool
    key: int
    grouped_width: int
    scheduled_tasks: int
    aligned_inner: int


DISABLED = RouteDecision(False, 0, 0, 0, 0)


def route(
    stage,
    *,
    input_elements,
    output_elements,
    reduce_elements,
    reduce_rank,
    last_reduce,
    inner,
    grouped_dim,
    outer_rows,
    input_type_bytes,
    fast_path=4,
    reduce_mode=0,
    last_reduce_stride_matches=True,
):
    if inner <= 0 or input_type_bytes not in (2, 4):
        return DISABLED
    elements_per_block = 32 // input_type_bytes
    aligned_inner = (
        (inner + elements_per_block - 1) // elements_per_block
        * elements_per_block
    )
    power_of_two = last_reduce > 1 and not (
        last_reduce & (last_reduce - 1)
    )
    allow_arbitrary = stage in {"s03n", "s03o"}
    padded = stage == "s03o" and inner % 8 != 0
    physical_row = last_reduce * inner
    aligned_physical_row = (
        (physical_row + elements_per_block - 1) // elements_per_block
        * elements_per_block
    )
    base_ok = (
        stage in {"s03m", "s03n", "s03o"}
        and fast_path == 4
        and reduce_mode == 0
        and input_elements >= (1 << 20)
        and reduce_elements >= 2048
        and reduce_rank > 1
        and last_reduce > 1
        and reduce_elements % last_reduce == 0
        and output_elements % inner == 0
        and last_reduce_stride_matches
        and (allow_arbitrary or power_of_two)
        and (padded or inner % 8 == 0)
        and (
            not padded
            or (
                power_of_two
                and inner <= 16
                and input_type_bytes == 2
            )
        )
    )
    if not base_ok:
        return RouteDecision(False, 0, 0, 0, aligned_inner)

    for width in (8, 4, 2):
        tasks = outer_rows * ((grouped_dim + width - 1) // width)
        block_count = width if padded else width * last_reduce
        buffer_elements = width * (
            aligned_physical_row if padded else physical_row
        )
        buffer_limit = 16384 if padded else 8192
        if (
            grouped_dim >= width
            and buffer_elements <= buffer_limit
            and width * inner <= 1024
            and width * aligned_inner <= 1024
            and tasks >= 32
            and block_count <= 65535
        ):
            return RouteDecision(
                True,
                8 if padded else 7,
                width,
                tasks,
                aligned_inner,
            )
    return RouteDecision(False, 0, 0, 0, aligned_inner)


def expect(stage, expected, **values):
    actual = route(stage, **values)
    if actual != expected:
        raise AssertionError(
            f"stage={stage} values={values} expected={expected} "
            f"actual={actual}"
        )


def common(**overrides):
    values = {
        "input_elements": 33_554_432,
        "output_elements": 4_096,
        "reduce_elements": 8_192,
        "reduce_rank": 2,
        "last_reduce": 128,
        "inner": 16,
        "grouped_dim": 32,
        "outer_rows": 8,
        "input_type_bytes": 2,
    }
    values.update(overrides)
    return values


def run_model_checks(stage):
    if stage == "s03m":
        expect(
            stage,
            RouteDecision(True, 7, 4, 64, 16),
            **common(),
        )
        expect(
            stage,
            RouteDecision(True, 7, 8, 32, 16),
            **common(
                input_elements=16_777_216,
                output_elements=4_096,
                reduce_elements=4_096,
                last_reduce=64,
                grouped_dim=64,
                outer_rows=4,
            ),
        )
        expect(
            stage,
            RouteDecision(True, 7, 4, 64, 16),
            **common(
                input_elements=16_777_216,
                reduce_elements=8_192,
                last_reduce=256,
                inner=8,
            ),
        )
        expect(
            stage,
            RouteDecision(True, 7, 2, 128, 16),
            **common(
                input_elements=33_554_432,
                reduce_elements=16_384,
                last_reduce=512,
                inner=8,
            ),
        )
        for overrides in (
            {"last_reduce": 127, "reduce_elements": 8_128},
            {"inner": 15, "output_elements": 3_840},
            {"grouped_dim": 2, "outer_rows": 31},
            {"input_elements": (1 << 20) - 1},
            {"reduce_elements": 2_047},
            {"fast_path": 3},
            {"reduce_mode": 2},
            {"last_reduce_stride_matches": False},
            {"last_reduce": 1},
            {"last_reduce": 1_024, "inner": 16},
        ):
            actual = route(stage, **common(**overrides))
            if actual.enabled:
                raise AssertionError(
                    f"S03M fallback case routed unexpectedly: {overrides}"
                )
    elif stage == "s03n":
        expected = {
            3: 8,
            7: 8,
            31: 8,
            33: 8,
            63: 8,
            65: 4,
            127: 4,
            129: 2,
        }
        for last_reduce, width in expected.items():
            reduce_elements = last_reduce * 1_024
            expect(
                stage,
                RouteDecision(
                    True,
                    7,
                    width,
                    8 * ((32 + width - 1) // width),
                    16,
                ),
                **common(
                    input_elements=max(1 << 20, reduce_elements * 4_096),
                    reduce_elements=reduce_elements,
                    last_reduce=last_reduce,
                ),
            )
    elif stage == "s03o":
        expect(
            stage,
            RouteDecision(True, 8, 8, 32, 16),
            **common(inner=15, output_elements=3_840),
        )
        expect(
            stage,
            RouteDecision(False, 0, 0, 0, 24),
            **common(
                inner=17,
                output_elements=4_352,
                input_type_bytes=4,
            ),
        )
        expect(
            stage,
            RouteDecision(False, 0, 0, 0, 208),
            **common(
                input_elements=117_374_976,
                output_elements=50_944,
                reduce_elements=2_304,
                last_reduce=3,
                inner=199,
                grouped_dim=32,
                outer_rows=8,
            ),
        )
        expect(
            stage,
            RouteDecision(True, 7, 4, 64, 16),
            **common(),
        )
        expect(
            stage,
            RouteDecision(True, 8, 8, 32, 16),
            **common(inner=7, output_elements=1_792),
        )
        expect(
            stage,
            RouteDecision(False, 0, 0, 0, 16),
            **common(
                inner=15,
                output_elements=3_840,
                input_type_bytes=4,
            ),
        )
        expect(
            stage,
            RouteDecision(False, 0, 0, 0, 16),
            **common(
                last_reduce=127,
                reduce_elements=8_128,
                inner=15,
                output_elements=3_840,
            ),
        )
    else:
        raise AssertionError(f"unknown stage: {stage}")


def audit_source(stage, source):
    source = source.resolve(strict=True)
    host = (source / "op_host/square_sum_v1.cpp").read_text(
        encoding="utf-8"
    )
    kernel = (source / "op_kernel/square_sum_v1.cpp").read_text(
        encoding="utf-8"
    )
    cmake = (source / "op_kernel/CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    required = {
        "s03m": (
            (host, "stridedGroupedRows"),
            (kernel, "ProcessStridedGroupedRows"),
            (kernel, "TILING_KEY_IS(7)"),
            (cmake, "--tiling_key=1,2,3,4,5,7"),
        ),
        "s03n": (
            (host, "stridedGroupedRows"),
            (kernel, "ReduceArbitraryRowsInto"),
            (kernel, "TILING_KEY_IS(7)"),
            (cmake, "--tiling_key=1,2,3,4,5,7"),
        ),
        "s03o": (
            (host, "stridedGroupedPaddedRows"),
            (host, "stridedGroupedSmallInner"),
            (host, "stridedGroupedNarrowType"),
            (kernel, "ProcessStridedGroupedPaddedRows"),
            (kernel, "ReduceUnalignedRowsInto"),
            (kernel, "TILING_KEY_IS(8)"),
            (cmake, "--tiling_key=1,2,3,4,5,7,8"),
        ),
    }
    for text, marker in required[stage]:
        if marker not in text:
            raise AssertionError(f"missing source audit marker: {marker}")
    if stage == "s03m":
        for forbidden in (
            "middleFullRows",
            "ProcessMiddleFullRows",
            "TILING_KEY_IS(6)",
        ):
            if forbidden in host or forbidden in kernel:
                raise AssertionError(
                    f"S03M contains rejected S03K marker: {forbidden}"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True, choices=("s03m", "s03n", "s03o")
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model-only", action="store_true")
    mode.add_argument("--source", type=Path)
    args = parser.parse_args()

    run_model_checks(args.stage)
    if args.source is not None:
        audit_source(args.stage, args.source)
    print(f"SQUARESUM_{args.stage.upper()}_STATIC_AUDIT_PASS")


if __name__ == "__main__":
    main()
