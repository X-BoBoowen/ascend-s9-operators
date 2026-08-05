import hashlib
import itertools
from pathlib import Path

from squaresum_s02bd_trailing_singleton_static_20260805 import metadata


ROOT = Path(__file__).resolve().parents[1]
BD = ROOT / (
    "candidates/squaresum_s02bd_trailing_singleton_last_20260805/"
    "SquareSumV1"
)
BE = ROOT / (
    "candidates/squaresum_s02be_empty_reduce_zero_20260805/"
    "SquareSumV1"
)
KERNEL = Path("op_kernel/square_sum_v1.cpp")
TILE_OUTPUTS = 1024


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def source_guards():
    before = (BD / KERNEL).read_text(encoding="utf-8")
    after = (BE / KERNEL).read_text(encoding="utf-8")
    process_point = """    __aicore__ inline void Process()
    {
        if (reduceMode_ == 2U) {"""
    process_guard = """    __aicore__ inline void Process()
    {
        if (reduceElements_ == 0U) {
            ProcessEmptyReduction();
            return;
        }
        if (reduceMode_ == 2U) {"""
    empty_method = """    __aicore__ inline void ProcessEmptyReduction()
    {
        AscendC::LocalTensor<T> outputLocal =
            outputBuffer_.Get<T>();
        const uint32_t elementsPerBlock =
            32U / sizeof(T);

        for (uint64_t tileOffset = 0;
             tileOffset < outputs_;
             tileOffset += TILE_OUTPUTS) {
            const uint32_t current = static_cast<uint32_t>(
                outputs_ - tileOffset < TILE_OUTPUTS
                    ? outputs_ - tileOffset
                    : TILE_OUTPUTS);
            const uint32_t padded =
                (current + elementsPerBlock - 1U) /
                elementsPerBlock * elementsPerBlock;
            AscendC::Duplicate(
                outputLocal,
                static_cast<T>(0),
                padded);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                vToMte3Event_);

            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = current * sizeof(T);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(
                outputGm_[firstOutput_ + tileOffset],
                outputLocal,
                copyParams);
            if (tileOffset + current < outputs_) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                    mte3ToVEvent_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                    mte3ToVEvent_);
            }
        }
    }

"""
    expected = before.replace(process_point, process_guard, 1).replace(
        "private:\n    __aicore__ inline void ProcessParallelReduction()",
        "private:\n"
        + empty_method
        + "    __aicore__ inline void ProcessParallelReduction()",
        1,
    )
    assert after == expected

    before_files = {
        path.relative_to(BD) for path in BD.rglob("*") if path.is_file()
    }
    after_files = {
        path.relative_to(BE) for path in BE.rglob("*") if path.is_file()
    }
    assert before_files == after_files
    for relative in before_files - {KERNEL}:
        assert sha256(BD / relative) == sha256(BE / relative)


def block_dim(output_elements, fast_path):
    if output_elements == 0:
        return 1
    if fast_path in (1, 3):
        desired = min(output_elements, 32)
    else:
        desired = (output_elements + 63) // 64
    return min(max(desired, 1), 40)


def check_schedule(output_elements, fast_path, type_bytes):
    blocks = block_dim(output_elements, fast_path)
    if output_elements == 0:
        return 0
    base = output_elements // blocks
    extra = output_elements % blocks
    visited = []
    tiles = 0
    for block in range(blocks):
        outputs = base + int(block < extra)
        first = block * base + min(block, extra)
        for tile_offset in range(0, outputs, TILE_OUTPUTS):
            current = min(outputs - tile_offset, TILE_OUTPUTS)
            elements_per_block = 32 // type_bytes
            padded = (
                (current + elements_per_block - 1)
                // elements_per_block
                * elements_per_block
            )
            assert 0 < current <= TILE_OUTPUTS
            assert current <= padded <= TILE_OUTPUTS
            visited.extend(range(first + tile_offset, first + tile_offset + current))
            tiles += 1
    assert visited == list(range(output_elements))
    return tiles


def exhaustive_empty_audit():
    layouts = 0
    skipped_empty_outputs = 0
    scheduled_outputs = 0
    tiles = 0
    fast_paths = set()
    for rank in range(1, 6):
        for shape in itertools.product((0, 1, 2, 3), repeat=rank):
            for mask in range(1, 1 << rank):
                axes = tuple(
                    axis for axis in range(rank) if mask & (1 << axis)
                )
                info = metadata(shape, axes, True)
                if info["reduce_elements"] != 0:
                    continue
                layouts += 1
                fast_paths.add(info["fast_path"])
                if info["output_elements"] == 0:
                    skipped_empty_outputs += 1
                    continue
                scheduled_outputs += info["output_elements"]
                for type_bytes in (2, 4):
                    tiles += check_schedule(
                        info["output_elements"],
                        info["fast_path"],
                        type_bytes,
                    )
    assert layouts > 0
    assert skipped_empty_outputs > 0
    assert scheduled_outputs > 0
    assert fast_paths == {1, 2, 3, 4}
    return layouts, skipped_empty_outputs, scheduled_outputs, tiles


def large_schedule_audit():
    outputs = (0, 1, 7, 8, 9, 63, 64, 65, 1023, 1024, 1025, 40960, 40961, 100000)
    checks = 0
    for output_elements, fast_path, type_bytes in itertools.product(
        outputs, (1, 2, 3, 4), (2, 4)
    ):
        check_schedule(output_elements, fast_path, type_bytes)
        checks += 1
    return checks


def main():
    source_guards()
    layouts, skipped, outputs, tiles = exhaustive_empty_audit()
    large = large_schedule_audit()
    print("SOURCE_GUARDS=PASS")
    print(f"EMPTY_LAYOUTS={layouts}")
    print(f"EMPTY_OUTPUT_LAYOUTS_SKIPPED={skipped}")
    print(f"SCHEDULED_OUTPUTS={outputs}")
    print(f"SCHEDULED_TILES_BY_DTYPE_CLASS={tiles}")
    print(f"LARGE_SCHEDULE_CHECKS={large}")
    print("FAST_PATHS=1,2,3,4")
    print("SUMMARY: S02BE empty-reduction static audit passed")


if __name__ == "__main__":
    main()
