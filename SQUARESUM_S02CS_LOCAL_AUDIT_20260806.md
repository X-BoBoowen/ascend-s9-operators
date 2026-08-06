# SquareSumV1 S02CS local audit

Updated: 2026-08-06 (Asia/Shanghai)

## Result

S02CS is a local-only candidate for one coverage hole left by S02CA/S02BI:
noncontiguous `fastPath=3` reductions whose naturally contiguous reduced suffix
is shorter than 1024 elements. S02BI accelerated suffixes `1024..16384` by
Split-K, but the same already-implemented kernel mode was not selected for
suffixes `1..1023`. Official S02BI changed `3985.330` to `3982.789 us`, which
is noise-level evidence that its old domain did not cover the hidden hot cases.

S02CS changes only `op_host/square_sum_v1.cpp`; the kernel and build files are
byte-identical to formal S02CA. It is not a release and has no claimed NPU
speedup yet.

## Specification and measurement re-audit

The S9 spreadsheet defines SquareSumV1 as
`torch.sum(torch.square(input), dim=axis, keepdim=keep_dims)`, with rank up to
five in the published shape notation, FP16/BF16/FP32, unaligned dimensions and
the following right-to-left limits: `N<=10000`, `N2<=10000`, `N3<=1000`,
`N4<=200`. No fixed performance shapes are disclosed.

The supplied test extension invokes 30 SquareSumV1 operations, each preceded
by an FP32 `4096x4096` `aclnnMul`. Its parser removes rows whose op name contains
`aclnnMul`, then takes the median of filtered task durations at positions
10..29. Consequently, Mul is profiler spacing rather than score work, and a
candidate must reduce the actual SquareSumV1 kernel task duration. Event timing
is retained only to discover affected public layouts.

Cases 2 and 3 accept torch tensors directly while the other supplied cases are
constructed from NumPy. This permits BF16 data in those two cases, but does not
prove their hidden dtype or shape and is not used as a route condition.

The CANN 8.5.0 Ascend C documentation confirms that
[`DataCopyPad`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0265.html)
with extended parameters and hard
[`SyncAll`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0204.html)
are supported on Atlas A2. It also requires the logical block dimension used by
`SyncAll` not to exceed physical cores. S02CS keeps S02CA's existing 40-block
limit and its already cloud-proven synchronization implementation.

## Route and cost gate

The added route requires all of the following:

- `fastPath == 3`;
- input elements at least `2^18` and reduction elements at least `2^15`;
- output elements in `1..16`;
- trailing contiguous reduced suffix in `1..1023`;
- at least 40 natural reduction rows;
- a valid grouped suffix layout and a source gap representable by the existing
  `DataCopyPad` parameters;
- the old critical natural-row count is at least twice the exact Split-K
  row-output cost after quotient/remainder partitioning.

The old path can launch at most one task per output (and sometimes fewer after
grouping), while every task traverses all natural reduction rows. S02CS launches
40 blocks and each block traverses all outputs but only its row share. For the
positive local boundary models, the conservative critical-row ratio is
`2.5x..5.0x`. The model does not count launch, synchronization or DMA command
overhead, so it is only a selection safety argument, not a performance claim.

The existing kernel caps each two-dimensional DMA to 4095 rows, splits at the
physical batch-axis boundary, pads by less than one 32-byte block, writes an
FP32 partial for every block and uses the existing tree finalizer. The candidate
output cap of 16 is far below the 512-FP32-element capacity of the smallest
reinterpreted output buffer.

## Local verification

- Published-domain matrix: `32/32`.
- Route/source contract and boundary controls: `14/14`.
- FP16/BF16/FP32 Split-K numerical model: `21/21` under the supplied official
  tolerance rule.
- Candidate kernel SHA-256 is exactly the formal S02CA kernel SHA-256.
- Python syntax and cloud-gate shell syntax pass.

The controls explicitly reject tail 1024 (the old S02BI domain), 39 rows,
output 17, a 41-row/output16 quotient-rounding loss, a below-threshold
reduction, contiguous fastPath1 and strided fastPath4.

## Required cloud gate

Run `diagnostics/run_squaresum_s02cs_cloud_gate_20260806.sh` with a clean
official project template and a new work directory. The gate builds S02CA and
S02CS from source into separate projects, installs them into isolated roots,
runs 30 candidate correctness checks, discovers all three dtypes on the
`short_tail_splitk` tier, and performs official-compatible S02CA/S02CS/S02CA
A/B/A profiling. It rejects less than 10% aggregate improvement, more than 3%
per-case regression, or more than 3% baseline drift.

Do not package or submit S02CS before this gate passes on CANN community 8.5.0.
