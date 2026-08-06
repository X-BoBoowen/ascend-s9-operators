# SquareSumV1 S02CT local audit

Updated: 2026-08-06 (Asia/Shanghai)

## Result

S02CT addresses a second routing hole found by exhaustive low-parallel review:
S02CA uses 40-block workspace Split-K for large contiguous last-axis/suffix
reductions only while output elements are at most eight. With 9..16 outputs,
the old path normally launches one block per output and each block traverses the
entire reduction, despite the existing Kernel and workspace finalizer already
being able to hold and merge these partial vectors.

S02CT changes one Host constant from 8 to 16. The Kernel and all CMake/tiling
files are byte-identical to formal S02CA. This is a local candidate only and has
no claimed NPU speedup.

## Why the route is bounded at 16

The candidate keeps S02CA's existing gates: input at least `2^18`, reduction at
least 2048, `fastPath=1`, 40 blocks and tree finalization. In the newly selected
domain, the input threshold and output cap imply a practical minimum reduction
of 16384 elements. The old critical block traverses one full reduction. The new
critical block traverses every output but approximately one fortieth of the
reduction, giving the following conservative work ratio:

```text
old/new ~= reduce / (outputs * ceil(reduce / 40))
```

The local positive boundaries produce `2.498x..4.440x`. Extending beyond 16
would reduce the modeled margin below 2.5x while retaining synchronization and
finalization overhead, so it is deliberately excluded pending real profiling.

The largest candidate partial vector is 16 FP32 values. S02CA's smallest
reinterpreted output buffer holds 512 FP32 values, and the existing workspace
stride/finalizer already supports substantially larger fastPath2 outputs. No
Kernel memory bound is widened.

## Local verification

- Source and route contract: `10/10` positive/boundary checks.
- Published-domain matrix: `31/31` after adding three S02CT profile layouts.
- FP16/BF16/FP32 Split-K numerical model: `15/15` under the supplied official
  tolerance rule.
- Candidate Kernel SHA-256 exactly matches formal S02CA.
- Candidate source diff is exactly one Host line.
- Python syntax and cloud-gate shell syntax pass.

Controls cover output 8, output 17, input just below `2^18`, fastPath2 and the
small public Case1. Negative/unsorted axes, keep_dims and rank-4 contiguous
suffixes are included in the selected correctness/profile matrix.

## Required cloud gate

Run `diagnostics/run_squaresum_s02ct_cloud_gate_20260806.sh` with a clean
official project template and a new work directory. It independently builds
and installs S02CA/S02CT, runs 27 candidate correctness cases, records all three
dtypes on the `last_output_splitk` tier, and performs official-compatible
S02CA/S02CT/S02CA A/B/A profiling. It rejects aggregate improvement below 10%,
any per-case regression above 3%, or baseline drift above 3%.

Do not package or submit S02CT before the CANN community 8.5.0 gate passes.
