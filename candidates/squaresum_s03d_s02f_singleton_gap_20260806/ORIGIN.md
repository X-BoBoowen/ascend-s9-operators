# SquareSumV1 S03D candidate

- Parent: `baselines/squaresum_s02f_global_best_20260806/SquareSumV1`
- Hypothesis: physically contiguous reduction groups separated only by a size-one retained dimension can use S02F's contiguous fastPath1/2 implementation. The promotion is retained only for reductions of at least 8192 elements when a suffix output remains, matching the later cloud-validated safety threshold.
- Scope: host metadata classification only; kernel and all build files are byte-identical to S02F.
- Motivation: S02F is the true global performance baseline, while the later S02BA lineage improved official Case3 by about 99 us but regressed Case4 by about 834 us. This isolates one plausible, previously validated Case3 improvement without importing the broad S02BA kernel/routing rewrite.
- Status: local route/coordinate gate required; NPU performance and correctness are not yet claimed.
