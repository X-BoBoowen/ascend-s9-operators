# SquareSumV1 S03B candidate

- Parent: `baselines/squaresum_s02f_global_best_20260806/SquareSumV1`
- Hypothesis: S02F leaves general fastPath4 workloads on at most `ceil(outputElements / 64)` cores. For large reductions with at most 1024 outputs, split the reduction across all 40 cores and tree-finalize FP32 partial sums.
- Scope: one new generic strided split-K kernel path and one conservative, shape-domain routing gate.
- Route gate: fastPath4, input elements at least `2^18`, reduction elements at least `2048`, and `1..1024` outputs.
- Status: local static and numerical-model gate required; NPU compile, correctness and performance are not yet claimed.

This candidate preserves S02F BF16 arithmetic and is intentionally independent of S03A.
