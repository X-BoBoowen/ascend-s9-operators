# SquareSumV1 S03A candidate

- Parent: `baselines/squaresum_s02f_global_best_20260806/SquareSumV1`
- Hypothesis: remove eight BF16 intermediate `FP32 -> BF16 -> FP32` round trips after squaring, while preserving every final BF16 `CAST_RINT` output conversion.
- Scope: kernel only, exactly 80 deleted lines and no added lines relative to S02F.
- Status: local static and numerical-model gate required; NPU compile, correctness and performance are not yet claimed.

This candidate is intentionally independent of the fast-path routing experiments so that an A/B result has one cause.
