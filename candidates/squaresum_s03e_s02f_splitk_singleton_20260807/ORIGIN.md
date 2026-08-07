# SquareSumV1 S03E candidate

- Parent: `candidates/squaresum_s03b_s02f_fast4_splitk_20260806/SquareSumV1`.
- Combined changes:
  - S03B: use a 40-core workspace/tree-finalized split-K path for large general-strided reductions with at most 1024 outputs.
  - S03D: treat size-one gaps between reduced axes as physically contiguous, subject to the proven conservative route guard.
- Independence: the S03B and S03D host routes are disjoint (`fastPath == 4` versus physical-contiguous `fastPath == 1/2`), so neither changes the other's tiling decision.
- Arithmetic: unchanged from S02F/S03B; this candidate does not include the still-unsettled S03A BF16 arithmetic experiment.
- Validation status (Ascend 910B4, CANN community 8.5.0, 2026-08-07):
  - isolated build/install/import passed;
  - combined correctness passed `126/126`;
  - general-strided Split-K A/B/A aggregate improvement: `86.7365%`;
  - singleton-gap A/B/A aggregate improvement: `84.5746%`;
  - the only first-pass baseline drift failure (`fp32 fast2`, `3.5813%`) was repeated independently and passed with `0.1849%` drift and `78.7486%` improvement;
  - promoted to `submission-src/SquareSumV1` and packaged for platform evaluation; no official hidden-case score is claimed yet.
