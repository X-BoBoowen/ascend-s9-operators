# S03L candidate origin

- Date: 2026-08-07
- Baseline: S03K (`bbd36b2ee3eecfa82633354beefb6c6a63fca15f`)
- Direction: generic grouped adjacent rows for large rank-5 fastPath4 reductions
- New tiling key: 7
- Cloud: Ascend 910B, CANN 8.5.0
- Target A/B/A aggregate improvement: 89.6064%
- Minimum target-point improvement: 79.2200%
- Control maximum observed regression: 7.2830%
- Correctness: 1235/1235 legal public-constraint cases
- Submission ZIP SHA-256: `3a61968292fde3b3cc876b4dcac0e585a9bc039683ccd0767ff17522dd158d4f`
- Formal platform status: not yet submitted / no five-Case result

The source snapshot is byte-identical to `submission-src/SquareSumV1` at the
time of packaging.  Compact machine-readable summaries are in `evidence/`;
the full analysis is in `SQUARESUM_S03L_CLOUD_GATE_20260807.md`.
