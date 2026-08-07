# S03M candidate origin

- Date: 2026-08-07
- Source baseline: S02F (`baselines/squaresum_s02f_global_best_20260806`)
- S02F formal platform result: `3223.995 us`
- Direction: clean grouped adjacent retained rows for large fastPath4 reductions
- Planned tiling key: 7
- Target environment: Ascend 910B, CANN community 8.5.0
- Internal gate status: not run
- Formal platform status: not submitted

The five operator source files in this initial snapshot are byte-identical to
S02F. S03J and S03K are deliberately excluded because their formal results
regressed. S03L is used only as an implementation reference for its isolated
key-7 grouped-row increment; none of its earlier accumulated changes may be
copied into this candidate.
