# S03K origin

- Baseline: S03J (`candidates/squaresum_s03j_s02f_fast2_small_inner_safe_20260807`).
- Change: for generic `fastPath2` reductions with at least 40 output rows,
  `reduce >= 2048`, `inner <= 8`, and at least 1 Mi elements, assign complete
  output rows to cores and replace repeated short 2-D DMA operations with a
  contiguous one-block transfer per tile.
- Cloud result: 1130/1130 correctness; 15 target A/B/A points aggregate
  improvement 84.7826%, minimum point improvement 50.8800%; 15 unmodified
  controls worst regression 2.0023%.
- Package: `提交相关材料/20260807/S03K/SquareSumV1.zip`.
- ZIP SHA-256: `9850faf54274057338b76a6ea3a125c5d03fddc2dffd2c9e2769f6e864ed0365`.
- Decision: packaged for official evaluation; no official score yet.
