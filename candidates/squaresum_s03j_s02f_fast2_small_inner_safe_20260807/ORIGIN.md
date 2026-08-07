# S03J origin

- Baseline: S02F (`candidates/squaresum_s02f_grouped_vector8_20260730_1603`).
- Change: S03I middle-tree implementation plus a 4096-byte FP32 long-key
  finalizer buffer; workspace tree finalize retains S02F key 3.
- Cloud result: 1061/1061 correctness; six target A/B/A points aggregate
  improvement 34.1427%; unmodified controls worst regression 2.3903%.
- Package: `提交相关材料/20260807/S03J/SquareSumV1.zip`.
- Decision: packaged for official evaluation; no official score yet.
