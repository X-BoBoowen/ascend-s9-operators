# S03H origin

- Baseline: S02F (`candidates/squaresum_s02f_grouped_vector8_20260730_1603`)
- Change: route generic `fastPath2`, `reduce>=2048`, `inner<=64` to the existing
  long-chunk key.
- Cloud result: FP16/BF16 correct without stable gain; FP32 only 120/128 correct.
- Decision: rejected, never packaged or submitted.
