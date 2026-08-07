# SquareSumV1 S03G candidate

- Parent: `candidates/squaresum_s03e_s02f_splitk_singleton_20260807/SquareSumV1`.
- Change: raise the existing non-FP32 contiguous-last workspace Split-K output limit from 8 to 16.
- Newly selected domain: `fastPath == 1`, fp16/bf16, input at least `2^18`, reduction at least 2048, and 9..16 outputs.
- Independence: this route is disjoint from S03E's `fastPath == 3/4` additions and from singleton-gap routing.
- Arithmetic and kernel source: byte-identical to S03E.
- Cloud result: target matrix aggregate regression 0.4920%; worst point
  regression 7.8792%.
- Status: rejected after isolated NPU screen; never packaged or submitted.
