# SquareSumV1 S03F candidate

- Parent: `candidates/squaresum_s03e_s02f_splitk_singleton_20260807/SquareSumV1`.
- Change: route conservative low-parallel `fastPath == 3` workloads with trailing reduced suffix `1..1023`, at most 16 outputs, at least 40 natural rows, input at least `2^18`, and reduction at least `2^15` through S03E's already validated generic 40-core Split-K kernel.
- Cost guard: the modeled old critical natural-row count must be at least twice the quotient/remainder Split-K row-output work.
- Independence: the new route is `fastPath == 3`; S03E's existing Split-K route is `fastPath == 4`, and singleton-gap routes are `fastPath == 1/2`.
- Arithmetic and kernel source: byte-identical to S03E.
- Cloud result: target matrix aggregate regression 217.6840%; worst point
  regression 1134.8041%.
- Status: rejected after isolated NPU screen; never packaged or submitted.
