# S03O origin

- Frozen baseline: `S03N` commit `4b16d5d`.
- Baseline source: `candidates/squaresum_s03n_arbitrary_grouped_reduce_20260807/SquareSumV1`.
- Clone check: the five competition files were byte-identical before S03O edits.
- S03N cloud evidence: build passed; S03M regression correctness `105/105`; arbitrary-reduction correctness `45/45`. FP16 target points improved `76.1786%` to `85.2234%`; BF16/FP32 target points improved `61.7746%` to `83.8161%` in internal official-equivalent profiling.
- S03O scope: add key 8 only for non-8-aligned inner dimensions, using two-dimensional `DataCopyPad` into 32-byte-aligned UB rows before the existing arbitrary-row FP32 reduction.
- S03O must preserve aligned key 7 and all older routes, must not add key 6, and must not use hidden-case identifiers or fixed hidden shapes.
