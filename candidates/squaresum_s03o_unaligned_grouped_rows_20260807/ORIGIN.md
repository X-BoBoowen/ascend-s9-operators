# S03O origin

- Frozen baseline: `S03N` commit `4b16d5d`.
- Baseline source: `candidates/squaresum_s03n_arbitrary_grouped_reduce_20260807/SquareSumV1`.
- Clone check: the five competition files were byte-identical before S03O edits.
- S03N cloud evidence: build passed; S03M regression correctness `105/105`; arbitrary-reduction correctness `45/45`. FP16 target points improved `76.1786%` to `85.2234%`; BF16/FP32 target points improved `61.7746%` to `83.8161%` in internal official-equivalent profiling.
- S03O scope: add key 8 for non-8-aligned small inner dimensions. The final implementation uses one contiguous DMA block per retained row, 32-byte-aligned UB row strides, hybrid vector reduction, and compact two-dimensional output.
- Final selective route: key 8 is enabled only for two-byte dtypes, non-8-aligned `inner <= 16`, and a power-of-two last reduce dimension. FP32, arbitrary reductions, larger inner dimensions, and capacity failures fall back to S03N.
- S03O must preserve aligned key 7 and all older routes, must not add key 6, and must not use hidden-case identifiers or fixed hidden shapes.
- Final CANN 8.5.0 / Ascend 910B evidence: build passed; complete correctness `1325/1325`; FP16/BF16 target A/B/A aggregate improvement `50.9727%`, minimum point improvement `50.5274%`, maximum baseline drift `0.2887%`; five focused fallback controls had maximum apparent regression `0.3094%`.
