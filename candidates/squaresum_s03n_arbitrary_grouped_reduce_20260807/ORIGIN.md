# S03N origin

- Frozen baseline: `S03M` commit `b168d47`.
- Baseline source: `candidates/squaresum_s03m_s02f_strided_grouped_clean_20260807/SquareSumV1`.
- Clone check: the five competition files were byte-identical before S03N edits.
- S03M cloud evidence: build passed; correctness `1235/1235`; all 15 target case/dtype points improved by at least `69.7758%`, with FP16 aggregate `90.4032%` and BF16/FP32 aggregate `84.2814%`.
- S03N scope: remove only the power-of-two restriction on the innermost grouped reduction and accumulate descending power-of-two chunks in FP32.
- S03N must not change the aligned S03M route, add key 6, or add hidden-case identifiers.
