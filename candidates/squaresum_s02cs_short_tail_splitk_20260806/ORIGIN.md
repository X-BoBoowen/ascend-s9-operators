# S02CS origin

- Baseline: `baselines/squaresum_s02ca_formal_best_20260806/SquareSumV1`
- Baseline host SHA-256: `0B5C6AEE3B01A63192A6CD4A59CF77A19373B6423274DC73968BAA77D84E9203`
- Baseline kernel SHA-256: `C6EA5927D44DDF905E50B309C08E811E9DF614B7018D47F2BEB6A71115EC4C80`
- Candidate host SHA-256: `4BFA7659C292092AAAF3C012A88583FE121243698EDBB3F4702B1B15C5315436`
- Candidate kernel SHA-256: `C6EA5927D44DDF905E50B309C08E811E9DF614B7018D47F2BEB6A71115EC4C80`

S02CS is an independent one-file host-routing experiment. It does not include
S02CQ or S02CR. The kernel is byte-identical to S02CA.

The new route reuses S02CA's already cloud-proven `reduceMode=3` grouped-suffix
workspace Split-K implementation for a previously excluded public domain:
`fastPath=3`, trailing reduced suffix `1..1023`, output `1..8`, at least 40
natural reduction rows, input at least `2^18`, and reduction at least `2^15`.
All conditions depend only on shape, axes, dtype-derived byte width and strides.
There is no case identifier, input-value test or official-result constant.

S02CS is not submission-ready until its isolated S02CA/S02CS CANN 8.5.0 cloud
gate passes correctness and official-compatible A/B/A profiling.
