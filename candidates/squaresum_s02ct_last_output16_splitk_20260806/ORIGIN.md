# S02CT origin

- Baseline: `baselines/squaresum_s02ca_formal_best_20260806/SquareSumV1`
- Baseline host SHA-256: `0B5C6AEE3B01A63192A6CD4A59CF77A19373B6423274DC73968BAA77D84E9203`
- Baseline kernel SHA-256: `C6EA5927D44DDF905E50B309C08E811E9DF614B7018D47F2BEB6A71115EC4C80`
- Candidate host SHA-256: `610B4500A3C1CED862B3A7A63129300FC593FA2DA1810F58530F6C62D319C11C`
- Candidate kernel SHA-256: `C6EA5927D44DDF905E50B309C08E811E9DF614B7018D47F2BEB6A71115EC4C80`

S02CT is an independent one-line Host routing experiment derived directly from
formal S02CA. It does not include S02CQ, S02CR or S02CS, and its Kernel is
byte-identical to S02CA.

The only change raises `WORKSPACE_LAST_MAX_OUTPUTS` from 8 to 16. Therefore
large, contiguous last-axis/suffix reductions with 9..16 outputs reuse S02CA's
existing 40-block `reduceMode=2` workspace Split-K. Selection depends only on
shape and axes through the existing input/reduction/output gates.

S02CT is not submission-ready until its isolated S02CA/S02CT CANN 8.5.0 cloud
gate passes correctness and official-compatible A/B/A profiling.
