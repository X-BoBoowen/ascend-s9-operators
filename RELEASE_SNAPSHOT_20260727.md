# S9 Git release snapshot — 2026-07-27

## Scope of this commit

This snapshot preserves the finalized source trees for:

- `Concat`
- `Greater`
- `IndexAdd`
- `Transpose`

The tracked `submission-src/SquareSumV1` working-tree changes are intentionally
excluded from this commit because that operator has not completed final
optimization, independent rebuild, package verification, and submission
acceptance.

## Verification summary

| Operator | Correctness coverage | Representative 910B4 result |
| --- | --- | --- |
| Concat | 9 directed + 100 random cases, bitwise equal | 16 cores retained after A/B testing |
| Greater | 246 combinations, including broadcasting, NaN/Inf/±0 and five dtypes | dtype/broadcast-specific paths retained |
| IndexAdd | 23 directed + 170 original random + 345 extended cases = 538 passed | public shape about 9.108 µs; high-collision int8 about 256.659 µs |
| Transpose | 48 directed + 200 random + 152 extended cases = 400 passed bitwise | public FP16 128×256 about 4.44–4.60 µs |

The four local source trees were copied from their final cloud experiment
directories and verified file-by-file with SHA-256 before committing.

## Submission package hashes

The packages are stored outside this Git repository under
`D:\29722\Desktop\GCC\提交相关材料`.

| Package | SHA-256 |
| --- | --- |
| `Concat.zip` | `82e17ebf1f062c42f61d64f0788b7c2ed6a2633ff26a9abd44fae5b6da7fe814` |
| `Greater.zip` | `6c61f94e838843b052dd72e4cd5622cf309f5d3811370e15a6a15d2353df2539` |
| `IndexAdd.zip` | `d2b087093f5a7e3bc1559cadacd703f7bf3592fbd1a5a5dcf72280a2d2ac2a5f` |
| `Transpose.zip` | `0b88b825e7ee6e5cb598e5c3eb4638d4f652a4a0eb32671087a10a4aa3ff57e8` |

## Final installer hashes

| Operator | Final `.run` SHA-256 |
| --- | --- |
| Concat | `60bdb960e7158bd4f259dc29ac9f95a904c813e3bc3edcfbd2c11dbb80a69095` |
| Greater | `ccc3e22b98dd6a39f5921ab100dd67d60933d12bc9492bd291fee92bd987af03` |
| IndexAdd | `75fda0d2a5f034610a9b3db50503750538ec427a46a168ea52e5364702644931` |
| Transpose | `1edad240181a0e73e1fe4ad607ea0457b12197cc38aa16218057873017287214` |

## Handoff notes

- `IndexAdd.zip` and `Transpose.zip` were uploaded back to the cloud, unpacked,
  and checked against their final sources and installers. The final checks
  reported `FINAL_INDEXADD_PACKAGE_REMOTE_VERIFY_OK` and
  `FINAL_TRANSPOSE_PACKAGE_REMOTE_VERIFY_OK`.
- Transpose retains 32 cores: the 40-core A/B result was slower
  (about 5.130 µs).
- Transpose includes identity/contiguous-copy, FP16 16×16 single and batched
  matrix paths, cyclic suffix transpose, stride gather, and a generic fallback.
- Continue with `SquareSumV1` next. Do not package or submit its current tracked
  working-tree state without rebuilding, installing the resulting `.run`,
  rerunning the full regression matrix, and verifying the ZIP independently.
