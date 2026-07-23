# Ascend S9 Operators

Development workspace for the Ascend AI Operator Challenge S9 on Ascend 910B and CANN 8.5.

## Layout

- `case_910b/`: challenge baselines and PyTorch NPU extension wrappers for the five operators.
- `greater-fast/`: experimental Ascend C fast path for `Greater`.
- `index-add-fast/`: experimental Ascend C fast path for `IndexAdd`.
- `diagnose_index_add.py`: deterministic `IndexAdd` diagnostic.
- `extra_correctness.py`: additional correctness and edge-case tests.

## Status

This repository is under active development. Baseline correctness and performance changes should be revalidated in the official CANN 8.5 environment before submission.
