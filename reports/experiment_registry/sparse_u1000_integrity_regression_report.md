# Sparse U1000 integrity regression report

Decision: PASS. Date: 2026-08-05. Runner: project Python 3.10, standard-library `unittest`, CPU.

## Validation results

- Python compilation: PASS for all modified source and test files, 0.051 s.
- Builder `--help`: PASS, 1.827 s.
- Evolve trainer `--help`: PASS, 1.495 s.
- Static trainer `--help`: PASS, 1.590 s.
- Full discovery: 35 passed, 0 failed, 0 skipped; framework time 0.284 s, process wall 2.481 s.
- Baseline tests: all original 18 continue to pass.
- New tests: 11 full-source/target provenance tests and 6 seed-binding tests pass.
- Dedicated sparse suite: all 13 pass.
- Dense compatibility: legacy dense default and dense model equivalence test pass.
- `git diff --check`: PASS before final staging.

One pre-existing environmental warning remains: PyTorch's existing smoke backward pass attempts CUDA initialization and receives error 304 because CUDA is unavailable. Tests remain on CPU. There were no unexplained warnings, NaN/Inf values, failures, skips, or leaked test directories.

## Independent post-fix assertions

- Hashing reads fixed 1 MiB chunks and closes sources through context managers.
- Sparse source identity is created and verified before preprocessing, then verified again before atomic completion; a mid-build source mutation therefore fails publication.
- Every selected catalogue and a real target table receive full SHA-256 entries.
- Manifest identity is canonical and insensitive to caller input order.
- Same-size, restored-mtime, target-content, omission, duplicate, role, structure, and top-level digest failures are rejected.
- `legacy_stat_only` is rejected for new sparse builds, while legacy metadata remains classifiable/readable without modifying loaders.
- Both trainers enforce manifest seed equality before output creation and preserve no-manifest behavior.
- No production dataset, production split, experiment configuration, checkpoint, prediction, or training epoch was created.
