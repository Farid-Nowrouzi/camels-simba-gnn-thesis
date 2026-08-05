# Sparse U1000 equivalence test report

Decision: PASS. Test date: 2026-08-05. Runner: project virtual-environment Python 3.10, standard-library `unittest`, CPU.

The full lightweight suite passed 18 tests: five existing experiment-pipeline tests and thirteen sparse tests. Sparse coverage includes invalid-row filtering; descending raw mass; equal-mass halo-ID ties; repeatability and Top2-prefix-of-Top3 nesting; periodic wraparound; ordinary positions; distance ties; `R<=k`; padding exclusion; symmetrization; duplicates; self-loops; deterministic order; dense edge-set equivalence; normalization weights; Static node/output/gradient equivalence; Evolve first-layer/output/gradient equivalence; variable-node five-snapshot batching; target order; final-snapshot identity; strict manifest failures/order in both trainers; atomic collision/checksum/cleanup; and the bounded build/model smoke.

Forward comparisons used `atol=1e-6, rtol=1e-5`. Static gradient comparisons used `atol=2e-6, rtol=2e-5`; Evolve gradient comparisons used `atol=3e-6, rtol=3e-5`. The observed deterministic fixtures were equal within these limits, and the initial direct model probe observed zero maximum output difference for both models. Every compared and smoke gradient was finite.

Commands:

```text
envs/camels-gnn/bin/python -m py_compile <all modified/created Python files>
envs/camels-gnn/bin/python -m src.data.build_temporal_sequences --help
envs/camels-gnn/bin/python -m src.training.train_static_gcn --help
envs/camels-gnn/bin/python -m src.training.train_evolvegcn_h --help
envs/camels-gnn/bin/python -m unittest discover -s tests -v
git diff --check
```

One warning appeared: PyTorch attempted a CUDA capability probe during backward and reported CUDA unavailable. Tests stayed on CPU; it did not indicate a semantic difference. No NaN/Inf, dense conversion, leaked temporary/lock file, failure, or skip was observed.
