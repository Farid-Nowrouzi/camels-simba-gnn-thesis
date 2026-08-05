# Sparse U1000 integrity-hardening audit

Audit date: 2026-08-05. Branch: `thesis-sparse-integrity-hardening`. Parent sparse implementation `7149cf7a7450cb8105b7ac0d432fec1197d17290` is an ancestor; the production-gate audit commit `9c391a5` was the branch starting point. No matching builder/trainer process was observed; `pgrep` itself remained unavailable due the environment uptime error.

## File-level implementation map

- `src/data/source_manifest.py`: streaming SHA-256, target-table inspection, canonical entry ordering/serialization, manifest construction, verification, and legacy classification.
- `src/data/build_temporal_sequences.py`: raw catalogue enumeration, sparse full-hash policy enforcement, pre-processing manifest creation/verification, post-processing re-verification, and metadata propagation.
- `src/training/split_manifest.py`: manifest seed parsing/type/equality validation, early seed-only validation, split SHA identity, dataset-sidecar provenance extraction, and training-code commit identity.
- `src/training/train_evolvegcn_h.py`: early Evolve seed binding, loader seed binding, and experiment identity propagation.
- `src/training/train_static_gcn.py`: early Static seed binding, loader seed binding, and experiment identity propagation.
- `tests/test_source_manifest_integrity.py`: 11 provenance regressions.
- `tests/test_split_manifest_seed_binding.py` and `tests/test_sparse_graph_pipeline.py`: six dedicated seed regressions and corrected matching-seed integration coverage.

No graph, feature, Top-N, normalization, kNN, model, pooling, optimizer, learning-rate, loss, epoch, patience, dropout, or batch setting changed.

## Source provenance result

PASS. New sparse builds resolve to policy `full_sha256` and reject `legacy_stat_only`. The builder deterministically enumerates the selected universe/snapshot catalogue paths, creates and verifies `camels_source_manifest_v1` before reading targets/building graphs, then re-verifies the unchanged manifest before atomic output. Each entry contains portable relative path, role, universe/snapshot where applicable, size, and complete SHA-256. Target tables are first-class entries with row count and detected universe/target columns. Missing production targets, duplicate paths, duplicate catalogue identities, bad roles, size/content changes, metadata disagreement, and canonical digest mismatch fail clearly.

Canonical order is role, numeric universe, numeric snapshot, relative path. Canonical UTF-8 JSON uses sorted keys and compact separators; absolute operational roots and summary/verification fields do not enter the portable top-level digest.

The bounded benchmark used the five real `LH_131` catalogues plus `outputs/target_inspection_750u.csv`: six files, 23,739,686 bytes, build-plus-verification 0.048507 s, 466.7315 MiB/s, 16,080 KiB peak RSS. Reversed catalogue input produced the identical digest `8da7eb649a682a98f158dcda5e1b260906766a10719d58ace455095299d9e029`. Linear projection for 22,148,571,647 catalogue bytes is 45.25 s, explicitly an estimate. The full corpus was not hashed.

Legacy dense builder semantics remain stat-only; existing files are not invalidated or mutated. Historical metadata classifies as legacy/unverified and historical dataset loading is unchanged.

## Seed-binding result

PASS. Both trainers require invocation seed equal to the manifest's strict integer seed. Mismatch errors contain both values and path. The top-level checks execute before output-directory creation; the bounded rejection test confirms no experiment tree or derivative artifact exists. Matching seed and no-manifest historical splitting pass.

New experiment configs jointly record dataset path/caller identity, published dataset SHA-256, source-manifest SHA-256/policy, target SHA-256, split-manifest SHA-256, manifest seed, invocation seed, ordered split hashes, builder/training commits, and established graph/model protocol fields. Missing legacy sidecars yield an explicit `legacy_or_missing` status rather than blocking historical runs.

## Re-audit conclusion

Both final integrity blockers are closed. Full content—including target content—drives sparse dataset source identity, mutations are detected even when size and mtime are preserved, and the seed-specific split contract is strict in both trainers. All 35 tests pass. No production or scientific artifact was created.
