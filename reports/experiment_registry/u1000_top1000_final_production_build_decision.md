# Final U1000 Top1000 production-build decision

## Decision: NO-GO

The sparse graph implementation itself passes: there is no reachable dense O(N²) tensor in the explicit sparse path; deterministic real-data Top-N, periodic sparse kNN, exact normalization, test suite, real-data Top1000 benchmark, atomic publication, Static reuse, disk/runtime, and projected memory all pass.

Two production-integrity requirements fail:

1. Both trainers accept a manifest declaring seed 42 when invoked with seed 999. `load_split_manifest` never receives or checks the expected training seed. This permits the wrong seed-specific validation/test population while the experiment configuration records another seed.
2. Dataset source provenance hashes only each catalogue's path, size, and mtime. It omits catalogue content and omits the target CSV entirely. The bounded full-hash benchmark shows full SHA-256 is cheap enough that this limitation is unjustified for production.

The target CSV is also absent, as expected under the task prohibition, and CUDA could not be validated in this environment. Those would be bounded later prerequisites, but they do not soften the two code/interface blockers above.

## Exactly one remediation task

Harden and regression-test manifest integrity end-to-end: require full SHA-256 records for every raw catalogue and target source in builder metadata, and require each trainer invocation seed to equal the split manifest's seed. Then rerun this production gate. Do not build U1000 before it passes.

## Required final record

1. Final decision: **NO-GO**.
2. Required branch: `thesis-sparse-u1000` (PASS).
3. HEAD commit: `7149cf7a7450cb8105b7ac0d432fec1197d17290` (PASS).
4. Branch ahead/behind: `0/0` relative to `origin/thesis-sparse-u1000`.
5. Working-tree status: no tracked or staged changes at preflight; numerous preserved pre-existing untracked artifacts.
6. Sparse CLI entry point: `envs/camels-gnn/bin/python -m src.data.build_temporal_sequences --graph_storage sparse_edge_index ...`.
7. Sparse schema: `camels_temporal_sparse_v1`.
8. Dense-default compatibility: PASS; `dense_adjacency` remains default.
9. Reachable dense-allocation result: PASS; none in explicit sparse path.
10. Full-dataset RAM accumulation result: confirmed; projected safe at about 2.5–4.5 GiB peak against about 60 GiB available.
11. Deterministic Top-N result: PASS; repeats, hashes, Top100⊂Top200 and Top500⊂Top1000 exact.
12. Mass-tie policy: raw Mvir descending, authoritative `col_1` ascending, stable original-row tertiary fallback.
13. Periodic sparse-kNN result: PASS; exact minimum image, deterministic, symmetric, unique, no loops/padding.
14. Largest intermediate tensor: `[1000,3]` float64 displacement row (plus `[1000]` distance/order rows).
15. kNN complexity: O(R²) distance work; current per-row lexsort adds O(R² log R) ordering; O(R+E) live storage.
16. Normalization equivalence: PASS; max absolute `0.0`, max relative `0.0`.
17. Unit-test result: 18 passed, 0 failed, 0 skipped; one CUDA-unavailable warning; 0.186 s framework time/2.261 s wall.
18. CLI validation result: compilation and builder/Static/Evolve `--help` all PASS.
19. Source-manifest current policy: aggregate SHA-256 of ordered catalogue path, size, and mtime_ns; target omitted.
20. Source-manifest recommendation: full per-catalogue and target-source SHA-256 manifest.
21. Source-manifest blocker status: production blocker.
22. Real-data benchmark universes: `LH_418`, `LH_131`, `LH_847`.
23. Real-data benchmark snapshots: all exact five per universe; 15 total.
24. Real-data benchmark Top-N: 1000 with real-node mask/padding.
25. Mean time per full snapshot: 0.48528 s across 13 full snapshots.
26. Slowest snapshot: `LH_847/0.25000`, 0.83951 s.
27. Peak RSS: 531,940 KiB whole-process high-water.
28. Temporary sample size: 2,987,990-byte `.pt` plus 344 sidecar bytes.
29. Projected U1000 dataset size: 0.85–1.20 GB; central 1.00 GB.
30. Projected temporary-space requirement: 2.4 GB; recommended pre-launch free margin 5 GB.
31. Current free disk: approximately 101 GiB.
32. Storage decision: PASS/sufficient.
33. Projected production build time: likely about 41 min total; optimistic about 31; pessimistic 72–75 min.
34. Restart-only classification: A, acceptable based on measured runtime.
35. Atomic-write result: PASS for lock/temp/fsync/validation/checksum/rename/marker; low risk from no post-marker directory fsync.
36. Collision test result: PASS; default refusal and explicit `/tmp` restart both verified.
37. Cleanup result: PASS; unique sample directory and lock/temp artifacts removed.
38. CUDA availability: unavailable; driver/NVML inaccessible.
39. Static forward/backward result: NOT AVAILABLE on CUDA; no long CPU Top1000 substitution.
40. Evolve forward/backward result: NOT AVAILABLE on CUDA; no long CPU Top1000 substitution.
41. Static final-snapshot reuse result: PASS by exact object identity for every required field.
42. Split-manifest result: **FAIL**; wrong invocation seed is accepted despite manifest seed, although other structural checks pass.
43. Exact proposed production command: recorded inertly in `u1000_top1000_proposed_production_build_command.md`; not authorized.
44. Expected output path: `data/processed/temporal_1000u_none_top1000_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1000_periodic_knn_sparse.pt`.
45. Expected dataset ID: logical `camels_simba_u1000_top1000_temporal5_none_periodic_knn_k8_box25_sparse_v1`; immutable identity is post-build `sha256:<64-hex>`.
46. Post-build validation plan: exact 1,000×5 population/protocol/schema/finite/mask/edge/hash/atomic checks, Static identity, 18 manifest checks, then batch-1 CUDA pilots; detailed in the audit.
47. Remaining risks: manifest seed mismatch, stat-only/target-omitting provenance, caller-supplied dataset identity, post-marker directory durability, absent target CSV, unavailable CUDA pilot.
48. Required manual actions: none before remediation/re-audit; after a future GO, create/hash target CSV and run the reviewed command in monitored `tmux`.
49. Highest-value next action: the single end-to-end manifest-integrity remediation task stated above.
50. Confirmation no source code changed: confirmed.
51. Confirmation no production dataset was built: confirmed.
52. Confirmation no production split was created: confirmed.
53. Confirmation no training occurred: confirmed; no optimizer/epoch loop.
54. Confirmation no checkpoint or prediction was created: confirmed.
55. Confirmation all `/tmp` outputs were removed: confirmed after final cleanup; only task-independent protected system directories were inaccessible to listing.
56. `git diff --check` result: PASS with no output; tracked and cached diff file counts both zero.
57. Final `git status --short`: no tracked/staged changes; the eight authorized reports are untracked task outputs, alongside preserved pre-existing untracked artifacts.
