# Independent post-sparse U1000 Top1000 production-build gate audit

Audit date: 2026-08-05. Mode: inspection and bounded validation only. Final decision: **NO-GO**.

## Scope and preflight

The required branch `thesis-sparse-u1000` was active at exact HEAD `7149cf7a7450cb8105b7ac0d432fec1197d17290`; required design commit `b49e520ce0348bc93e7de3f74ca167c3219ee18b` and implementation commit were present. `origin/thesis-sparse-u1000...HEAD` was `0 0`. There were no tracked or staged modifications at preflight. Numerous unrelated untracked artifacts already existed and were preserved. `ps` showed no matching builder/trainer; `pgrep` itself was unavailable due an environment `/proc`/uptime error.

Environment: Python 3.10.12; PyTorch 2.6.0+cu124; PyG 2.7.0. CUDA was unavailable (`torch.cuda.is_available() == False`, device count 0); `nvidia-smi` could not communicate with a driver. Host memory was 62 GiB total/about 60 GiB available; no swap. Workspace and `/tmp` are the same ext4 filesystem, with about 101 GiB free.

## Scientific protocol and code-path audit

The explicit `--graph_storage sparse_edge_index` path implements `camels_temporal_sparse_v1` while `dense_adjacency` remains the CLI/function default. Sparse processing follows:

```text
builder CLI
  -> build_temporal_dataset (LH_0..LH_999 loop)
  -> build_universe_sequence (numeric five-snapshot order)
  -> process_snapshot
  -> read_hlist_file
  -> clean_halo_dataframe
  -> select_top_halos
  -> build_node_features/build_positions/normalize_features/pad_nodes
  -> build_sparse_knn_edge_index
  -> temporal edge_index_list/Nodes_list/mask_list serialization
  -> atomic_write_sparse_dataset
  -> trainer dataset view
  -> collate_sparse_temporal or final-snapshot static view
  -> disconnected sparse batch
  -> normalize_sparse_edges/index_add message passing
  -> EvolveGCN-H or Static GCN
```

The source columns are exactly Mvir `col_10`, XYZ `col_17..19`, and VXYZ `col_20..22`; feature 0 becomes `log10(Mvir)` only after raw-mass selection. `normalization=none` leaves these float32 features unscaled. Padding is zeros plus a mask and is excluded during sparse collation. Target normalization remains none.

Search and reachability tracing found dense `[N,N,3]`, `[N,N]`, `[B,N,N]`, identity expansion, and dense multiplication only in legacy functions/branches and test equivalence fixtures. The sparse branch never calls `compute_pairwise_distances`, `build_knn_adjacency`, `to_dense`, or `to_dense_adj`; neither sparse model recreates adjacency. Optional edge weights stay 1-D. The complete matrix is in `u1000_top1000_post_sparse_code_path_matrix.csv`.

## Deterministic Top-N

Cleaning drops nonfinite required feature rows and nonpositive raw Mvir. Selection uses stable mergesort over `(raw Mvir descending, col_1 ascending)`; stable input order is the tertiary fallback for identical mass/ID. A deterministic fixture returned equal-mass IDs `[3,5,8]`, then the lower-mass ID `1`.

On real `LH_131/0.51209`, repeat selections at Top100/200/500/1000 produced identical keys and hashes. Top100 was the exact prefix of Top200 and Top500 the exact prefix of Top1000. The Top1000 hash was `9f2944fdaa0725f0260657579a6037f9f7f01851557dd1517fa46163285db6e9`. Result: PASS. A minor provenance limitation is that duplicate authoritative IDs would make the stored key hash omit the original-row fallback value even though ordering itself remains deterministic.

## Periodic sparse kNN

For each real source node the implementation computes `diff = positions - source`, then `min(abs(diff), box_size-abs(diff))`, squared Euclidean distance, excludes self with infinity, and lexsorts by `(distance, selected rank)`. It selects `min(8,R-1)` neighbors, inserts both orientations into a set, and sorts the final unique edge pairs. There are no builder self-loops or padded edges. Model normalization alone adds self-loops.

The bounded sample confirmed the severe-padding case (R=157), an R=869 case, and 13 full R=1000 cases. Largest intermediates were `[1000,3]` float64 displacement and `[1000]` distance/order rows; no `[R,R]`, `[R,R,3]`, `[N,N]`, or `[B,N,N]` appeared. Complexity is O(R²) distance arithmetic and O(R log R) ordering per source in the current `lexsort`, with O(R+E) live numerical storage plus the O(E) Python edge set. Result: PASS.

## Graph normalization

Sparse normalization inserts loops, creates float32 implicit weights when absent, coalesces duplicate coordinates by summation, defines degree at targets after loop insertion, and computes `w / sqrt((d_target+eps)(d_source+eps))`. Zero-degree nodes receive their self-loop when loops are enabled; without loops eps prevents nonfinite arithmetic. Device derives from edge/value tensors and batched graphs remain disconnected through index offsets.

An independent deterministic graph containing duplicate directed coordinates and an isolated node matched dense `D^-1/2(A+I)D^-1/2` with maximum absolute difference `0.0` and maximum relative difference `0.0`; weights were finite float32. Both models use this same function. Result: PASS.

## Tests and CLI

- Python compilation of all nine sparse-related source files and the sparse test file: PASS, 0.053 s.
- Builder `--help`: PASS, 2.018 s.
- Static trainer `--help`: PASS, 1.400 s.
- Evolve trainer `--help`: PASS, 1.541 s.
- Full unittest discovery: 18 passed, 0 failed, 0 skipped in framework-reported 0.186 s (2.261 s process wall).
- Dedicated sparse suite: 13 passed, 0 failed, 0 skipped in 0.268 s (2.254 s process wall).
- Affected experiment-pipeline regression suite: 5 passed, 0 failed, 0 skipped in 0.011 s (0.065 s wall).

One warning occurred: PyTorch's backward engine attempted CUDA initialization and received error 304. All tests stayed on CPU. No package was installed and no test regression occurred.

## Real-data bounded benchmark

The authorized `/tmp` benchmark covered `LH_418` (157/869 early snapshots and substantial padding), `LH_131` (typical full case), and `LH_847` (high-count case including 9,795 rows), all five snapshots, Top1000, none normalization, periodic kNN k=8, box 25, sparse schema.

Across 13 full snapshots, mean preprocessing time was 0.48528 s; across all 15, median was 0.46422 s. Slowest was `LH_847/0.25000` at 0.83951 s. Mean full-snapshot kNN time was 0.08113 s. Actual second-pass construction of all three temporal samples was 6.3129 s. Whole benchmark wall time including repeat construction, atomic tests, hashing, and checks was 13.3493 s. Process peak RSS was 531,940 KiB. Per-snapshot rows, edge counts, timings, hashes, and intermediate shapes are in the benchmark CSV.

The atomic sample was 2,987,990 bytes, or 995,997 bytes/universe. Publication used an exclusive lock and same-directory UUID temp files; data fsync preceded validation and SHA-256; metadata was fsynced; data and metadata used `os.replace`; the parent directory was fsynced; and a fsynced completion marker was created. Checksum/loading, collision refusal, and explicit overwrite restart all passed. No lock/temp remained and the unique sample directory was removed. The completion marker's parent directory is not fsynced after marker creation, a low power-loss durability risk.

## Static reuse and model pilot

For every bounded real universe, Static `X`, `edge_index`, `edge_weight`, `mask`, `target`, and snapshot metadata were the exact final temporal objects (`is`, not merely equal); universe IDs matched. No selection, features, kNN, or normalization were rebuilt. Result: PASS.

CUDA Top1000 forward/backward was not available. No long CPU Top1000 model pass was substituted. Static and Evolve production-environment forward/backward results are therefore NOT AVAILABLE, not failures. A batch-1 CUDA pilot remains required before training, not before a CPU dataset build.

## Split-manifest interface

The loader rejects duplicate IDs, IDs absent from the dataset, split overlap, wrong stored counts, wrong ordered-ID hashes, wrong dataset identity, missing required fields, and a non-prefix parent. Both trainers preserve stored order and historical internal splitting remains available without a manifest. Nested prefix representation can encode Train20 ⊂ Train50 ⊂ Train100 ⊂ Train200 ⊂ Train450 ⊂ Train700.

Result: **FAIL**. The manifest's required `seed` is never compared with the trainer's invocation seed. The committed integration test explicitly passes a seed-42 manifest to both loader factories with `seed=999` and asserts success. Consequently seed-specific fixed validation/test membership is not enforced, and experiment metadata can claim a seed different from the consumed split.

## Provenance, storage, runtime, and memory

Current source-manifest policy hashes ordered catalogue path, size, and mtime_ns only. It does not hash inode, header, chunks, file contents, or any target-source field/content. A 15-file, 62.21 MB sample hashed fully at 1,441.6 MiB/s (0.0412 s). Linear full-corpus estimate is 14.7 s on warm cache; allow 1–4 minutes cold. Full per-file SHA-256 plus target SHA-256 is recommended. Current policy is a production blocker.

Projected U1000 output is 0.85–1.20 GB, central 1.00 GB; reserve 2.4 GB for atomic temp/final operation and 5 GB free as the recommended safety floor. Current 101 GiB free passes. Likely total build wall is about 41 minutes, optimistic about 31 and pessimistic 72–75 minutes. Restart-only classification is A (acceptable).

The builder retains all universes in `dataset` before one `torch.save`; it does not stream. The sparse projected file is only about 1 GB, and a conservative 2–4× container factor plus import baseline remains far below about 60 GiB available. Full-scale host peak is estimated 2.5–4.5 GiB. Accumulation is confirmed but safe at fixed U1000.

## Post-build validation required after a future authorized build

1. Require matching `.pt`, `.metadata.json`, `.complete`, and recomputed SHA-256; exact schema `camels_temporal_sparse_v1`, graph storage, builder commit/config hash, full source/target manifest hash, and logical dataset ID.
2. Require exactly 1,000 unique ordered IDs `LH_0..LH_999`; five exact ordered snapshots each; seven exact feature definitions; finite raw targets/features; Top1000 masks and documented padding.
3. Reject any dense adjacency key/tensor, out-of-bounds/duplicate/asymmetric/self/padded edge; validate k=8 selection/union bounds and periodic-boundary fixtures.
4. Recompute deterministic selection and graph hashes on a reviewed sample and aggregate hashes over all snapshots.
5. Confirm Static final graph exact identity for x/edges/weights/mask/target/universe/snapshot.
6. Only after dataset checksum exists, create/review the 18 immutable split manifests; require seed binding, fixed ordered val/test, exact counts/hashes/identity, disjointness, and all nested prefixes.
7. Before training, run batch-1 CUDA forward/backward pilots for Evolve Train700 seed42 and Static Train700 seed42: no OOM; finite output/loss/gradients; exact test ordering; nonconstant predictions; runtime and peak GPU/host memory recorded. No optimizer epoch is part of the dataset validation.

## Decision

**NO-GO.** Sparse compute, storage, runtime, memory, and atomic operation are adequate. Seed-specific split enforcement and byte-exact source/target provenance are not. The one remediation task is to harden and regression-test manifest integrity end-to-end, covering both full source/target SHA-256 binding and trainer seed-to-manifest binding, then rerun this gate.

No production dataset, production split, experiment config, training epoch, optimizer step, checkpoint, or scientific prediction was created. No source code was changed. All unique benchmark output and ephemeral audit text/help/results under `/tmp` were removed; the final audit-created `/tmp` count was zero.
