# U1000 Top1000 sparse storage, runtime, and memory projection

Audit date: 2026-08-05. Estimates use the three-universe, 15-snapshot real-data sample in `u1000_top1000_sparse_realdata_benchmark.csv`. All projections are estimates, not production measurements.

## Measured sample

- Sample output: 2,987,990 bytes for three universes, or 995,997 bytes/universe.
- Direct tensor payload: 2,748,672 bytes total, or 916,224 bytes/universe.
- Per-universe direct payload composition: node features 140,000 bytes; masks 20,000 bytes; edge indices approximately 756,224 bytes in this sample; edge weights 0; target 4 bytes.
- Pickle/container/snapshot metadata overhead: approximately 79,773 bytes/universe, inclusive of tensor-record overhead and selected-key metadata. Atomic sidecars were 344 bytes for the bounded file.
- Peak process RSS: 531,940 KiB. This is whole-process high-water RSS including Python, pandas, NumPy, and PyTorch imports, cached raw frames, two consecutive bounded passes, and the in-memory sample; it is not incremental builder memory.
- Largest sparse-kNN live work arrays at Top1000: float64 displacement `[1000,3]` = 24,000 bytes, float64 distance `[1000]` = 8,000 bytes, plus an order/key row of roughly 8,000 bytes. Edge-set Python overhead is larger but remains O(E), not O(N²).

## Production storage estimate

| Component | Central U1000 estimate |
|---|---:|
| Node features | 140.0 MB |
| Edge indices | 756.2 MB |
| Edge weights | 0 MB (implicit binary weights) |
| Masks | 20.0 MB |
| Targets | negligible (about 4 KB tensor payload) |
| Metadata and Python/pickle/container overhead | 79.8 MB |
| Dataset `.pt` central total | 996.0 MB (0.928 GiB) |
| Checksum/metadata/completion files | approximately 1–3 MB, dominated by aggregate metadata |

Range: lower 0.85 GB, central 1.00 GB, upper 1.20 GB decimal. Same-directory atomic publication needs roughly one final-size temporary file, not a dense duplicate; reserve 2.4 GB for temp plus final/validation headroom. A conservative operational safety margin is 5 GB free before launch. Current free disk was approximately 101 GiB, so storage is sufficient by a very large margin.

## Runtime estimate

The 13 full-Top1000 snapshots averaged 0.48528 s (median across all 15 snapshots 0.46422 s). The slowest was `LH_847/0.25000` at 0.83951 s. Mean kNN time for full snapshots was 0.08113 s; catalogue parsing dominated at 0.39349 s.

- Optimistic: 30 minutes preprocessing plus under 1 minute serialization/validation/checksum.
- Likely: 40.4 minutes preprocessing (`0.48528 × 5,000`) plus about 1 minute publication, approximately 41 minutes total.
- Pessimistic: 70.0 minutes preprocessing (`0.83951 × 5,000`) plus 2–5 minutes publication/cold-I/O allowance, approximately 72–75 minutes total.
- Time per universe: measured component benchmark 2.153 s/universe; actual second-pass sequence assembly 2.104 s/universe.

Restart-only classification: **A — restart-only is acceptable.** Losing at most roughly 75 minutes is operationally tolerable, and the final output is not published without validation/checksum/completion. Sharding is not required by measured runtime, though it would improve convenience.

## Full-scale memory behavior

The exact builder stores every completed sequence in `dataset` and calls one `torch.save(dataset, ...)` only after all 1,000 universes are complete. Serialization therefore accumulates rather than streams; metadata and selected-halo lists also grow linearly. At approximately 1.0 GB serialized payload, even a conservative 2–4× in-memory/container factor plus the measured 0.52 GiB baseline implies roughly 2.5–4.5 GiB peak host use. The current host has 62 GiB total and about 60 GiB available. This accumulation is architecturally undesirable but safe at the fixed U1000 scope. No all-universe dense adjacency exists.

Storage, runtime, and host-memory conclusions are PASS; they do not override the manifest-integrity blockers documented by the final decision.
