# U1000 Top1500 resource-risk assessment

Audit date: 2026-08-06 UTC. No production build, CUDA pilot, or training was run.

## Decision

**Resource risk: LOW.** The implementation is restart-only and accumulates all 1,000 five-snapshot samples before one `torch.save`, but its sparse representation and row-streamed kNN keep projected memory and disk use far below current capacity. This resource decision does not override the separate matrix-preflight integrity blocker.

## Current machine

| Resource | Observation |
|---|---:|
| Output filesystem | `/dev/mapper/pve-vm--178--disk--0` |
| Capacity / used / available | 211,048,095,744 / 96,011,522,048 / 105,979,060,224 bytes |
| Human-readable available disk | approximately 99 GiB |
| RAM total / currently available | 67,108,864,000 / 63,005,167,616 bytes (62.5 / 58.7 GiB) |
| Swap | 0 bytes |
| Logical CPUs | 16 |
| GPU | `nvidia-smi` could not communicate with the driver; irrelevant to the CPU-only builder |

## Production anchor

The completed Top1000 `.pt` is 1,047,033,642 bytes. Its production log began at 20:42:19 UTC and publication completed at approximately 21:32:06 UTC: about 49 minutes 47 seconds. The pre-production bounded benchmark reported 531,940 KiB process high-water for a three-universe sample and projected 2.5–4.5 GiB for full Top1000; the production launcher did not record actual peak RSS.

## Builder strategy

- The builder retains all 1,000 universe records in the Python `dataset` dictionary and serializes once; it does not shard or stream.
- Each sparse kNN source iteration retains one `[R,3]` displacement row and `[R]` distance/order vectors. It performs O(R²) distance work but does not retain `[R,R]`, `[R,R,3]`, `[N,N]`, or `[B,N,N]` tensors.
- Per-snapshot Pandas/NumPy intermediates become unreachable after the snapshot record is built; the retained objects are padded float32 features/masks, int64 sparse edges, targets, and metadata.
- `torch.save` writes one same-directory temporary `.pt`. It is renamed to the final path, so a new build does not require simultaneous temporary and final dataset copies. Metadata is about 1.5 MB at Top1000 and scales primarily with selected-halo provenance.
- kNN runs on CPU; `CUDA_VISIBLE_DEVICES` is empty and `--device cpu` is explicit.

## Estimates

| Quantity | Estimate | Basis |
|---|---:|---|
| Final Top1500 `.pt` | 1.50–1.65 GB (central 1.57 GB) | 1.5× Top1000 node/mask/edge/key payload, reduced slightly by 0.884% padding |
| Metadata + marker | about 2–3 MB | Top1000 metadata is 1,492,868 bytes; selected-key lists scale with Top-N |
| Incremental peak disk during new build | about 1.65 GB | one temporary dataset plus temporary metadata; final replaces temp |
| Conservative free-space reservation | 4 GiB launcher gate; 6 GiB preferred operational margin | covers serialization variance, log, and failure residue |
| Peak host memory | approximately 4–7 GiB | Top1000 projection scaled for 1.5× retained payload plus serializer/import/Pandas margin |
| Expected duration | approximately 60–100 minutes | measured Top1000 49m47s; Top1500 kNN arithmetic is up to 2.25× while parsing/hash work is near-constant |
| Relative kNN work | up to 2.25× Top1000 for full snapshots | quadratic distance work: `(1500/1000)^2` |

Available disk exceeds the central temporary requirement by roughly 67× and available RAM exceeds the high estimate by roughly 8×. Lack of swap is noted but does not create a realistic OOM risk at these margins. A failure requires a full restart, but an interruption cannot publish a completion marker for a partial dataset.

## Operational caveats

The Top1500 launcher uses a 4 GiB free-space floor, whereas the Top1000 launcher used 5 GiB. Both exceed the projected atomic-write requirement, but harmonizing the floor to 5 GiB would improve parity and is optional rather than a build blocker. The writer does not fsync the directory after creating the completion marker; this is a low power-loss durability caveat already present in the Top1000 path, not a Top1500-specific resource risk.
