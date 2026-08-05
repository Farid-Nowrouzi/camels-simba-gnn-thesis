# U1000 Top1000 storage and compute audit

## Measured storage

- Filesystem: 197 GiB capacity, 88 GiB used, 101 GiB free (47%).
- Repository: 58 GiB; `data/raw`: 21 GiB; `data/processed`: 23 GiB; `experiments`: 14 GiB; reports: 64 MiB; outputs: 97 MiB.
- Dense U750 Top1000: 15,125,161,534 bytes (14.09 GiB).
- Relevant linear-head checkpoints: 40,927,714 bytes each; prediction CSVs are roughly 9–40 KB per split.

## Dense master estimate

U750 ratio gives 20.167 GB decimal (18.78 GiB) for U1000. Conservative range: 18.8–22.5 GB; central 20.2 GB. Reserve 45–50 GB for temp+final+validation/checksum and margin. With 101 GiB free, disk is sufficient and archival is unnecessary for capacity. This does not make dense RAM/compute safe.

## Runtime evidence and estimates

Filesystem timestamps: seed42 13m10s/111 epochs, seed123 21m32s/189, seed2025 9m54s/92. Approximate config-to-log epoch rates are 6.4–7.0 s; final prediction/metrics took about 4.5 minutes. These are inferred, not logged telemetry. Train700 epoch work is ~1.556× Train450. Six sizes sum to 3.378 Train450 training-set equivalents per seed. Build work is ~1.333× U750 in universe count but remains O(N²) per graph. Static absolute runtime and dense build duration are not recorded; no invented values are supplied.

## Memory and bottlenecks

Historical machine record: NVIDIA L40 46,068 MiB, host RAM 31 GiB with 14 GiB then available. Current CUDA/GPU inspection is unavailable. No peak telemetry or dataset-load time exists. The ~20.2 GB serialized dense dataset plus Python/tensor/collation copies makes eager host load unsafe. Main bottlenecks: pairwise distance/sort, dense serialization/eager loading, dense batch adjacency and matmul, and fixed-cost prediction over 300 val/test graphs. Checkpoint/prediction storage is minor.

## Decision

Storage capacity: sufficient. Dense memory/compute architecture: not acceptable. Refactor to sparse/lazy or sharded loading before build; then pilot Evolve Train700 seed42 followed by Static and record wall time plus peak RSS/GPU memory.
