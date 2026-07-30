# Canonical k × Hidden-Dimension Factorial Preflight

## Decision

The experiment `controlled_knn_by_hidden_dim_factorial_500u_top100` consists of two parallel, independently controlled within-model factorials:

- EvolveGCN-H uses fixed `num_layers=2` and `batch_size=4`.
- Static GCN uses fixed `num_layers=3` and `batch_size=8`.
- Only k, hidden dimension, and seed vary within either model.
- Cross-model comparisons are descriptive because temporal input, architecture, depth, head implementation, and batch size differ.

The complete design has 48 rows. Twenty-seven artifact-complete rows are reusable and exactly 21 new runs are required. All eight k-specific processed datasets already exist and pass lightweight metadata checks, so no graph rebuilding is required.

No training, tmux launch, dataset deserialization, checkpoint loading, experiment editing, or Git mutation was performed.

## Fixed protocols

| Field | EvolveGCN-H | Static GCN |
|---|---|---|
| Input | five-snapshot temporal graph sequence | native final-snapshot static graph |
| Hidden dimensions | 32, 64 | 32, 64 |
| Layers | 2 fixed | 3 fixed |
| Batch size | 4 fixed | 8 fixed |
| Epochs / patience | 300 / 40 | 300 / 40 |
| Learning rate / weight decay | 0.001 / 0.00001 | 0.001 / 0.00001 |
| Dropout / activation | 0.2 / ReLU | 0.2 / ReLU |
| Graph pooling | mean | mean |
| Temporal pooling | mean | not applicable |
| Convolution | EvolveGCN-H | GCN |
| Regression head | MLP | established Static MLP |
| Self loops | enabled | enabled by established GCN implementation |
| Gradient clipping | 1.0 | 1.0 |
| Summary features / target normalization | disabled / disabled | disabled / disabled |
| Ratios | 0.70 / 0.15 / 0.15 | 0.70 / 0.15 / 0.15 |

Both protocols use 500 CAMELS-SIMBA universes, Top100 raw-Mvir-descending nodes, seven features (`log10_Mvir`, X, Y, Z, VX, VY, VZ), minmax normalization, periodic kNN, box size 25, k values 4/6/8/12, and seeds 42/123/2025.

## Live family status

| Family | Required | Complete | Missing | Partial | Runnable | Skipped complete |
|---|---:|---:|---:|---:|---:|---:|
| EvolveGCN-H | 24 | 15 | 9 | 0 | 9 | 15 |
| Static GCN | 24 | 12 | 12 | 0 | 12 | 12 |
| Total | 48 | 27 | 21 | 0 | 21 | 27 |

The Evolve dry run selected only h32/L2 at k=4,6,12 for seeds 42,123,2025. The Static dry run selected only h32/L3 at k=4,6,8,12 for those seeds. No h64 run, Evolve h32 k8 run, Static h32/L2 run, graph-building command, or unrelated experiment was selected.

Allow-incomplete verification independently recomputed prediction metrics for all 27 complete rows and found no compatibility errors. Its expected verdict is `INCOMPLETE`, with 15/24 Evolve rows and 12/24 Static rows verified.

## Dataset mapping

| k | Evolve temporal dataset | Static final-snapshot dataset |
|---:|---|---|
| 4 | `data/processed/temporal_500u_logmass_minmax_top100_periodic_knn_k4/camels_500u_temporal_logmass_minmax_top100_periodic_knn_k4.pt` | `data/processed/static_500u_logmass_minmax_top100_periodic_knn_k4/camels_500u_static_logmass_minmax_top100_periodic_knn_k4.pt` |
| 6 | `data/processed/temporal_500u_logmass_minmax_top100_periodic_knn_k6/camels_500u_temporal_logmass_minmax_top100_periodic_knn_k6.pt` | `data/processed/static_500u_logmass_minmax_top100_periodic_knn_k6/camels_500u_static_logmass_minmax_top100_periodic_knn_k6.pt` |
| 8 | `data/processed/temporal_500u_minmax/camels_500u_temporal_logmass_minmax_top100_periodic_knn.pt` | `data/processed/static_500u_logmass_minmax_top100_periodic_knn/camels_500u_static_logmass_minmax_top100_periodic_knn.pt` |
| 12 | `data/processed/temporal_500u_logmass_minmax_top100_periodic_knn_k12/camels_500u_temporal_logmass_minmax_top100_periodic_knn_k12.pt` | `data/processed/static_500u_logmass_minmax_top100_periodic_knn_k12/camels_500u_static_logmass_minmax_top100_periodic_knn_k12.pt` |

All eight dataset files and sidecars exist. Sidecars verify 500 successful universes, Top100, minmax, periodic flags, the correct k, box size 25, the seven required features, raw-Mvir-descending node selection, five snapshots for Evolve, and final snapshot 1.0 for Static.

## Split verification

| Seed | Canonical source | SHA-256 ordered split signature |
|---:|---|---|
| 42 | `experiments/evolvegcn_h_500u_seed42_final32/config.json` | `0f963679cd284fca861fc2c59d88bdae8e8f1f21e2cbe1bb73bd593b49056748` |
| 123 | `experiments/evolvegcn_h_500u_seed123_final32/config.json` | `853549f16ef8eb3d7f18ae850c94b13c0c8bf0e770bb99cfbffff48b03530266` |
| 2025 | `experiments/evolvegcn_h_500u_seed2025_final32/config.json` | `3ce48b66c11e30bec459c52ba7f4a900809dd2b45be0995b8b56aeaefc747951` |

Each source contains ordered 350/75/75 splits, no overlap, and exactly `LH_0` through `LH_499`. All 27 reusable rows match the corresponding signature. Static new-run commands explicitly pass the canonical `--split_config_path`. Evolve uses the audited deterministic sorted-ID, seeded-shuffle procedure; each planned row records the expected signature for mandatory post-run verification.

## Pipeline and collision checks

- The model CLIs expose all required settings; no generic pipeline modification was needed.
- The family schema can represent the eight `(hidden_dim, k)` cells as grouping values and applies per-row hidden-dimension overrides.
- Both training modules imported successfully in `envs/camels-gnn`.
- All 21 planned experiment names are unique.
- None of the 21 planned experiment paths exists.
- Zero partial directories and zero output collisions were found.
- The generic tmux launcher accepts one specification only. A dedicated wrapper was therefore required to run and verify the two families sequentially in one session.
- The preferred tmux session `canonical-knn-hidden-factorial` was absent during preflight.

## Resource assessment

At preflight:

- repository filesystem: 197 GB total, 103 GB available, 46% used;
- RAM: 62 GiB total, 60 GiB available;
- swap: none;
- load average: 11.33 / 13.46 / 12.98 immediately after host startup;
- GPU: `nvidia-smi` could not communicate with the NVIDIA driver, so GPU readiness must be rechecked immediately before future execution;
- no active EvolveGCN-H/Static GCN training process and no executing family runner were found.

Disk and RAM exceed the prior safety thresholds. GPU telemetry is the one execution-time prerequisite that remains to be rechecked; preparation and dry-run behavior are unaffected.

## Lightweight validation

- New specification JSON parsing: PASS.
- Family schema/status: PASS.
- Family dry runs: PASS, exactly 9 + 12 jobs.
- Allow-incomplete family verification: PASS with expected incomplete verdicts.
- Exact split-signature verification: PASS.
- Dataset path and sidecar verification: PASS.
- Output-name uniqueness/collision verification: PASS.
- Wrapper shell syntax: PASS.
- Notebook execution, `.pt` loading, checkpoint loading, training, and tmux launch: not performed.
