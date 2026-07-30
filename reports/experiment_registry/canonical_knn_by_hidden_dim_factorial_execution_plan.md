# Canonical k × Hidden-Dimension Factorial Execution Plan

## Launch policy

This plan is prepared but has not been launched.

Future execution is strictly sequential in one tmux session:

1. preflight both families;
2. execute the nine missing EvolveGCN-H runs;
3. verify all 24 Evolve rows;
4. execute the twelve missing Static GCN runs;
5. verify all 24 Static rows;
6. stop immediately on any nonzero exit status.

The generic runner skips only artifact-complete compatible rows, rejects partial output directories, never selects `reuse` rows for training, writes a per-experiment log, and verifies artifact completeness after each run. The wrapper adds a combined timestamped log and is restart-safe because it repeats both preflights before creating the tmux session.

## Exact future launch command

```bash
bash scripts/launch_knn_by_hidden_dim_factorial_tmux.sh \
  --session canonical-knn-hidden-factorial
```

Do not run this command until `nvidia-smi` succeeds and a fresh process/resource preflight is clean.

## Exact execution order

| Order | Model | k | Seed | Experiment |
|---:|---|---:|---:|---|
| 1 | EvolveGCN-H h32/L2 | 4 | 42 | `evolvegcn_h_u500_top100_norm-minmax_k4_h32_l2_factorial_seed42` |
| 2 | EvolveGCN-H h32/L2 | 4 | 123 | `evolvegcn_h_u500_top100_norm-minmax_k4_h32_l2_factorial_seed123` |
| 3 | EvolveGCN-H h32/L2 | 4 | 2025 | `evolvegcn_h_u500_top100_norm-minmax_k4_h32_l2_factorial_seed2025` |
| 4 | EvolveGCN-H h32/L2 | 6 | 42 | `evolvegcn_h_u500_top100_norm-minmax_k6_h32_l2_factorial_seed42` |
| 5 | EvolveGCN-H h32/L2 | 6 | 123 | `evolvegcn_h_u500_top100_norm-minmax_k6_h32_l2_factorial_seed123` |
| 6 | EvolveGCN-H h32/L2 | 6 | 2025 | `evolvegcn_h_u500_top100_norm-minmax_k6_h32_l2_factorial_seed2025` |
| 7 | EvolveGCN-H h32/L2 | 12 | 42 | `evolvegcn_h_u500_top100_norm-minmax_k12_h32_l2_factorial_seed42` |
| 8 | EvolveGCN-H h32/L2 | 12 | 123 | `evolvegcn_h_u500_top100_norm-minmax_k12_h32_l2_factorial_seed123` |
| 9 | EvolveGCN-H h32/L2 | 12 | 2025 | `evolvegcn_h_u500_top100_norm-minmax_k12_h32_l2_factorial_seed2025` |
| 10 | Static GCN h32/L3 | 4 | 42 | `static_gcn_u500_top100_norm-minmax_k4_h32_l3_gpool-mean_conv-gcn_batch8_factorial_seed42` |
| 11 | Static GCN h32/L3 | 4 | 123 | `static_gcn_u500_top100_norm-minmax_k4_h32_l3_gpool-mean_conv-gcn_batch8_factorial_seed123` |
| 12 | Static GCN h32/L3 | 4 | 2025 | `static_gcn_u500_top100_norm-minmax_k4_h32_l3_gpool-mean_conv-gcn_batch8_factorial_seed2025` |
| 13 | Static GCN h32/L3 | 6 | 42 | `static_gcn_u500_top100_norm-minmax_k6_h32_l3_gpool-mean_conv-gcn_batch8_factorial_seed42` |
| 14 | Static GCN h32/L3 | 6 | 123 | `static_gcn_u500_top100_norm-minmax_k6_h32_l3_gpool-mean_conv-gcn_batch8_factorial_seed123` |
| 15 | Static GCN h32/L3 | 6 | 2025 | `static_gcn_u500_top100_norm-minmax_k6_h32_l3_gpool-mean_conv-gcn_batch8_factorial_seed2025` |
| 16 | Static GCN h32/L3 | 8 | 42 | `static_gcn_u500_top100_norm-minmax_k8_h32_l3_gpool-mean_conv-gcn_batch8_factorial_seed42` |
| 17 | Static GCN h32/L3 | 8 | 123 | `static_gcn_u500_top100_norm-minmax_k8_h32_l3_gpool-mean_conv-gcn_batch8_factorial_seed123` |
| 18 | Static GCN h32/L3 | 8 | 2025 | `static_gcn_u500_top100_norm-minmax_k8_h32_l3_gpool-mean_conv-gcn_batch8_factorial_seed2025` |
| 19 | Static GCN h32/L3 | 12 | 42 | `static_gcn_u500_top100_norm-minmax_k12_h32_l3_gpool-mean_conv-gcn_batch8_factorial_seed42` |
| 20 | Static GCN h32/L3 | 12 | 123 | `static_gcn_u500_top100_norm-minmax_k12_h32_l3_gpool-mean_conv-gcn_batch8_factorial_seed123` |
| 21 | Static GCN h32/L3 | 12 | 2025 | `static_gcn_u500_top100_norm-minmax_k12_h32_l3_gpool-mean_conv-gcn_batch8_factorial_seed2025` |

The exact generated training commands are encoded deterministically in the two family specifications and were printed by the successful dry runs. Static commands explicitly pass the seed-matched canonical split config; Evolve commands use the same seed and fixed split ratios.

## Status and monitoring

```bash
python3 scripts/status_experiment_family.py \
  --spec configs/experiment_families/canonical_evolvegcn_knn_by_hidden_dim_factorial_500u_top100.json

python3 scripts/status_experiment_family.py \
  --spec configs/experiment_families/canonical_static_gcn_knn_by_hidden_dim_factorial_500u_top100.json

tmux attach -t canonical-knn-hidden-factorial

tail -f "$(ls -1t logs/canonical-knn-hidden-factorial-*.log | head -1)"
```

Detach from tmux with `Ctrl-b`, then `d`.

Per-run logs are written under:

- `logs/canonical_evolvegcn_knn_hidden_factorial/`
- `logs/canonical_static_gcn_knn_hidden_factorial/`

## Post-completion verification

```bash
python3 scripts/verify_experiment_family.py \
  --spec configs/experiment_families/canonical_evolvegcn_knn_by_hidden_dim_factorial_500u_top100.json

python3 scripts/verify_experiment_family.py \
  --spec configs/experiment_families/canonical_static_gcn_knn_by_hidden_dim_factorial_500u_top100.json
```

The final acceptance condition is 24/24 verified rows in each family, matching split signatures for every seed, zero partial rows, and no compatibility or metric-recomputation errors.
