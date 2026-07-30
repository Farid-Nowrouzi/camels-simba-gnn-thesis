# Controlled EvolveGCN-H graph-pooling ablation: final verification

## Outcome

The canonical U750 Top1000 graph-pooling family is complete and independently
verified. It contains ten existing runs: mean and mean_max pooling at seeds 42,
123, 777, 999, and 2025. No training, graph rebuilding, checkpoint loading, or
graph-dataset deserialization was required or performed.

## Exact canonical membership

| Graph pooling | Seed | Experiment |
|---|---:|---|
| mean | 42 | `evolvegcn_h_750u_top1000_h32_seed42_none_linear_head` |
| mean | 123 | `evolvegcn_h_750u_top1000_h32_seed123_none_linear_head` |
| mean | 777 | `evolvegcn_h_750u_top1000_h32_seed777_none_linear_head` |
| mean | 999 | `evolvegcn_h_750u_top1000_h32_seed999_none_linear_head` |
| mean | 2025 | `evolvegcn_h_750u_top1000_h32_seed2025_none_linear_head` |
| mean_max | 42 | `evolvegcn_h_750u_top1000_h32_seed42_none_linear_head_meanmax` |
| mean_max | 123 | `evolvegcn_h_750u_top1000_h32_seed123_none_linear_head_meanmax` |
| mean_max | 777 | `evolvegcn_h_750u_top1000_h32_seed777_none_linear_head_meanmax` |
| mean_max | 999 | `evolvegcn_h_750u_top1000_h32_seed999_none_linear_head_meanmax` |
| mean_max | 2025 | `evolvegcn_h_750u_top1000_h32_seed2025_none_linear_head_meanmax` |

Membership was resolved from the candidate matrix and independently confirmed
against parsed configs. Every row uses EvolveGCN-H, 750 universes, Top1000,
five temporal snapshots, no node-feature scaling after log10(Mvir), periodic
kNN with k=8 and box size 25, hidden dimension 32, two layers, batch size 4,
ReLU, temporal mean pooling, a linear head, no target normalization, and no
summary features.

## Artifact verification

For all ten runs:

- `config.json` and `metrics.json` exist and parse;
- `train_log.csv` exists and parses;
- `predictions/test_predictions.csv` exists and parses;
- `checkpoints/best_model.pt` exists.

Checkpoint paths were tested only for existence. No checkpoint was opened.

Each prediction file contains exactly 201 rows, finite targets and
predictions, 201 unique IDs, and exact ordered agreement with the declared
test split. All ten experiment paths, prediction paths, and prediction hashes
are unique.

## Split verification

Every config contains 450 training, 99 validation, and 201 test IDs. The three
sets are disjoint and cover `LH_0` through `LH_749` exactly once. Mean and
mean_max use identical ordered splits within every seed:

| Seed | Ordered split signature |
|---:|---|
| 42 | `e27e32ef2d9482fbe0719a56555f66075f7a23e178c15848a0227660ea0df7d9` |
| 123 | `bd9367b034a0ac7ebc07fe79c19a62d51f17402cb7e5dba98d1748e5168f72f1` |
| 777 | `6e71569b7785aa0483b56e359c68bac69e9c997617880855b75fd2bedcd84dae` |
| 999 | `27af8142ba8da15f65baf969086cca0e3b72e105c29a41ac778d6c702a8cb678` |
| 2025 | `3f80461e5ec26ee906f6c1781f34d40877521ac854d93f1fb09783b18b8d24a7` |

## Metric recomputation

The verifier recomputed MAE, RMSE, MSE, R², Pearson, target and prediction
means and sample SDs, prediction-SD ratio, unique-prediction count, exact and
approximate repeat fractions, prediction minimum/maximum/range, residual mean
and sample SD, and maximum absolute residual.

Tolerances:

- saved primary metrics: absolute tolerance `1e-6`;
- undefined Pearson variance threshold: `1e-12`;
- approximate repeated-prediction tolerance: `1e-12`.

Maximum saved/recomputed absolute differences:

| Metric | Maximum difference |
|---|---:|
| MAE | 0 |
| RMSE | 8.326672684688674e-17 |
| MSE | 2.0816681711721685e-17 |

All primary metrics are finite. Pearson is mathematically defined for all ten
runs. Two negative-R² mean_max rows, every poor seed, and the maximum observed
absolute residual of 1.0181953609 are retained.

## Prediction-dispersion and repetition diagnostics

| Graph pooling | Mean prediction-SD ratio | SD ratio | Mean exact-repeat fraction | Undefined Pearson |
|---|---:|---:|---:|---:|
| mean | 0.862457 | 0.134882 | 0 | 0 |
| mean_max | 0.863478 | 0.252198 | 0 | 0 |

No run contains exact or approximate repeated predictions. Aggregate
dispersion is nearly unchanged, although seed-level ratio changes are
inconsistent.

## Paired scientific result

The convention is method(mean_max) minus method(mean).

| Seed | ΔMAE | ΔRMSE | ΔR² | Δprediction-SD ratio |
|---:|---:|---:|---:|---:|
| 42 | +0.041155 | +0.050563 | -0.601621 | -0.009396 |
| 123 | +0.023050 | +0.049078 | -0.689408 | -0.027855 |
| 777 | +0.024843 | +0.037934 | -0.574945 | +0.495616 |
| 999 | +0.026894 | +0.029839 | -0.357005 | -0.237654 |
| 2025 | +0.009709 | +0.007757 | -0.087763 | -0.215602 |

Mean paired ΔMAE is +0.025130 with seed SD 0.011204. All five MAE and
RMSE differences are positive, and all five R² differences are negative.
Mean pooling therefore has lower error and higher R² in every matched pair.

## Implementation verification

Source inspection confirms:

- mean computes the masked sum divided by real-node count and returns
  `[B, T, H]`;
- mean_max concatenates the masked mean and maximum and returns
  `[B, T, 2H]`;
- at H=32, graph embeddings are 32 and 64 wide;
- the linear mean_max head has exactly 32 necessary additional input weights;
- temporal pooling is mean and the head type is linear in all ten runs;
- both pooling methods are permutation invariant and applied independently
  for every graph and snapshot.

The preprocessing validation prevents zero-real-node snapshots. If that
guarantee were bypassed, masked maximum has a latent all-masked edge case; it
does not affect these verified runs. Attention and hierarchical pooling remain
unimplemented and untested.

## Final verifier result

The analysis validator passes:

- 10 seed-level rows;
- 2 aggregate rows;
- 5 paired-pooling rows;
- 10 prediction-diagnostic rows;
- 2 median-MAE representative rows;
- exact split pairing;
- unique mappings and prediction hashes;
- finite metrics and saved/recomputed agreement;
- correct mean then mean_max order;
- all required tables and figure triples;
- exact manifest coverage.

The final analysis specification is
`configs/analysis_reports/controlled_evolvegcn_graph_pooling_ablation_750u_top1000.json`;
the output directory is
`reports/analysis/controlled_evolvegcn_graph_pooling_ablation_750u_top1000/`.

## Backward compatibility

Four established analyses were rebuilt and validated in isolated temporary
mirrors:

1. `controlled_static_vs_evolvegcn_universe_scaling_top100`;
2. `controlled_static_vs_evolvegcn_knn_ablation_500u_top100_h64`;
3. `controlled_knn_by_hidden_dim_factorial_500u_top100`;
4. `controlled_static_vs_evolvegcn_normalization_ablation_500u_top500`.

Their established seed-level, aggregate, and paired scientific CSV values
remained unchanged within `1e-6`; row counts and manifests passed their
validators. No existing repository analysis output was regenerated in place.

## Scientific conclusion

Under the controlled U750 Top1000 EvolveGCN-H protocol, directly appending
maximum node embeddings worsened MAE, RMSE, and R² in every matched seed.
Prediction dispersion remained similar in aggregate and neither method
repeated predictions, so the degradation was not primarily an
additional-collapse effect. This result applies only to simple mean versus
mean-and-maximum concatenation and does not establish that mean is universally
optimal or that learned attention pooling would fail.
