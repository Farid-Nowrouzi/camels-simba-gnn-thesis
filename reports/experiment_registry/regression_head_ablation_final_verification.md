# Regression-head ablation final verification

## Outcome

Phase B is allowed and complete. Ten existing EvolveGCN-H experiments form a
controlled U750 Top1000 comparison of `linear` and `mlp` regression heads at
seeds 42, 123, 777, 999, and 2025. No training, graph rebuilding, dataset
deserialization, or checkpoint loading was required.

## Canonical membership

### Linear

- `evolvegcn_h_750u_top1000_h32_seed42_none_linear_head`
- `evolvegcn_h_750u_top1000_h32_seed123_none_linear_head`
- `evolvegcn_h_750u_top1000_h32_seed777_none_linear_head`
- `evolvegcn_h_750u_top1000_h32_seed999_none_linear_head`
- `evolvegcn_h_750u_top1000_h32_seed2025_none_linear_head`

### MLP

- `evolvegcn_h_750u_top1000_h32_seed42_none`
- `evolvegcn_h_750u_top1000_h32_seed123_none`
- `evolvegcn_h_750u_top1000_h32_seed777_none`
- `evolvegcn_h_750u_top1000_h32_seed999_none`
- `evolvegcn_h_750u_top1000_h32_seed2025_none`

## Fixed protocol

All runs use EvolveGCN-H, the same temporal U750 LH population, Top1000
raw-Mvir halo selection, five snapshots, the seven canonical node features,
no feature normalization after `log10(Mvir)`, periodic kNN with k=8 and box
size 25, hidden dimension 32, two graph-convolution layers, batch size 4,
graph mean pooling, temporal mean pooling, ReLU encoder activation, dropout
0.2, MSE loss, AdamW at learning rate 0.001 and weight decay 1e-5, gradient
clipping at 1.0, raw Omega_m targets, no summary features, and identical
seed-specific ordered splits. Only `head_type` differs.

## Artifact verification

For every canonical experiment:

- `config.json` exists and parses;
- `metrics.json` exists and parses;
- `train_log.csv` exists and parses;
- `predictions/test_predictions.csv` exists and parses;
- `checkpoints/best_model.pt` exists.

Checkpoint verification was existence-only. No checkpoint bytes were opened.
No graph `.pt` file was opened.

Every prediction file contains exactly 201 finite targets and predictions,
201 nonempty unique universe IDs, no duplicate IDs, no missing IDs, and exact
ordered agreement with the declared test split. The ten experiment paths,
prediction paths, and prediction SHA-256 hashes are unique.

## Split verification

Every run has 450 train, 99 validation, and 201 test IDs. Within each run the
three sets are disjoint and cover `LH_0` through `LH_749` exactly once.
Linear and MLP ordered split signatures match exactly for every seed:

| Seed | Ordered split signature |
|---:|---|
| 42 | `e27e32ef2d9482fbe0719a56555f66075f7a23e178c15848a0227660ea0df7d9` |
| 123 | `bd9367b034a0ac7ebc07fe79c19a62d51f17402cb7e5dba98d1748e5168f72f1` |
| 777 | `6e71569b7785aa0483b56e359c68bac69e9c997617880855b75fd2bedcd84dae` |
| 999 | `27af8142ba8da15f65baf969086cca0e3b72e105c29a41ac778d6c702a8cb678` |
| 2025 | `3f80461e5ec26ee906f6c1781f34d40877521ac854d93f1fb09783b18b8d24a7` |

## Metric recomputation

The analysis independently recomputed MAE, RMSE, MSE, R², Pearson where
defined, target and prediction means and sample standard deviations,
prediction-SD ratio, exact and approximate repeat fractions, unique counts,
prediction bounds and range, residual mean and sample SD, and maximum absolute
residual.

Saved primary metrics agree at absolute tolerance `1e-6`. The maximum observed
saved/recomputed discrepancy among the wider 38-row candidate audit was
`8.326672684688674e-17`; canonical rows also pass this tolerance. Pearson is
defined for all ten canonical rows. Poor predictions and repeated values are
retained.

## Head implementation and parameter counts

Both heads receive the same `[B,32]` representation after graph and temporal
pooling and produce `[B,1]`.

- Linear: `Linear(32,1)`, bias enabled, identity output, 33 parameters.
- MLP: `Linear(32,32) → ReLU → Dropout(0.2) → Linear(32,1)`, biases enabled,
  identity output, 1,089 parameters.

The modules use PyTorch Linear default initialization. The optimizer includes
all model parameters. Different head types require different checkpoint
structures, but no checkpoint was loaded.

No implementation bug invalidates the comparison. Historical configurations
without `head_type` depend on the backward-compatible MLP default.

## Results

| Head | Mean MAE ± SD | Mean RMSE ± SD | Mean R² ± SD | Mean prediction-SD ratio | Mean exact-repeat fraction |
|---|---:|---:|---:|---:|---:|
| linear | 0.055843 ± 0.008748 | 0.071136 ± 0.009807 | 0.627697 ± 0.102848 | 0.862457 | 0 |
| MLP | 0.061404 ± 0.005142 | 0.076820 ± 0.005965 | 0.571180 ± 0.057321 | 0.779290 | 0.061692 |

Using `linear - MLP`, mean paired ΔMAE is `-0.005561 ± 0.005265`,
mean ΔRMSE is `-0.005684 ± 0.006821`, and mean ΔR² is
`+0.056517 ± 0.072197`. Linear improves all three metrics for four seeds;
seed 777 favours MLP.

Mean paired prediction-SD-ratio change is `+0.083167 ± 0.092151`. Linear has
no repeated predictions. MLP has repeated values in seeds 42 and 777, reaching
an exact-repeat fraction of 0.228856 in seed 42.

The linear advantage is modest and comparable to between-seed variability.
It is compatible with a linearly accessible pooled signal, but does not prove
that mechanism. Mean validation-minus-train MAE gaps are similar
(`0.017388` linear; `0.016990` MLP), so the available train/validation evidence
does not support a specific claim that the MLP overfits.

## Alternatives and duplicates

The exact duplicate seed-42 MLP reproduction is retained in the candidate
matrix but excluded from independent evidence. Temporal-last linear runs,
mean_max linear runs, the U500 one-seed pilot, target-normalized runs, summary
hybrids, Static GCN anchors, and GraphSAGE anchors are alternative protocols
and are not mixed into the causal comparison.

## Final outputs and validation

- Analysis specification:
  `configs/analysis_reports/controlled_evolvegcn_regression_head_ablation_750u_top1000.json`
- Output directory:
  `reports/analysis/controlled_evolvegcn_regression_head_ablation_750u_top1000/`
- Seed, aggregate, paired, diagnostic, and representative row counts:
  10, 2, 5, 10, and 2.
- Representative policy: median test MAE; linear seed 123 and MLP seed 42.
- Requested figure triples: 16 of 16.
- Final analysis validator: PASS.

The manifest exactly covers all generated package files.

## Backward compatibility

The following packages were rebuilt and validated in an isolated temporary
mirror:

1. `controlled_static_vs_evolvegcn_universe_scaling_top100`;
2. `controlled_static_vs_evolvegcn_knn_ablation_500u_top100_h64`;
3. `controlled_knn_by_hidden_dim_factorial_500u_top100`;
4. `controlled_static_vs_evolvegcn_normalization_ablation_500u_top500`;
5. `controlled_evolvegcn_graph_pooling_ablation_750u_top1000`.

All five validators passed. Row counts and established scientific CSV values
were unchanged within absolute tolerance `1e-6`; manifests remained valid.
The temporary mirror was isolated from tracked report outputs.
