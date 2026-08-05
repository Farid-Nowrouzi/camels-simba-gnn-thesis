# Reporting Plan: Fixed-k=8 Hidden Dimension

## Readiness

The proposed analysis name is:

`controlled_hidden_dim_ablation_500u_top100_k8_h32_vs_h64`

The EvolveGCN-H half is ready without training. The Static half is not yet a pure hidden-dimension ablation because h32 uses two layers and h64 uses three. Therefore the official combined family and analysis specifications should be created only after the three depth-matched Static h64 replacements are complete.

Future specification paths:

- `configs/experiment_families/canonical_evolvegcn_hidden_dim_ablation_500u_top100_k8.json`
- `configs/experiment_families/canonical_static_gcn_hidden_dim_ablation_500u_top100_k8.json`
- `configs/analysis_reports/controlled_hidden_dim_ablation_500u_top100_k8_h32_vs_h64.json`

No specification is created by this audit.

## Canonical repaired protocol

- Models: EvolveGCN-H and Static GCN
- Hidden dimensions: 32 and 64
- Grouping field: `hidden_dim`
- k: 8
- Universes: 500
- Top-N: 100
- Normalization: minmax
- Periodic kNN: true
- Box size: 25
- Seeds: 42, 123, 2025
- Layers: two within each model
- Evolve batch size: 4
- Static batch size: 8
- Exact ordered split IDs paired by seed

The three existing Evolve h32 and three Evolve h64 rows can be reused. The three existing canonical Static h32 rows can be reused. Three new Static h64 two-layer rows are needed for the clean protocol.

## Required outputs

1. Main results table by model and hidden dimension.
2. Test MAE for h32 versus h64 with seed points and mean ± sample SD.
3. Test RMSE for h32 versus h64.
4. Test R² for h32 versus h64, retaining negative values and a zero reference line.
5. Paired `MAE_h64 - MAE_h32` by model and seed.
6. Between-seed MAE variability by hidden dimension.
7. Prediction-standard-deviation ratio by hidden dimension, with a reference line at 1.
8. Repeated-prediction fraction by hidden dimension.
9. Median-MAE representative true-versus-predicted plots.
10. Median-MAE representative residual plots.
11. Protocol and compatibility report.
12. Cautious scientific summary.

## Paired-difference convention

Define:

`paired_mae_difference = MAE_h64 - MAE_h32`

- Negative values favour h64.
- Positive values favour h32.
- Pair only identical model/seed rows with identical ordered split IDs.

## Interpretation constraints

- Do not call the current Static comparison a pure hidden-dimension ablation.
- Do not attribute an Evolve width difference to capacity alone without discussing seed variability.
- Do not select a best seed for representative plots; use the median-MAE seed.
- Do not infer that h64 is universally better from a small negative mean paired difference.
- Retain prediction-compression and repeated-prediction diagnostics even when performance is poor.

## Highest-value next action

If a clean combined thesis result is valuable, run only the three Static GCN h64, two-layer, batch-size-8 replacements with the existing k=8 dataset and exact seed-specific splits. Do not build a full h32 k-grid and do not rebuild graphs.
