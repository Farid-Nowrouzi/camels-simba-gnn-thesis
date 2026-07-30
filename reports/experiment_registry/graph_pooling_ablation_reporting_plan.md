# Reporting plan: controlled EvolveGCN-H graph-pooling ablation

## Proposed analysis

Use the identifier:

`controlled_evolvegcn_graph_pooling_ablation_750u_top1000`

The identifier deliberately names only EvolveGCN-H. No controlled Static GCN
pooling contrast has been trained.

## Canonical selection

Select exactly the ten artifact-complete U750 Top1000 EvolveGCN-H runs:

- graph pooling: `mean`, `mean_max`;
- seeds: 42, 123, 777, 999, 2025;
- node normalization: none;
- periodic kNN: `k=8`, box size 25;
- hidden dimension: 32;
- Evolve layers: 2;
- batch size: 4;
- epochs/patience: 300/40;
- activation: ReLU;
- temporal pooling: mean;
- regression head: linear;
- no target normalization;
- no summary features.

Pair methods by seed and exact ordered train/validation/test split. Use:

`MAE_mean_max_minus_mean = MAE(mean_max) - MAE(mean)`

Negative values favour mean_max; positive values favour mean.

Exclude the U750 MLP-head mean runs, temporal-last runs, the exact seed-42 MLP
duplicate, the U500 single-seed pilot, hybrid summary runs, target-normalized
run, LeakyReLU run, Static mean-only anchors, and the Static GraphSAGE pilot.

## Required future tables

1. Protocol table.
2. Ten-row seed-level pooling results.
3. Two-row aggregate pooling results.
4. Five-row paired pooling differences.
5. Prediction-compression diagnostics.
6. Descriptive best-pooling summary.

Report MAE, RMSE, MSE, R², Pearson status, prediction-SD ratio, exact and
approximate repeated-prediction fractions, prediction range, and residual
diagnostics. Retain all seeds and any poor, negative-R², compressed, or
constant-prediction result.

## Required future figures

1. Test MAE by graph pooling.
2. Test RMSE by graph pooling.
3. Test R² by graph pooling.
4. Paired MAE differences.
5. Seed variability.
6. Prediction-SD ratio.
7. Repeated-prediction fraction.
8. Representative true versus predicted.
9. Representative residuals versus truth.
10. Representative prediction distributions.

Every figure should include PNG, PDF, and plot-data CSV. Show aggregate means
with seed SD and individual seed points.

Representative runs must be the median test-MAE seed within each pooling cell,
never the best seed.

## Scientific questions

The final report should answer:

1. Does mean_max improve MAE or RMSE over mean in matched seeds?
2. Is the sign consistent across all five seeds?
3. Is the effect larger than between-seed variability?
4. Does mean_max improve or worsen prediction dispersion?
5. Does either method produce repeated or constant predictions?
6. Does the evidence support mean pooling as a principal bottleneck?

The already verified descriptive result is that mean_max has higher MAE and
RMSE in every seed. The final package should report this result, not retest or
select a favourable subset.

## Success and failure criteria

Success for the richer readout would require consistent, practically
meaningful error reduction while preserving or increasing useful prediction
dispersion.

Failure is supported when mean_max worsens error, behaves inconsistently in
dispersion, or fails to reduce prediction compression. Under this protocol,
failure of simple mean_max does not prove that all learned or
distribution-aware readouts would fail.

## Model comparison policy

Do not add Static GCN to the analysis identifier or main comparison. Static
mean anchors may be mentioned only as evidence that Static alternatives were
not trained. Training ten Static max/mean_max extensions is optional future
work, not required to complete the existing Evolve result.

## Highest-value next action

Build and validate the lightweight final analysis package from the ten
existing runs. No training or graph rebuilding is required. Attention pooling
may be listed as future work, but should not be proposed as a prerequisite.
