# Regression-head ablation reporting plan

## Decision

Proceed to Phase B using the complete U750 Top1000 EvolveGCN-H family.

## Canonical comparison

Compare `linear` and `mlp` at seeds 42, 123, 777, 999, and 2025. Pair by seed
and exact ordered split signature. Hold graph mean pooling, temporal mean
pooling, encoder, optimizer, raw target scale, and all data settings fixed.

Use `linear - mlp` differences. Report all seeds, aggregate mean/SD/median,
prediction dispersion and repetition, and median-MAE representative runs.
Avoid p-values for five seeds.

## Progression policy

The controlled progression table contains only MLP-to-linear under the same
protocol. Temporal-last and mean_max findings belong in a separately labelled
historical context table because those protocols change other factors.

## Outputs

Build `controlled_evolvegcn_regression_head_ablation_750u_top1000` with the
requested tables, sixteen figure triples, thesis section, summaries, and a
complete manifest. Validate it and the five established analyses in isolated
temporary mirrors. No training or graph rebuilding is required.
