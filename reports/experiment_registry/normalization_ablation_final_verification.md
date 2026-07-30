# Controlled normalization ablation: final verification

## Scope and outcome

The controlled U500 Top500 periodic-kNN (`k=8`) normalization ablation is
artifact-complete. No training or graph rebuilding was required or performed.
All 30 canonical runs were retained: 15 EvolveGCN-H and 15 Static GCN runs,
covering `none`, `minmax`, and `zscore` at seeds 42, 123, 777, 999, and 2025.

The authoritative selection is encoded in:

- `configs/experiment_families/canonical_evolvegcn_normalization_ablation_500u_top500.json`
- `configs/experiment_families/canonical_static_gcn_normalization_ablation_500u_top500.json`
- `configs/analysis_reports/controlled_static_vs_evolvegcn_normalization_ablation_500u_top500.json`

The generated analysis is:
`reports/analysis/controlled_static_vs_evolvegcn_normalization_ablation_500u_top500/`.

## Preflight

- Repository and branch: `/home/ml/thesis-camels`,
  `thesis-controlled-scaling`.
- The initial tracked diff check passed.
- `pgrep -af 'train_evolvegcn_h|train_static_gcn|run_experiment_family.py'`
  found no trainer.
- `tmux ls` showed `canonical-scaling` with an idle `bash` pane and
  `canonical-knn-hidden-factorial-r3` with a dead `bash` pane; neither pane was
  running a trainer. No tmux session was launched.
- All five authoritative audit files existed and parsed. The candidate matrix
  contained 47 candidates, including exactly 30 canonical reusable rows and no
  missing canonical row.

## Canonical experiment mappings

| Model | Normalization | Seed | Experiment |
|---|---|---:|---|
| EvolveGCN-H | none | 42 | `evolvegcn_h_500u_top500_h32_seed42_none_norm` |
| EvolveGCN-H | none | 123 | `evolvegcn_h_500u_top500_h32_seed123_none` |
| EvolveGCN-H | none | 777 | `evolvegcn_h_500u_top500_h32_seed777_none` |
| EvolveGCN-H | none | 999 | `evolvegcn_h_500u_top500_h32_seed999_none` |
| EvolveGCN-H | none | 2025 | `evolvegcn_h_500u_top500_h32_seed2025_none` |
| EvolveGCN-H | minmax | 42 | `evolvegcn_h_500u_top500_h32_seed42` |
| EvolveGCN-H | minmax | 123 | `evolvegcn_h_500u_top500_h32_seed123_minmax` |
| EvolveGCN-H | minmax | 777 | `evolvegcn_h_500u_top500_h32_seed777_minmax` |
| EvolveGCN-H | minmax | 999 | `evolvegcn_h_500u_top500_h32_seed999_minmax` |
| EvolveGCN-H | minmax | 2025 | `evolvegcn_h_500u_top500_h32_seed2025_minmax` |
| EvolveGCN-H | zscore | 42 | `evolvegcn_h_500u_top500_h32_seed42_zscore` |
| EvolveGCN-H | zscore | 123 | `evolvegcn_h_500u_top500_h32_seed123_zscore` |
| EvolveGCN-H | zscore | 777 | `evolvegcn_h_500u_top500_h32_seed777_zscore` |
| EvolveGCN-H | zscore | 999 | `evolvegcn_h_500u_top500_h32_seed999_zscore` |
| EvolveGCN-H | zscore | 2025 | `evolvegcn_h_500u_top500_h32_seed2025_zscore` |
| Static GCN | none | 42 | `static_gcn_500u_top500_final_snapshot_h32_seed42_none` |
| Static GCN | none | 123 | `static_gcn_500u_top500_final_snapshot_h32_seed123_none` |
| Static GCN | none | 777 | `static_gcn_500u_top500_final_snapshot_h32_seed777_none` |
| Static GCN | none | 999 | `static_gcn_500u_top500_final_snapshot_h32_seed999_none` |
| Static GCN | none | 2025 | `static_gcn_500u_top500_final_snapshot_h32_seed2025_none` |
| Static GCN | minmax | 42 | `static_gcn_top500_final_snapshot_same_split_h32_seed42` |
| Static GCN | minmax | 123 | `static_gcn_500u_top500_final_snapshot_h32_seed123_minmax` |
| Static GCN | minmax | 777 | `static_gcn_500u_top500_final_snapshot_h32_seed777_minmax` |
| Static GCN | minmax | 999 | `static_gcn_500u_top500_final_snapshot_h32_seed999_minmax` |
| Static GCN | minmax | 2025 | `static_gcn_500u_top500_final_snapshot_h32_seed2025_minmax` |
| Static GCN | zscore | 42 | `static_gcn_500u_top500_final_snapshot_h32_seed42_zscore` |
| Static GCN | zscore | 123 | `static_gcn_500u_top500_final_snapshot_h32_seed123_zscore` |
| Static GCN | zscore | 777 | `static_gcn_500u_top500_final_snapshot_h32_seed777_zscore` |
| Static GCN | zscore | 999 | `static_gcn_500u_top500_final_snapshot_h32_seed999_zscore` |
| Static GCN | zscore | 2025 | `static_gcn_500u_top500_final_snapshot_h32_seed2025_zscore` |

Selection used the candidate-matrix classification and the parsed experiment
configs, not directory names alone.

## Artifact and split verification

For every canonical mapping, `config.json` and `metrics.json` existed and
parsed; `train_log.csv`, `predictions/test_predictions.csv`, and
`checkpoints/best_model.pt` existed. The 30 checkpoint paths were checked only
for file existence and were never opened.

Every prediction file contained exactly 75 finite target/prediction rows with
75 unique universe IDs in the exact order recorded by `config.json`. For every
seed, the ordered train/validation/test lists were identical across both models
and all three normalization methods. Their sizes were 350/75/75, they were
pairwise disjoint, and together they covered every ID from `LH_0` through
`LH_499` exactly once. Thus all within-model and descriptive model pairs use
identical test rows.

## Metric verification

The analysis independently recomputed test MAE, RMSE, MSE, R², Pearson when
defined, target and prediction means and sample standard deviations,
prediction-SD ratio, exact and approximate repeated-prediction fractions,
prediction range, and residual mean and sample standard deviation.

The maximum absolute saved/recomputed differences were:

| Metric | Maximum absolute difference |
|---|---:|
| MAE | 0 |
| RMSE | 1.3877787807814457e-17 |
| MSE | 3.469446951953614e-18 |

All were within `1e-6`. All primary metrics were finite. Thirteen negative-R²
rows and all eight rows with exact repeated predictions were retained. Pearson
was undefined for one run,
`static_gcn_top500_final_snapshot_same_split_h32_seed42`, because its prediction
standard deviation was exactly zero; it remains `NaN` with status
`undefined_zero_prediction_variance`, not zero. Its exact and approximate
repeated-prediction fractions were both 0.9866666667.

## Normalization implementation

The seven features are `log10(Mvir), X, Y, Z, VX, VY, VZ`.

- `none`: Mvir remains log-transformed; the feature matrix is cast to float32
  without subsequent feature scaling.
- `minmax`: independently for every universe, snapshot, and feature,
  `x_scaled = (x - x_min) / (x_max - x_min)`.
- `zscore`: independently for every universe, snapshot, and feature,
  `x_scaled = (x - mean) / standard_deviation`, using the population standard
  deviation.

For minmax and zscore, an absolute denominator below `1e-8` is replaced by
1.0. Top500 selection uses raw Mvir before feature construction and
normalization. Periodic kNN uses a separate raw physical XYZ copy, `k=8`, and
box size 25, so normalization does not intentionally change topology.
Omega_m is not normalized, no summary features are used, and no
cross-universe scaler is fitted. There is no target or cross-split leakage.
The scientific limitation is that sample-local normalization removes absolute
between-universe feature-scale information.

## Analysis results

| Model | Method | Mean MAE | SD MAE | Mean R² | Mean prediction-SD ratio | Mean exact repeat fraction |
|---|---|---:|---:|---:|---:|---:|
| EvolveGCN-H | none | 0.065798 | 0.004922 | 0.472338 | 0.690695 | 0.032000 |
| EvolveGCN-H | minmax | 0.085000 | 0.004433 | 0.165206 | 0.483250 | 0 |
| EvolveGCN-H | zscore | 0.094548 | 0.006507 | 0.007733 | 0.253646 | 0 |
| Static GCN | none | 0.045304 | 0.003725 | 0.734660 | 0.845814 | 0 |
| Static GCN | minmax | 0.095986 | 0.006421 | -0.024472 | 0.110663 | 0.426667 |
| Static GCN | zscore | 0.099335 | 0.005608 | -0.103769 | 0.262136 | 0.056000 |

Minmax-minus-none mean paired MAE was +0.019202 for EvolveGCN-H and
+0.050682 for Static GCN. Zscore-minus-none was +0.028751 and +0.054031,
respectively. Every one of these 20 matched comparisons was positive, so none
had lower MAE in all matched seeds for both models. These paired effects exceed
the corresponding within-cell seed SDs. Normalized representations also had
lower mean prediction-SD ratios in both models. Repetition was especially
severe for Static minmax, whereas Evolve's normalized runs did not show exact
repetition despite their compressed prediction dispersion.

This is strong descriptive evidence that per-universe, per-snapshot
normalization is a major contributor to degraded performance and prediction
compression under the tested protocol, especially for Static GCN. It does not
show that normalization universally harms GNNs or that temporal processing
causes the descriptive cross-model difference.

## Output and compatibility validation

The final package contains 30 seed-level rows, 6 aggregate rows, 30
within-model paired-normalization rows, and 15 descriptive model-paired rows.
All six requested table sets and all 13 requested figures exist in every
requested format (PNG, PDF, and plot-data CSV). Representative runs are the
median-MAE seed in each model × normalization cell.

The generic builder, focused normalization rebuild, and extended validator all
passed. In isolated temporary repository mirrors, the following established
analyses rebuilt and validated, and their seed-level, aggregate, and paired
scientific CSV values remained unchanged within `1e-6`:

1. `controlled_static_vs_evolvegcn_universe_scaling_top100`
2. `controlled_static_vs_evolvegcn_knn_ablation_500u_top100_h64`
3. `controlled_knn_by_hidden_dim_factorial_500u_top100`

No graph dataset, checkpoint, prediction artifact, experiment config, metric
file, or training log was modified.
