# kNN Ablation Reporting Plan

## Readiness

The controlled h64 family passes the audit. A future analysis specification
is justified, but was not created during this inspection-only task.

Recommended analysis name:

`controlled_static_vs_evolvegcn_knn_ablation_500u_top100_h64`

## Family definition

Use exactly:

- EvolveGCN-H h64/L2, batch 4;
- Static GCN h64/L3, batch 8;
- 500 universes, Top100, minmax;
- periodic kNN, box size 25;
- k values 4, 6, 8, and 12;
- seeds 42, 123, and 2025;
- the 24 rows marked `canonical_h64_compatible` in
  `knn_ablation_candidate_matrix.csv`.

Do not include k=2, h32 scaling anchors, or the older Static batch-4 h32
alternatives.

The comparison statement should be:

> Matched universe population, Top100/minmax periodic-kNN graph protocol,
> hidden dimension, seeds, and exact split IDs, with k varied within each
> model and intentional architecture, depth, temporal-input, head, and batch
> differences across models.

Cross-model classification: **mostly controlled**.

## Minimum useful outputs

1. Main results table by model and k, including seed-level values and
   mean ± sample SD.
2. Test MAE versus k with individual seed points and mean ± SD.
3. Test RMSE versus k.
4. Test R² versus k, retaining negative values and showing R²=0.
5. Static-minus-Evolve paired MAE difference by k, matched by seed.
6. Between-seed variability versus k.
7. Prediction-SD/target-SD ratio versus k with a ratio=1 reference.
8. Representative true-versus-predicted and residual panels for selected k
   values using median-MAE seeds rather than best seeds.
9. Scientific summary and compatibility report.

Every figure must have an accompanying plotting-data CSV. Error bars must be
sample standard deviations across the three seeds, not pooled test-sample
uncertainty.

## Required validation gates

Before generation:

- require all 24 artifact-complete rows;
- verify dataset metadata rather than infer k from names;
- require the fixed within-model settings documented by the audit;
- require exactly one split signature per seed across all k values and both
  models;
- recompute MAE, RMSE, R², and Pearson from predictions;
- fail on missing, partial, duplicate-counted, or incompatible rows;
- retain negative R² and undefined correlation diagnostics;
- record the h32 exclusions and reasons in the manifest.

## Interpretation requirements

The report should state that k=8 has the lowest observed mean MAE for both
models, while differences across k are smaller than seed variability and are
not monotonic. It should avoid claiming that k=8 is universally optimal.

Prediction-collapse diagnostics are essential. Several Static h64 rows have
very low prediction dispersion and high exact-repeat fractions, including
over 95% repeats for two seed-42 rows. The main table should therefore be
interpreted together with prediction-SD ratio and representative prediction
plots.

## Highest-value next action

Create two generic experiment-family JSON specifications—one for EvolveGCN-H
and one for Static GCN—grouped by `k`, each covering 4×3 rows. Validate them
with the existing family verifier, then create the paired analysis
specification using the reporting pipeline. No new training is required.
