# Final Verification — kNN × Hidden-Dimension Factorial

## Verdict

**PASS.** All 48 canonical factorial rows are artifact-complete, independently
verified, included in the final analysis, and mapped exactly once.

No training, checkpoint deserialization, graph-dataset loading, notebook
execution, or experiment-artifact modification occurred during finalization.

## Exact family status summaries

```text
Family: canonical_evolvegcn_knn_by_hidden_dim_factorial_500u_top100
Summary: total_required=24 complete=24 missing=0 partial=0 excluded=0 runnable=0
```

```text
Family: canonical_static_gcn_knn_by_hidden_dim_factorial_500u_top100
Summary: total_required=24 complete=24 missing=0 partial=0 excluded=0 runnable=0
```

## Artifact, prediction, and split checks

For every one of the 48 rows:

- `config.json`, `metrics.json`, `train_log.csv`,
  `predictions/test_predictions.csv`, and `checkpoints/best_model.pt` exist and
  are readable;
- checkpoints were checked only for file readability and were not deserialized;
- the prediction CSV has exactly 75 rows;
- targets and predictions are finite;
- universe IDs are nonempty and unique;
- ordered prediction IDs exactly equal the config's ordered `test_ids`;
- no prediction row or test ID is missing or duplicated;
- train, validation, and test IDs are disjoint;
- exact split signatures agree across every valid width and model pair;
- saved MAE, RMSE, and MSE agree with prediction-derived recomputation;
- recomputed MAE, RMSE, MSE, and R² are finite.

Metric agreement uses absolute tolerance `1e-6` and zero relative tolerance.

## Undefined Pearson policy

Pearson correlation remains a secondary diagnostic. It is represented as
blank/NaN, never as zero, when either sample standard deviation is at or below
the explicit absolute threshold `1e-12`.

Structured statuses are:

- `defined`
- `undefined_zero_prediction_variance`
- `undefined_zero_target_variance`
- `undefined_zero_target_and_prediction_variance`

Undefined Pearson passes only when targets and predictions are finite, ordered
IDs and splits are exact, artifacts are complete, and primary metric
recomputation passes. Non-finite values, missing rows, ID mismatches, primary
metric mismatches, and unexplained non-finite Pearson remain hard failures.

Approximate repeated predictions use absolute tolerance `1e-12`. Exact repeats
are reported separately.

## Undefined-Pearson and severe-collapse row

Exact experiment:

`static_gcn_u500_top100_norm-minmax_k4_h32_l3_gpool-mean_conv-gcn_batch8_factorial_seed42`

| Diagnostic | Value |
|---|---:|
| Test rows | 75 |
| Target sample SD | 0.10202514478236673 |
| Prediction sample SD | 0.0 |
| Prediction value | 0.2994593679904938 |
| Exact unique predictions | 1 |
| Approximate unique predictions (`1e-12`) | 1 |
| Exact repeated-prediction fraction | 0.9866666666666667 |
| Approximate repeated-prediction fraction | 0.9866666666666667 |
| Test MAE | 0.08639862438042958 |
| Test RMSE | 0.10163982028879918 |
| Test MSE | 0.010330653068339394 |
| Test R² | -0.005872375437060873 |
| Pearson status | `undefined_zero_prediction_variance` |

The target and prediction vectors, required primary metrics, universe IDs, and
split are valid. Pearson is mathematically undefined because the prediction
variance is exactly zero. The run is retained and classified as severe,
constant prediction collapse.

No other factorial row has undefined Pearson.

## Verifier regression checks

Five harmless temporary-fixture tests pass:

1. finite variable predictions produce finite Pearson and pass;
2. finite constant predictions produce an explicit warning and pass;
3. a NaN prediction fails;
4. a missing prediction row fails;
5. mismatched MAE and RMSE fail.

```text
Ran 5 tests in 0.011s
OK
```

## Family-verifier results

```text
EvolveGCN-H: PASS — all 24 required rows are verified.
Static GCN: PASS — all 24 required rows are verified.
Static warning:
factorial_cell=h32_k4, seed=42: test_pearson is undefined;
status=undefined_zero_prediction_variance;
target_std=0.10202514478236673; prediction_std=0; tolerance=1.0e-12
```

## Final analysis validation

Specification:

`configs/analysis_reports/controlled_knn_by_hidden_dim_factorial_500u_top100.json`

Output:

`reports/analysis/controlled_knn_by_hidden_dim_factorial_500u_top100/`

Validated counts:

- seed-level rows: 48;
- aggregate model × width × k cells: 16;
- paired width rows: 24;
- paired model rows: 24;
- prediction diagnostics: 48;
- unique experiment mappings: 48;
- unique prediction-file mappings: 48;
- median-MAE representative runs: 16.

The analysis validator reports split and metric/artifact compatibility PASS.
The package retains negative R² values, compressed runs, repeated predictions,
and the constant prediction row.

## h64 reproduction and backward compatibility

The new h64 slice passes exact membership and split-signature comparison with
the established h64 package. Seed-level MAE/RMSE/R², aggregate means/SDs, and
paired model differences agree within absolute tolerance `1e-6`; no
disagreement was found.

The existing universe-scaling and h64 analyses were rebuilt and validated:

- universe scaling: 30 seed rows, 10 aggregate rows, 15 pairs, PASS;
- h64 kNN analysis: 24 seed rows, 8 aggregate rows, 12 pairs, PASS.

Universe-scaling seed, aggregate, and pair CSV hashes remained byte-identical.
The h64 aggregate and pair CSV hashes remained byte-identical; the temporary
rebuilt seed CSV differed only in line endings, with identical scientific
values. Both established tracked packages were restored to their original
bytes after this compatibility check.

## Scientific summary

- k-cell mean-MAE differences are generally smaller than between-seed
  variability. Moving from k=4 to k=12 does not consistently improve either
  model at h32 or h64.
- Descriptively lowest mean MAE under the tested protocol:
  EvolveGCN-H h32 k=12; EvolveGCN-H h64 k=8; Static GCN h32 k=6; Static GCN
  h64 k=8.
- EvolveGCN-H h64 beats h32 in 5/12 matched rows; its mean
  h64-minus-h32 MAE is +0.000604. Static GCN h64 beats h32 in only 1/12 rows;
  its mean difference is +0.001288. Width 64 therefore provides no consistent
  improvement.
- Width effects vary with k and sometimes change sign. This is descriptive
  evidence of a k × hidden-dimension interaction, not an inferential claim
  with only three seeds.
- In the descriptive cross-model comparison, EvolveGCN-H has lower MAE in
  10/24 matched rows and Static GCN in 14/24. Neither protocol consistently
  dominates.
- Prediction compression occurs across both widths, especially for Static GCN.
  Several Static rows contain repeated predictions, and the h32/k4/seed42 row
  is exactly constant. Denser connectivity does not consistently resolve
  compression.
