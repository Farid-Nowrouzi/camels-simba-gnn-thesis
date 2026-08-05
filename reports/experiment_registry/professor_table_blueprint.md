# Professor Table Blueprint

All performance entries should be test MAE mean ± sample SD over independent
seeds, followed by seed count. Individual seed values belong in an appendix.
Never average protocol-incompatible rows.

## Table 1 — Historical progression from 20U to 750U

- **Question:** How did the research design and evidence evolve?
- **Rows:** major 20, 50, 100, 200, 500 Top100; 500 Top200; 500 Top500-none;
  750 Top1000-none stages, separated by model.
- **Columns:** U, Top-N, norm, model, h, layers, k, graph/temporal pooling,
  head, epochs, batch, seeds, valid MAE, purpose, comparison quality, lesson.
- **Include:** complete major research milestones.
- **Exclude:** debug/archive/duplicates and any implication that 750U is pure
  scaling.
- **Figure:** annotated timeline with protocol-change markers.
- **Evidence:** registry, raw configs/metrics, `thesis_completion_blueprint.md`.
- **Missing:** none for historical narrative.
- **Interpretation:** design progressed from sample-size pilots to halo,
  normalization and readout studies; protocol changes explain why the entire
  history is not one controlled curve.

## Table 2 — Controlled Top100 universe scaling

- **Question:** Does MAE change with U when settings are fixed?
- **Rows:** Evolve H and Static GCN × U={20,50,100,200,500}.
- **Columns:** model, U, Top-N, norm, k, h, layers, batch, epochs, seeds,
  MAE ± SD, compatibility.
- **Include:** only canonical three-seed 300-epoch rows with fixed
  model-specific protocol and 70/15/15 seed-generated splits.
- **Exclude:** legacy 200-epoch rows; 750U; Top200/500/1000.
- **Figure:** two-line MAE-versus-U plot with SD error bars.
- **Evidence:** `controlled_experiment_matrix.csv` and experiment metrics.
- **Missing:** 6 Evolve low-U, 6 Static low-U, and (strictly) 3 Static 500U
  B8 runs.
- **Interpretation:** with present data, only seed 123 supports an Evolve
  cross-U line; do not publish a three-seed clean curve until gaps are filled.

## Table 3 — Controlled 500U Top-N scaling

- **Question:** Does retaining more halos improve prediction?
- **Rows:** each model × Top-N={100,200,500}.
- **Columns:** model, Top-N, dataset, norm, k, h, layers, pool/head, training,
  seeds, split match, MAE ± SD.
- **Include:** minmax, k8, seed-matched 500U families.
- **Exclude:** 750U Top1000 and normalization-none rows.
- **Figure:** MAE versus Top-N (log-scaled x-axis), model lines.
- **Evidence:** Top100 final32, Top200, and Top500 minmax configs/metrics.
- **Missing:** none for Evolve; Static Top100 L3/B8 seeds 42/123/2025.
- **Interpretation:** Evolve shows a modest observed decline; Static cannot be
  interpreted across Top-N until architecture is matched.

## Table 4 — Periodic-kNN neighborhood ablation

- **Question:** How sensitive are models to k?
- **Rows:** model × k={4,6,8,12}.
- **Columns:** model, k, h (explicitly 64), layers, Top-N, norm, fixed settings,
  seeds, MAE ± SD.
- **Include:** completed h64 U500 Top100 minmax seed-matched families.
- **Exclude:** h32 scaling runs and absolute cross-model architecture claims.
- **Figure:** MAE versus k with SD bars.
- **Evidence:** `experiments/{evolvegcn_h,static_gcn}_500u_k*`.
- **Missing:** none.
- **Interpretation:** differences are smaller than seed variation; k=8 is a
  reasonable fixed default, not a demonstrated major performance driver.

## Table 5 — Normalization ablation at 500U Top500

- **Question:** Does per-graph feature normalization remove predictive signal?
- **Rows:** each model × none/minmax/zscore.
- **Columns:** model, normalization, h, layers, seeds, MAE ± SD, delta to none.
- **Include:** five behaviorally compatible seeds 42/123/777/999/2025; normalize
  missing legacy defaults in the table annotation.
- **Exclude:** debug 50U and target-normalization pilots.
- **Figure:** grouped bars or seed-paired dot plot.
- **Evidence:** 500U Top500 normalization experiment metrics.
- **Missing:** none.
- **Interpretation:** none is decisively best (Evolve .065798; Static .045304);
  normalization removes useful absolute-scale information.

## Table 6 — Regression-head ablation at 750U Top1000

- **Question:** Does a linear head reduce MLP collapse and error?
- **Rows:** MLP, linear.
- **Columns:** head, architecture, seeds, MAE ± SD, prediction-spread diagnostic.
- **Include:** five matched none/k8/h32/L2/mean/mean seeds.
- **Exclude:** 500U single-seed linear/activation/target-normalization pilots.
- **Figure:** paired seed slope plot plus predicted-vs-true inset.
- **Evidence:** 750U MLP and linear-head metrics/predictions.
- **Missing:** none.
- **Interpretation:** linear improves mean MAE and is the established 750U
  readout, though the advantage is seed-dependent.

## Table 7 — Graph-pooling ablation at 750U Top1000

- **Question:** Do extrema improve graph readout?
- **Rows:** mean, mean_max.
- **Columns:** graph pool, embedding width, fixed settings, seeds, MAE ± SD.
- **Include:** five matched linear-head temporal-mean families.
- **Exclude:** single-seed 500U mean_max pilot.
- **Figure:** paired seed slope plot.
- **Evidence:** 750U linear and linear-meanmax metrics.
- **Missing:** none.
- **Interpretation:** mean_max clearly degrades MAE; simple mean pooling remains
  preferred.

## Table 8 — Temporal-pooling ablation at 750U Top1000

- **Question:** Is the final temporal embedding better than the mean?
- **Rows:** temporal mean, temporal last.
- **Columns:** temporal pool, fixed graph pool/head, seeds, MAE ± SD, per-seed
  delta.
- **Include:** five matched linear/graph-mean families.
- **Exclude:** Static GCN (no temporal pooling).
- **Figure:** paired seed slope plot.
- **Evidence:** 750U linear and temporal-last metrics.
- **Missing:** none.
- **Interpretation:** aggregate MAEs are nearly tied; do not claim a robust
  temporal-last advantage.

## Table 9 — Static GCN versus EvolveGCN-H

- **Question:** Does temporal modeling outperform a final-snapshot model?
- **Rows:** (A) established 500U Top100 matched family; (B) 500U Top500 none
  close match; (C) proposed 750U Top1000 close match.
- **Columns:** comparison tier, model, temporal access, U, Top-N, norm, h,
  layers, head, splits, seeds, MAE ± SD, confounders.
- **Include:** only seed/split/dataset-matched pairs; disclose architecture.
- **Exclude:** unmatched 750U Static placeholder and claims of a pure
  architecture effect.
- **Figure:** paired-seed model differences, faceted by comparison tier.
- **Evidence:** matched configs/metrics and split signatures.
- **Missing:** three 750U Static seeds (42/123/2025).
- **Interpretation:** current 500U none evidence favors Static final snapshot;
  this tests information pathway plus established architecture, not temporal
  history alone.

## Table 10 — Best graph models versus validated summary baselines

- **Question:** Do GNNs add value beyond simple distribution summaries?
- **Rows:** best matched Evolve, best matched Static, Ridge, Random Forest,
  Gradient Boosting on 500U Top500 none seed-42 split.
- **Columns:** method, input information, temporal access, training split,
  test MAE/RMSE, seed count, validation status.
- **Include:** exact same dataset and split; show all three classical models.
- **Exclude:** unmatched temporal summary baselines, post-hoc “winner” wording,
  and the baseline until raw-column provenance is confirmed.
- **Figure:** test predicted-versus-true panels or residual distributions for
  Static, Evolve, RF, and GB.
- **Evidence:** `summary_features_500u_none_top500_matched_seed42/metrics.json`,
  matched graph prediction CSVs, `baseline_validation_audit.md`.
- **Missing:** semantic raw-column/target provenance confirmation, not a run.
- **Interpretation:** if validated, distribution summaries dramatically
  outperform current GNNs and indicate that absolute halo-scale statistics are
  sufficient for much of the target signal.
