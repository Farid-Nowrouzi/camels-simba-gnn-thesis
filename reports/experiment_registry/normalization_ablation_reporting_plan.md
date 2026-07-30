# Reporting plan: controlled Static GCN and EvolveGCN-H normalization ablation

## Recommended future analysis

Use the identifier `controlled_static_vs_evolvegcn_normalization_ablation_500u_top500`.
The analysis contains the two complete model-specific families under one
model-stratified report. Normalization effects must be estimated within model;
the report must not present the Static-versus-Evolve difference as an effect of
normalization because temporal access, depth, batch size, and model mechanics
differ.

No training or graph rebuilding is required. The future analysis may consume
the 30 existing configs, metrics files, and test-prediction CSVs listed as
`reusable_existing` in the candidate matrix.

## Scientific question and decision rule

Hypothesis: changing the scale of the seven node features changes optimization
and learned representation scale, and can therefore change Omega_m prediction
accuracy and prediction compression.

The primary outcomes are test MAE, RMSE, and R². Diagnostics are the
prediction-SD ratio, exact repeated-prediction fraction, prediction range,
representative true-versus-predicted and residual plots, and across-seed
variability.

Call a method preferable only when its improvement is consistent across matched
seeds, practically meaningful relative to seed variability, and not bought at
the cost of stronger prediction compression. If differences are inconsistent,
smaller than seed variability, or accompanied by collapse, report no reliable
winner.

## Core tables

1. Protocol table: U500, Top500 selected by raw Mvir, seven features, periodic
   kNN k=8, box size 25, node-feature normalization implementation, split
   policy, and the model-specific architectures.
2. Seed-level normalization results: one row for each of 30
   model × normalization × seed cells with MAE, RMSE, MSE, R², Pearson status,
   prediction-SD ratio, and exact repeated-prediction fraction.
3. Aggregate normalization results: mean, standard deviation, median, minimum,
   and maximum across the five seeds, stratified by model and normalization.
4. Paired normalization differences: within model and seed, report none-minus-
   minmax, none-minus-zscore, and minmax-minus-zscore for MAE, RMSE, and R².
5. Descriptive best-normalization summary: seed wins and aggregate ranking,
   explicitly labeled descriptive rather than a significance test.
6. Prediction-collapse diagnostics: prediction and target SD, ratio, range,
   repeated fraction, and undefined-Pearson reason.

## Core figures

Create model-faceted figures for test MAE, RMSE, and R² versus normalization;
paired MAE differences; seed variability; prediction-SD ratio; and exact
repeated-prediction fraction. Also create representative
true-versus-predicted, residual, and prediction-distribution panels.

Show individual seeds in every aggregate performance plot. Do not connect
unmatched observations.

## Representative-seed policy

For each model × normalization cell, select the seed with median test MAE.
Never select the best seed. If two candidates are equally close to the median,
use the lower numerical seed and disclose the tie rule.

## Required caveats

- `none` still uses log10(Mvir); it means no subsequent node-feature scaling.
- Minmax and zscore are computed separately for every universe and snapshot,
  not fitted on a training split or globally across universes.
- Static GCN consumes the final snapshot from the same temporal source
  datasets; it does not use an independently built Static Top500 family.
- Code and metadata prove the same topology-generating protocol and raw
  positions, but no stored topology hash proves byte-identical adjacency
  tensors without a later read-only graph check.
- Legacy seed-42 minmax configs omit several now-explicit defaults. Resolve
  Evolve activation/head as ReLU/MLP and Static convolution/self-loops as
  GCN/enabled, and preserve an annotation that these are behaviorally resolved
  defaults.
- Pearson is undefined—not zero—when either vector has effectively zero
  variance under the repository threshold.

## Interpretation templates

- If `none` remains consistently better, state that per-snapshot scaling may
  remove useful absolute-scale or distributional information under these
  architectures.
- If minmax is better, state that bounded feature ranges may improve
  conditioning.
- If zscore is better, state that centered standardized inputs may improve
  conditioning.
- If no method is consistent, state that node-feature normalization is not the
  dominant source of model weakness.
- If a normalized family has lower prediction-SD ratio or more repeated
  predictions, state that it is associated with stronger compression. Do not
  claim causality beyond this controlled protocol.
