# Normalization-ablation control audit

Audit date: 2026-07-30

Repository: `/home/ml/thesis-camels`

Branch: `thesis-controlled-scaling`

Mode: inspection only; no checkpoints or graph datasets were loaded

## 1. Executive summary

Outcome **C** applies: complete, controlled five-seed normalization families
exist for both EvolveGCN-H and Static GCN. Each family contains none, minmax,
and zscore at seeds 42, 123, 777, 999, and 2025. All 30 canonical cells have a
config, metrics, training log, test predictions, and checkpoint. Their exact
ordered train/validation/test splits match across normalization methods within
each seed and also across the two models.

The recommended design is therefore two parallel within-model comparisons in
one model-stratified analysis:
`controlled_static_vs_evolvegcn_normalization_ablation_500u_top500`.
No graph rebuilding, replacement run, or new training is required.

The audit discovered 47 candidates in total: 30 canonical reusable cells, 12
artifact-complete alternative/debug cells, and 5 metrics-only summary-feature
pilots. The candidate matrix deliberately retains poor and collapsed runs.
None is descriptively best in both canonical families. This is particularly
important for Static GCN, where minmax and zscore show severe prediction
compression. That finding is an association within this protocol; the final
analysis must not generalize it to all normalization designs.

## 2. Normalization methods found

| Raw evidence | Canonical meaning | Actual implementation | Safe to combine? |
|---|---|---|---|
| `none` | no scaling after feature construction | return the seven-feature float32 matrix unchanged | Yes, when all other fields match |
| experiment suffix `none_norm` | `none` | dataset metadata explicitly says `normalization: none` | Yes; the suffix is only a run-name variant |
| generic legacy seed-42 names without a normalization suffix | `minmax` | their authoritative dataset metadata says `normalization: minmax` | Yes, with resolved-default annotation |
| `minmax` | sample-local min-max scaling | for each feature, `(x - min) / (max - min)`; denominator becomes 1 when its absolute value is below `1e-8` | Yes, when all other fields match |
| `zscore` | sample-local standardization | for each feature, `(x - mean) / std`; standard deviation becomes 1 when its absolute value is below `1e-8` | Yes, when all other fields match |

No actual node-feature dataset labeled `raw`, `standard`, `standardized`,
`standardization`, `min_max`, or `z_score` was found. Those words must not be
treated as aliases without new evidence. In particular, `none` is not raw Mvir:
Mvir is first transformed to log10(Mvir), which is the feature definition.

Across all 47 candidates, raw metadata labels occur as 20 minmax, 13 none, and
14 zscore rows. These totals include excluded architectural and summary-feature
pilots and are not the canonical cell counts.

## 3. Exact implementation of each method

`src/data/camels_graph_utils.py` constructs
`[log10(Mvir), X, Y, Z, VX, VY, VZ]`, then calls `normalize_features`.
The implementation dates to the initial pipeline commit and is still the
implementation used by the metadata-defined preprocessing protocol.

- None performs only a float32 cast.
- Minmax computes a seven-element minimum and maximum over the selected nodes
  of one snapshot and scales each feature independently to [0, 1].
- Zscore computes a seven-element population mean and population standard
  deviation over the selected nodes of one snapshot and standardizes each
  feature independently.
- Both scaled methods guard a zero or near-zero denominator with `eps=1e-8`.
- Scaling parameters are not persisted in dataset sidecars.

## 4. Feature-level normalization scope

All seven model inputs are transformed by minmax or zscore:
log10(Mvir), X, Y, Z, VX, VY, and VZ. The operation is per feature, per
snapshot, and per universe after raw-Mvir Top-N selection.

The processing order is:

1. clean invalid halos;
2. rank and select Top-N using raw Mvir;
3. construct log10(Mvir) plus raw positions and velocities;
4. copy raw physical XYZ for topology;
5. normalize the seven-feature ML matrix;
6. pad if needed;
7. construct adjacency from the separate raw-position copy.

## 5. Fit scope and leakage assessment

Node normalization is neither global over 500 universes nor fitted on the
training split. Every universe/snapshot supplies its own feature-wise
statistics. Consequently:

- validation and test labels do not influence normalization;
- training universes do not influence validation/test scaling;
- validation/test universes do not influence training scaling;
- no full-dataset fitted scaler exists;
- each inference graph uses its own observable node values to calculate its
  transform.

This is not target leakage or cross-split leakage. It is nevertheless a
scientifically consequential sample-local transform: it intentionally removes
or compresses between-universe absolute scale information, and its test-time
behavior differs from a train-fitted global scaler. The future report must call
it “per-universe, per-snapshot normalization,” not generic dataset
normalization.

The alternative `hybrid_summary_norm` run fits its summary-feature scaler on
the training split only. The alternative `linear_head_targetnorm` run fits the
Omega_m mean and standard deviation on the training split only. Neither is part
of the node-feature-only family.

## 6. Graph-topology assessment

The builder selects nodes using raw Mvir and constructs periodic kNN adjacency
from raw physical XYZ retained before feature normalization. Canonical dataset
sidecars agree on all scientific fields except normalization and output path:
500 successful LH universes, five snapshots, Top500, k=8, periodic boundaries,
box size 25, the same feature columns, the same raw-Mvir selection rule, and
`graph_positions: raw_physical_XYZ_before_feature_normalization`.

Therefore normalization does not enter the topology-generating calculation and
the edge lists **should** be identical across none/minmax/zscore. This is strong
protocol-level evidence, not a byte-level proof: no stored topology hash or
lightweight edge-list manifest exists, and this audit did not deserialize the
`.pt` datasets. If an empirical proof is later required, the smallest step is a
read-only utility that loads each dataset once and hashes ordered adjacency
tensors. Rebuilding graphs is not required.

## 7. Target-normalization assessment

All 30 canonical configs use unnormalized Omega_m. The recent explicit configs
record `normalize_target: false`; the legacy seed-42 anchors predate that
field, and their trainer behavior and prediction scale resolve to the same
unnormalized-target protocol.

One excluded Evolve seed-42 pilot uses a linear head, LeakyReLU, and
training-split target standardization with recorded mean 0.2998319864 and
standard deviation 0.1166983247. It is classified
`target_normalization_incompatible`.

## 8. Candidate experiment counts

The candidate inventory contains 47 rows:

- 24 EvolveGCN-H;
- 18 Static GCN;
- 5 summary-feature baselines;
- 39 at 500 universes and 8 at 50 universes;
- 37 explicitly Top500, 6 explicitly Top100, and 4 metrics-only rows whose
  Top-N is not authoritative in a config/sidecar row.

Compatibility classes are:

- 30 `canonical_normalization_compatible`;
- 5 `architecture_incompatible`;
- 1 `target_normalization_incompatible`;
- 8 `alternative_protocol`;
- 3 `artifact_incomplete`.

The last count does not equal all incomplete artifacts because two incomplete
debug summary rows are more informatively classified as
`alternative_protocol`.

## 9. Artifact completeness

Forty-two of 47 candidates are artifact-complete under the GNN requirement:
config, metrics, training log, test predictions, and checkpoint all exist.
All 30 canonical rows are complete. Five summary-feature pilots are
metrics-only and lack at least canonical configs, prediction CSVs, training
logs, or checkpoints; none enters the final family.

Checkpoint existence was checked by path only. No checkpoint was loaded.

## 10. Model protocols found

The canonical EvolveGCN-H protocol is U500, five snapshots, Top500, periodic
kNN k=8, box 25, h32, 2 layers, batch 4, 300 epochs, patience 40, learning rate
0.001, weight decay 1e-5, dropout 0.2, ReLU, mean graph pooling, mean temporal
pooling, MLP head, self-loops enabled, gradient clipping 1.0, and 70/15/15
splits. The raw catalogue contains the same five scale factors for every
canonical universe: 0.20000, 0.25000, 0.51209, 0.75065, and 1.00000.

The canonical Static GCN protocol uses the final snapshot extracted from the
same temporal Top500 datasets: U500, periodic kNN k=8, box 25, h32, 3 layers,
batch 8, 300 epochs, patience 40, learning rate 0.001, weight decay 1e-5,
dropout 0.2, GCN convolution, ReLU model operations, mean pooling, the fixed
Static MLP-style readout, self-loops enabled, gradient clipping 1.0, and the
same 70/15/15 splits. Its snapshot scale factor is 1.00000.

The legacy seed-42 minmax configs omit fields that later trainers write
explicitly. Trainer defaults and sibling configs resolve Evolve activation/head
to ReLU/MLP and Static convolution/self-loops to GCN/enabled. No other resolved
field differs within a model/seed triplet.

Historical alternatives include 50U Top100 dropout-zero/1000-epoch debug
protocols; Evolve hybrid-summary, LeakyReLU, linear-head, target-normalized, and
mean-max pilots; and metrics-only classical summary baselines.

## 11. Dataset protocols found

Candidate datasets cover:

- temporal 500U Top500 none/minmax/zscore, five snapshots;
- those same temporal datasets consumed at the final snapshot by Static GCN;
- temporal 50U Top100 none/minmax/zscore debug datasets;
- native Static 50U Top100 none/minmax/zscore debug datasets.

All graph candidates use the seven node features, periodic kNN with k=8, box
size 25, and raw-Mvir Top-N selection. Canonical families use CAMELS-SIMBA
`LH_0` through `LH_499`. No canonical native-Static Top500 dataset family
exists; Static intentionally reads the final snapshot of each matched temporal
dataset.

## 12. Split compatibility

For every canonical model × seed triplet:

- ordered train IDs are identical across none/minmax/zscore;
- ordered validation IDs are identical;
- ordered test IDs are identical;
- split sizes are 350/75/75;
- the three sets have no overlap;
- their union is exactly `LH_0` through `LH_499`;
- prediction rows exactly match ordered test IDs.

The same ordered split also matches across Static and Evolve for each seed.
There are five distinct deterministic seed splits: 42, 123, 777, 999, and
2025. No normalization method receives a different test population.

## 13. Duplicate and alternative-protocol findings

No two discovered prediction CSVs have the same SHA-256 hash. No canonical cell
is a duplicate reproduction, and no runs were counted as independent seeds
more than once.

The two generic seed-42 minmax names are anchors, not duplicates: each supplies
the sole minmax cell for its model at seed 42. The `none_norm` suffix is a
name-level variant whose dataset metadata proves node normalization `none`.

Seventeen rows are noncanonical:

- five architecture-incompatible Top500 Evolve pilots (hybrid summaries,
  LeakyReLU, linear head, or mean-max pooling);
- one target-normalization-incompatible Top500 Evolve pilot;
- six GNN debug runs at 50U Top100, where the `none` rows also change learning
  rate from 0.001 to 0.0001;
- five summary-feature pilots with different estimators or incomplete
  artifacts.

## 14. Metric recomputation result

All 42 prediction-bearing candidates have finite targets and predictions,
unique nonempty universe IDs, expected row counts, and ordered test IDs matching
their configs. MAE, RMSE, MSE, R², Pearson when defined, means, sample standard
deviations, SD ratio, exact repeated fraction, prediction range, and residual
statistics were recomputed directly from CSVs.

Every saved MAE and RMSE matches recomputation within the repository’s
established absolute tolerance of `1e-6`. All prediction files were hashed;
there are no hash duplicates.

Pearson follows the repository policy and is undefined rather than zero when a
target or prediction vector has standard deviation at or below `1e-12`.
Within the canonical family, Static minmax seed 42 has zero prediction
variance, so its Pearson is undefined.

## 15. Prediction-collapse findings

Five-seed descriptive means are:

| Model | Normalization | MAE | RMSE | R² | prediction-SD ratio | exact repeated fraction |
|---|---:|---:|---:|---:|---:|---:|
| EvolveGCN-H | none | 0.065798 | 0.080031 | 0.472338 | 0.690695 | 0.032000 |
| EvolveGCN-H | minmax | 0.085000 | 0.100902 | 0.165206 | 0.483250 | 0.000000 |
| EvolveGCN-H | zscore | 0.094548 | 0.110390 | 0.007733 | 0.253646 | 0.000000 |
| Static GCN | none | 0.045304 | 0.056607 | 0.734660 | 0.845814 | 0.000000 |
| Static GCN | minmax | 0.095986 | 0.112223 | -0.024472 | 0.110663 | 0.426667 |
| Static GCN | zscore | 0.099335 | 0.116358 | -0.103769 | 0.262136 | 0.056000 |

None wins descriptively for every seed in both models. Static minmax seed 42 is
effectively constant: prediction-SD ratio 0 and exact repeated fraction
74/75 = 0.986667. Several other Static normalized rows have negative R²,
strong compression, or repeated predictions; they remain in the evidence.

The Evolve none family has some exact repeats despite its better variance
ratio, so it should not be described as collapse-free. The final analysis must
show seed-level values and paired differences rather than relying only on these
means.

## 16. Possible controlled families

The family summary records seven discovered groupings. Three are scientifically
useful:

1. EvolveGCN-H U500 Top500: 15/15 complete cells.
2. Static GCN U500 Top500: 15/15 complete cells.
3. A combined model-stratified report containing both complete families:
   30/30 cells.

The 50U debug families have only one seed and their none rows change learning
rate. Summary-feature groupings have mismatched or unrecorded splits,
different estimators, missing prediction artifacts, or a missing none cell.
They are historical evidence only.

## 17. Recommended canonical family

Recommend the two complete U500 Top500 families under the shared analysis
identifier
`controlled_static_vs_evolvegcn_normalization_ablation_500u_top500`.
Primary normalization effects must be computed separately within EvolveGCN-H
and Static GCN. The cross-model alignment is useful for comparing qualitative
patterns, but architecture and temporal-input differences prevent a pooled
causal normalization estimate.

This recommendation satisfies the priority order: exact matched splits; only
node-feature normalization differs within model; identical topology protocol;
complete predictions; five seeds; established protocols; zero training; and a
direct thesis interpretation.

## 18. Existing reusable rows

All 30 model × normalization × seed rows are `reusable_existing`:

- EvolveGCN-H: none/minmax/zscore × seeds 42, 123, 777, 999, 2025;
- Static GCN: none/minmax/zscore × the same five seeds.

The candidate matrix gives the exact experiment and dataset paths, protocol
fields, split hashes, prediction hashes, recomputed metrics, and collapse
diagnostics for every row.

## 19. Missing or replacement rows

There are zero missing canonical cells, zero partial canonical cells, and zero
replacement rows. The missing-run decisions CSV therefore contains its header
and an audit-note comment but no data rows.

No replacement should be added merely to rename the legacy seed-42 minmax
anchors or make their now-default fields explicit. Their actual behavior is
compatible and documented.

## 20. Whether graph rebuilding is required

No. Existing canonical datasets and predictions are sufficient. Optional
future topology hashing would verify stored edge equality but would not rebuild
graphs and is not required to generate the final analysis.

## 21. Whether new training is required

No. Required rows: 30. Reusable rows: 30. Missing, partial, and replacement
rows: 0. Optional training extensions: 0 for the audited thesis question.

## 22. Proposed future analysis name

`controlled_static_vs_evolvegcn_normalization_ablation_500u_top500`

## 23. Highest-value next action

Generate the final analysis package from the 30 existing configs and prediction
CSVs, using model-stratified paired seed comparisons and the undefined-Pearson
policy. Do not train first.

## 24. Scientific caveats

- The result tests per-universe, per-snapshot scaling, not a global
  train-fitted scaler.
- `none` retains the log10 mass transform.
- Static uses only the final snapshot while Evolve uses five snapshots.
- Static and Evolve differ in depth, batch size, and architecture; compare
  normalization within model.
- Protocol identity strongly implies matching graph topology, but stored
  adjacency equality has no hash proof.
- Seed-42 legacy defaults are behaviorally resolved and must be annotated.
- Summary-feature and target-normalized pilots are different scientific
  interventions.
- The observed superiority of none and compression of normalized Static runs
  apply to this U500 Top500 k=8 protocol; avoid universal causal claims.

## 25. Exact final Git status

The final status is recorded after all five outputs and validation checks are
complete. Files created by this audit are the five
`normalization_ablation_*` files. All other untracked paths shown are
pre-existing. No tracked source file was modified.

```text
?? experiments/
?? reports/experiment_registry/analysis_artifact_index.csv
?? reports/experiment_registry/baseline_validation_audit.md
?? reports/experiment_registry/canonical_scaling_family.csv
?? reports/experiment_registry/canonical_scaling_live_status.csv
?? reports/experiment_registry/canonical_scaling_results/
?? reports/experiment_registry/canonical_static_scaling_results/
?? reports/experiment_registry/controlled_experiment_matrix.csv
?? reports/experiment_registry/current_thesis_execution_status.md
?? reports/experiment_registry/experiment_family_summary.csv
?? reports/experiment_registry/family_compatibility_audit.md
?? reports/experiment_registry/final_table_readiness_status.md
?? reports/experiment_registry/hidden_dim_h32_vs_h64_k8_audit.md
?? reports/experiment_registry/hidden_dim_h32_vs_h64_k8_candidate_matrix.csv
?? reports/experiment_registry/hidden_dim_h32_vs_h64_k8_missing_run_decisions.csv
?? reports/experiment_registry/hidden_dim_h32_vs_h64_k8_reporting_plan.md
?? reports/experiment_registry/incomplete_or_duplicate_experiments.csv
?? reports/experiment_registry/knn_by_hidden_dim_factorial_48row_matrix.csv
?? reports/experiment_registry/knn_by_hidden_dim_factorial_audit.md
?? reports/experiment_registry/knn_by_hidden_dim_factorial_missing_runs.csv
?? reports/experiment_registry/knn_by_hidden_dim_factorial_protocol_decision.md
?? reports/experiment_registry/master_experiment_registry.csv
?? reports/experiment_registry/master_experiment_registry.md
?? reports/experiment_registry/missing_run_decision_matrix.csv
?? reports/experiment_registry/normalization_ablation_candidate_matrix.csv
?? reports/experiment_registry/normalization_ablation_control_audit.md
?? reports/experiment_registry/normalization_ablation_family_summary.csv
?? reports/experiment_registry/normalization_ablation_missing_run_decisions.csv
?? reports/experiment_registry/normalization_ablation_reporting_plan.md
?? reports/experiment_registry/professor_request_experiment_inventory.md
?? reports/experiment_registry/professor_table_blueprint.md
?? reports/experiment_registry/registry_build_metadata.json
?? reports/experiment_registry/registry_validation_report.md
?? reports/experiment_registry/source_code_provenance_audit.md
?? reports/experiment_registry/thesis_completion_blueprint.md
?? reports/experiment_registry/thesis_seed_completion_plan.md
```
