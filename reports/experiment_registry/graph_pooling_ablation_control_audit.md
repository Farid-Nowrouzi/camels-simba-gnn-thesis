# Controlled graph-pooling ablation audit

## 1. Executive summary

Outcome **A** applies: a complete, controlled, zero-training EvolveGCN-H
mean-versus-mean_max family already exists.

The recommended family contains ten artifact-complete U750 Top1000 runs:
`mean` and `mean_max` graph pooling at seeds 42, 123, 777, 999, and 2025.
Configs, dataset sidecars, ordered splits, artifacts, predictions, and saved
metrics verify that graph pooling and its mathematically required downstream
input dimension are the only intended differences.

Mean_max did not improve this protocol. Its test MAE was higher in all five
matched seeds. The mean paired difference
`MAE(mean_max) - MAE(mean)` was +0.025130 ± 0.011204. Mean_max also had lower
R² in all seeds and did not consistently improve prediction dispersion.
Therefore simple mean_max concatenation does not support the hypothesis that
mean pooling is the principal bottleneck under this protocol.

No graph rebuilding, replacement run, or new training is required. The
highest-value next action is a future lightweight analysis package named
`controlled_evolvegcn_graph_pooling_ablation_750u_top1000`.

## 2. Search and candidate definition

All 176 parseable experiment configs were inspected. Their raw
`graph_pooling` census was:

- 156 explicit `mean`;
- 14 historical configs without the field, which resolve to default `mean`;
- 6 explicit `mean_max`;
- 0 `sum`;
- 0 `max`;
- 0 attention or hierarchical pooling.

The candidate matrix retains all six non-mean runs and every experiment using
either dataset on which a non-mean method was trained. This yields 38
scientifically relevant candidates while avoiding the false implication that
every unrelated mean-only scaling or kNN run is a pooling ablation.

Candidate classification counts are:

| Compatibility class | Rows |
|---|---:|
| `canonical_graph_pooling_compatible` | 10 |
| `compatible_anchor_only` | 11 |
| `duplicate_reproduction` | 1 |
| `head_incompatible` | 6 |
| `temporal_pooling_incompatible` | 5 |
| `summary_hybrid_incompatible` | 2 |
| `architecture_incompatible` | 2 |
| `normalization_incompatible` | 1 |

## 3. Pooling methods implemented and trained

EvolveGCN-H implements `mean`, `sum`, and `mean_max`. Static GCN/GraphSAGE
implements `mean`, `max`, and `mean_max`.

Only `mean` and EvolveGCN-H `mean_max` were actually trained. The six
mean_max runs comprise five U750 Top1000 none-normalized linear-head runs and
one U500 Top500 minmax MLP-head pilot. There are no trained sum, max-only,
attention, TopK, SAGPool, or Set2Set runs.

Attention pooling is neither implemented nor trained. It remains possible
future work, not missing evidence required by this audit.

## 4. Exact implementation and tensor dimensions

For EvolveGCN-H, node embeddings enter graph pooling as `[B, T, N, H]`.
Masked mean and sum reduce `N` to produce `[B, T, H]`. Mean_max concatenates
the masked mean and masked maximum to produce `[B, T, 2H]`. Pooling is
independent for every graph and snapshot. Temporal mean or last then reduces
`T` without changing the embedding width.

For Static GCN, node embeddings enter as `[B, N, H]`. Mean and max produce
`[B, H]`; mean_max produces `[B, 2H]`. Static pooling occurs once on the
final-snapshot graph.

The regression input dimension changes correctly. With `H=32`, the U750
linear mean_max head has 32 additional weights relative to mean. The U500 MLP
mean_max pilot has 1,024 additional first-layer weights. These are necessary
consequences of concatenation, not different head designs.

Mean, sum, max, and mean_max are permutation invariant. Mask assignments are
batch- and snapshot-correct. All candidate datasets have fixed padded Top-N
shapes and validation rejects snapshots with zero real nodes. The complete
source-level findings are in `graph_pooling_implementation_audit.md`.

No current bug invalidates the controlled family. A latent edge case remains:
if an all-masked graph bypassed preprocessing validation, maximum pooling would
return the smallest representable finite value. Maximum pooling may also be
scientifically sensitive to outliers.

## 5. Defaults and aliases

The model and CLI default is `mean`. Historical configs without the field are
backward compatible through that default.

Accepted values are exact. Directory text such as `meanmax` is not a config
alias; authoritative configs use `mean_max`. `mean+max`, `global_mean_pool`,
`global_max_pool`, `add`, and attention-related names are not accepted
aliases.

## 6. Artifact completeness and metric verification

All 38 candidate rows contain and parse:

- `config.json`;
- `metrics.json`;
- `train_log.csv`;
- `predictions/test_predictions.csv`;
- `checkpoints/best_model.pt`.

Checkpoint paths were checked only for existence and were never opened.

Every prediction file has the expected number of finite rows, unique universe
IDs, no missing or duplicated IDs, and exact ordered agreement with the
declared test split. All 38 saved primary metric sets agree with independent
recomputation at absolute tolerance `1e-6`; the maximum observed difference
was `8.326672684688674e-17`.

The audit recomputed MAE, RMSE, MSE, R², Pearson when defined, target and
prediction means and sample SDs, prediction-SD ratio, exact and approximate
repeat fractions at tolerance `1e-12`, prediction range, and residual mean and
sample SD.

Twelve negative-R² rows and seven rows with repeated predictions were retained.
Two alternative U500 runs have undefined Pearson because prediction variance
is exactly zero; they remain explicitly undefined rather than being replaced
with zero.

## 7. Dataset and model protocols found

| Protocol | Input | Universes | Top-N | Normalization | Snapshots | k | Models represented |
|---|---|---:|---:|---|---:|---:|---|
| U500 Top500 | temporal dataset; Evolve uses all snapshots, Static selects final | 500 | 500 | minmax per universe/snapshot/feature | 5 | 8 | EvolveGCN-H, Static GCN, Static GraphSAGE pilot |
| U750 Top1000 | temporal sequence | 750 | 1000 | none after log10(Mvir) | 5 | 8 | EvolveGCN-H |

Both metadata sidecars specify periodic kNN, box size 25, the seven canonical
features, raw-Mvir Top-N selection, and raw physical XYZ for topology. Exact
snapshot identities are not persisted in the lightweight sidecars; controlled
pairs nevertheless use the identical dataset path and therefore the same
stored graph sequences.

Candidate hidden dimension is 32. Evolve uses two layers and batch size 4.
Static uses three layers and batch size 8. Evolve temporal pooling values are
mean and last; Static has no temporal pooling. Head types found are Evolve MLP,
Evolve linear, and the Static MLP readout.

## 8. Split and fairness audit

The canonical U750 family uses 450 training, 99 validation, and 201 test
universes. Each split is disjoint and together covers `LH_0` through `LH_749`
exactly once. For each seed, mean and mean_max have identical ordered lists:

| Seed | Ordered split signature |
|---:|---|
| 42 | `e27e32ef2d9482fbe0719a56555f66075f7a23e178c15848a0227660ea0df7d9` |
| 123 | `bd9367b034a0ac7ebc07fe79c19a62d51f17402cb7e5dba98d1748e5168f72f1` |
| 777 | `6e71569b7785aa0483b56e359c68bac69e9c997617880855b75fd2bedcd84dae` |
| 999 | `27af8142ba8da15f65baf969086cca0e3b72e105c29a41ac778d6c702a8cb678` |
| 2025 | `3f80461e5ec26ee906f6c1781f34d40877521ac854d93f1fb09783b18b8d24a7` |

The U500 seed-42 pilot is also exactly paired at 350/75/75, but only one
mean_max seed exists. No split-incompatible run enters the canonical family.

## 9. Controlled paired results

The convention is:

`difference = metric(mean_max) - metric(mean)`

Positive MAE/RMSE favours mean. Negative R² favours mean.

| Seed | ΔMAE | ΔRMSE | ΔR² | Δprediction-SD ratio | Δexact-repeat fraction |
|---:|---:|---:|---:|---:|---:|
| 42 | +0.041155 | +0.050563 | -0.601621 | -0.009396 | 0 |
| 123 | +0.023050 | +0.049078 | -0.689408 | -0.027855 | 0 |
| 777 | +0.024843 | +0.037934 | -0.574945 | +0.495616 | 0 |
| 999 | +0.026894 | +0.029839 | -0.357005 | -0.237654 | 0 |
| 2025 | +0.009709 | +0.007757 | -0.087763 | -0.215602 | 0 |

Summary:

- mean ΔMAE: +0.025130; SD: 0.011204; sign count 0/0/5
  negative/zero/positive;
- mean ΔRMSE: +0.035034; SD: 0.017449; all five positive;
- mean ΔR²: -0.462148; SD: 0.242406; all five negative;
- mean Δprediction-SD ratio: +0.001022; SD: 0.295575; four negative and one
  positive;
- neither method has exact repeated predictions in the canonical family.

Aggregate mean MAE was 0.055843 ± 0.008748 for mean and
0.080973 ± 0.010152 for mean_max. Mean R² was 0.627697 versus 0.165548.
Mean prediction-SD ratios were nearly equal at 0.862457 and 0.863478, but the
seed-level effect was highly inconsistent.

The error penalty is larger than the between-seed MAE SD in either cell.
Mean_max therefore consistently worsens prediction error and does not provide
a stable dispersion benefit.

The U500 single-seed pilot points in the same direction but is not sufficient
independent evidence: ΔMAE is +0.003083, ΔR² is -0.055641, and the
prediction-SD ratio falls from 0.362423 to 0.222955.

## 10. Duplicate and alternative protocols

One exact duplicate reproduction was found:

- `evolvegcn_h_750u_top1000_h32_seed42_none`;
- `evolvegcn_h_750u_top1000_h32_seed42_none_mlp_relu_repeat`.

Their prediction files have identical SHA-256
`e38c1d1e21829d266cb44bba53a06e558eae456594e4ec43d62a21374ebd8a34`.
The duplicate is retained in the matrix but is not counted as independent
evidence.

Alternatives retained with exclusions include:

- five U750 MLP-head mean anchors, incompatible with the linear-head family;
- five U750 temporal-last linear runs;
- U500 hybrid summary-feature runs;
- U500 linear-head and target-normalized pilots;
- a U500 LeakyReLU run;
- five Static mean-only anchors;
- one Static GraphSAGE mean pilot.

These concepts are not silently combined with graph pooling.

## 11. Possible controlled families

Six plausible families or anchor sets are recorded in
`graph_pooling_ablation_family_summary.csv`.

The strongest is the complete U750 temporal-mean linear-head family. A U500
MLP family has five mean anchors but only one mean_max run. U750 MLP and
temporal-last families have only mean anchors. Static GCN has five mean anchors
but no max or mean_max runs. Static GraphSAGE has only a seed-42 mean pilot.

## 12. Canonical decision and run requirements

Recommended canonical design:

- EvolveGCN-H;
- 750 universes, Top1000;
- five snapshots;
- no node-feature scaling after log10(Mvir);
- periodic kNN, `k=8`, box size 25;
- hidden dimension 32, two layers;
- batch size 4;
- ReLU;
- temporal mean;
- linear head;
- graph pooling `mean` versus `mean_max`;
- seeds 42, 123, 777, 999, 2025;
- exact paired splits.

Run accounting for the recommended family:

- total required rows: 10;
- reusable existing rows: 10;
- missing required rows: 0;
- partial rows: 0;
- replacement rows: 0;
- duplicates counted as evidence: 0;
- graph rebuilding required: no;
- new training required: no.

The missing-run decision matrix lists 26 possible Static, head-specific,
temporal-last, U500, and GraphSAGE extensions as
`optional_extension_not_required`. None is recommended for the current thesis.

## 13. Scientific interpretation and caveats

Under this U750 Top1000 EvolveGCN-H linear-head protocol, concatenating maximum
node embeddings consistently worsens error. This suggests that simple maxima
add noise or unstable extreme responses rather than recovering useful
information lost by a mean.

The result does not prove that:

- all graph readouts are equivalent;
- learned attention would fail;
- maximum halo properties are scientifically irrelevant;
- mean pooling is always optimal;
- the same result holds for Static GCN, normalized inputs, other Top-N values,
  or other heads.

No trained attention family exists. Attention should be described as future
work, not fabricated evidence or a required completion run.

## 14. Proposed future analysis and next action

Proposed identifier:

`controlled_evolvegcn_graph_pooling_ablation_750u_top1000`

Highest-value next action: build the final tables, diagnostics, and figures
from the ten verified runs using median-MAE representative selection. Do not
train additional pooling methods before reporting the complete controlled
result.

## 15. Exact Git status

Six files were created by this audit:

```text
?? reports/experiment_registry/graph_pooling_ablation_candidate_matrix.csv
?? reports/experiment_registry/graph_pooling_ablation_control_audit.md
?? reports/experiment_registry/graph_pooling_ablation_family_summary.csv
?? reports/experiment_registry/graph_pooling_ablation_missing_run_decisions.csv
?? reports/experiment_registry/graph_pooling_ablation_reporting_plan.md
?? reports/experiment_registry/graph_pooling_implementation_audit.md
```

All other untracked paths below pre-dated this audit:

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
?? reports/experiment_registry/professor_request_experiment_inventory.md
?? reports/experiment_registry/professor_table_blueprint.md
?? reports/experiment_registry/registry_build_metadata.json
?? reports/experiment_registry/registry_validation_report.md
?? reports/experiment_registry/source_code_provenance_audit.md
?? reports/experiment_registry/thesis_completion_blueprint.md
?? reports/experiment_registry/thesis_seed_completion_plan.md
```

There were no pre-existing tracked modifications at audit start, and this
audit did not modify tracked source or experiment artifacts. Nothing was
staged, committed, or pushed.
