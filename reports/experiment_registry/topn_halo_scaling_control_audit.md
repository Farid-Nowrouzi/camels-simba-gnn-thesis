# Controlled Top-N halo-count scaling audit

## 1. Executive summary

A complete zero-training, model-stratified U500 Top100/Top200/Top500 design
exists. EvolveGCN-H and Static GCN each contribute nine artifact-complete rows:
three Top-N levels at seeds 42, 123, and 2025. Exact ordered splits match
within every model and seed. The recommended future analysis therefore
contains 18 rows and does not require graph rebuilding or training.

For EvolveGCN-H, MAE improves monotonically in all three seeds from Top100 to
Top200 and again to Top500. Mean MAE changes from
0.096962 to
0.092174 and
0.086166. For Static GCN, mean MAE is
nearly unchanged (0.096262,
0.096985,
0.096039) and paired directions are mixed.

U750 Top1000 is descriptive context only: universe count, normalization, and
parts of the readout protocol differ. It is not a fourth point in the U500
Top-N causal trend.

## 2. Universe-count versus Top-N

Universe-count scaling changes the number of independent simulated universes.
Top-N scaling fixes the universe population and changes the maximum number of
raw-Mvir-ranked halos retained per universe and snapshot. Connectivity k,
hidden width, and model capacity are separate factors.

## 3. Dataset inventory

Twenty-eight `.pt` paths have parseable JSON sidecars and were inspected
without loading tensor contents.

| Top-N | Dataset sidecars | Universe counts |
|---:|---:|---|
| 100 | 22 | 1|20|50|100|200|500 |
| 200 | 2 | 500 |
| 500 | 3 | 500 |
| 1000 | 1 | 750 |

No Top50, Top300, or Top750 dataset exists. Top1000 exists only for U750.
The dataset matrix records every path, metadata SHA-256, and protocol.

## 4. Experiment inventory

All 176 parseable experiment configs were inspected. Fourteen non-GNN mean
baselines/debug baselines do not encode a trained graph Top-N intervention.
The candidate matrix contains all 162 trained GNN runs:
91 EvolveGCN-H and 71 Static GCN
(including the GraphSAGE convolution pilot under the Static trainer).

Top-N experiment counts are {100: 98, 200: 6, 500: 37, 1000: 21}. Universe counts
are [20, 50, 100, 200, 500, 750]; k values [4, 6, 8, 12]; hidden dimensions [32, 64];
layer counts [2, 3]; batch sizes [4, 8]; seeds [42, 123, 777, 999, 2025].

| Compatibility class | Rows |
|---|---:|
| `alternative_protocol` | 36 |
| `architecture_incompatible` | 12 |
| `canonical_topn_compatible` | 18 |
| `compatible_anchor_only` | 6 |
| `duplicate_reproduction` | 14 |
| `graph_pooling_incompatible` | 1 |
| `head_incompatible` | 2 |
| `hybrid_incompatible` | 2 |
| `normalization_incompatible` | 20 |
| `universe_count_incompatible` | 51 |

## 5. Top-N selection implementation

`clean_halo_dataframe` removes invalid required values and non-positive raw
Mvir. `select_top_halos` sorts `col_10` (raw Mvir) descending and applies
`head(num_nodes)` independently inside each processed snapshot. Node features
are then constructed as log10(Mvir), XYZ, and velocities. Thus selection is
per universe and per snapshot, with the same configured maximum N across the
sequence.

Selection precedes log transformation and normalization. Raw physical XYZ are
copied before feature normalization and used for periodic minimum-image kNN.
Graph construction is repeated for every Top-N dataset.

## 6. Nesting, padding, and masking

For identical cleaned input and software behavior, sorting the same raw-Mvir
table before `head(N)` makes smaller selections prefixes of larger selections.
However, sidecars do not store selected halo IDs or rank hashes, so nesting is
not byte-proven. Pandas sorting has no explicit stable algorithm or secondary
tie key; equal-mass boundary ties are therefore a reproducibility caveat.

When fewer than N valid halos exist, features and raw positions are zero-padded
after normalization. Masks use one for real nodes and zero for padding; padded
nodes remain disconnected. Temporal validation rejects zero-real-node
snapshots. Dataset-level sidecars do not record raw or selected halo counts per
snapshot, so actual padding prevalence cannot be audited without forbidden
graph deserialization.

Node nesting does not imply topology nesting. Adding halos can replace kNN
neighbors among previously retained nodes, and every adjacency is rebuilt.

## 7. Normalization interaction

All canonical U500 datasets use minmax independently per universe, snapshot,
and feature after Top-N selection. Therefore changing Top-N changes both node
count and the sample-local min/max statistics. This is part of the historical
intervention, not leakage. The available family does not isolate raw node
count from normalization-statistic changes.

The raw/no-normalization U500 evidence exists only at Top500. U750 Top1000 is
also raw/no normalization, but its universe population differs.

## 8. Artifact, split, and metric verification

All 162 candidate GNN runs contain config, metrics, CSV
training log, test predictions, and checkpoint path. Checkpoints were checked
only for existence and size.

Every prediction file has finite values, unique nonempty universe IDs, the
declared test-row count, and exact ordered agreement with its config split.
Every config split is disjoint and covers its declared LH population exactly.

Saved primary metrics agree with independent recomputation at absolute
tolerance 1e-6; maximum discrepancy is 8.3266726846886741e-17. The audit retained
105 negative-R² rows, 29 rows with exact repeated
predictions, and 5 undefined Pearson values. Undefined Pearson is
never encoded as zero.

## 9. Canonical protocol

Both families use U500 `LH_0..LH_499`, minmax features, periodic kNN k=8,
box size 25, h32, mean graph pooling, raw Omega_m targets, MSE loss, 300
epochs maximum, patience 40, learning rate 0.001, weight decay 1e-5, dropout
0.2, and seeds 42/123/2025.

Evolve uses five temporal snapshots, two EvolveGCN-H layers, temporal mean,
MLP head, and batch size 4. Static uses the a=1.0 final snapshot, three GCN
layers, its established MLP readout, and batch size 8. Results must be
model-stratified; Static-versus-Evolve differences are descriptive.

The Static Top500 input is stored as a temporal dataset but the Static loader
selects only its final snapshot. Source and metadata establish the same a=1.0
final-snapshot intent as the Top100/Top200 static datasets.

## 10. Canonical results

| Model | Top-N | Mean MAE ± SD | Mean RMSE | Mean R² | Mean prediction-SD ratio | Mean repeat fraction |
|---|---:|---:|---:|---:|---:|---:|
| EvolveGCN-H | 100 | 0.096962 ± 0.009492 | 0.113241 | -0.032351 | 0.062896 | 0.000000 |
| EvolveGCN-H | 200 | 0.092174 ± 0.008329 | 0.109696 | 0.025709 | 0.394002 | 0.000000 |
| EvolveGCN-H | 500 | 0.086166 ± 0.004255 | 0.102848 | 0.141301 | 0.494102 | 0.000000 |
| Static GCN | 100 | 0.096262 ± 0.008340 | 0.112127 | -0.013416 | 0.043305 | 0.448889 |
| Static GCN | 200 | 0.096985 ± 0.009836 | 0.112920 | -0.026294 | 0.070753 | 0.466667 |
| Static GCN | 500 | 0.096039 ± 0.008923 | 0.111753 | -0.005794 | 0.063510 | 0.462222 |

Paired convention is larger Top-N minus smaller Top-N:

| Model | Comparison | Mean ΔMAE ± SD | MAE negative/positive | Mean ΔRMSE | Mean ΔR² | Mean ΔSD ratio |
|---|---|---:|---:|---:|---:|---:|
| EvolveGCN-H | 200-100 | -0.004788 ± 0.005264 | 3/0 | -0.003546 | +0.058060 | +0.331106 |
| EvolveGCN-H | 500-200 | -0.006008 ± 0.004694 | 3/0 | -0.006848 | +0.115592 | +0.100101 |
| EvolveGCN-H | 500-100 | -0.010796 ± 0.005376 | 3/0 | -0.010394 | +0.173652 | +0.431206 |
| Static GCN | 200-100 | +0.000722 ± 0.001511 | 1/2 | +0.000794 | -0.012879 | +0.027448 |
| Static GCN | 500-200 | -0.000945 ± 0.001154 | 3/0 | -0.001167 | +0.020500 | -0.007242 |
| Static GCN | 500-100 | -0.000223 ± 0.000733 | 2/1 | -0.000373 | +0.007621 | +0.020206 |

Evolve Top500-minus-Top100 has ΔMAE
-0.010796 ±
0.005376, with all three
seeds favouring Top500. The improvement is comparable to, and slightly larger
than, the Top100 between-seed MAE SD. Static Top500-minus-Top100 is
-0.000223 ±
0.000733, with mixed signs and
negligible magnitude.

## 11. Duplicates and alternatives

There are 7 exact prediction-hash duplicate groups,
representing 7 non-independent
reproduction rows. No selected canonical row is duplicated.

Alternatives retained include U20/U50/U100/U200 universe scaling, h64, k4/k6/
k12, normalization variants, linear and target-normalized heads, summary
hybrids, mean_max graph pooling, temporal-last pooling, GraphSAGE, legacy
Static L2/batch4 anchors, and U750 Top1000.

## 12. Prediction compression

Larger Top-N improves Evolve error, but prediction-SD ratios remain below one
on average and must be reported alongside error. Static dispersion and error
change little. Thus additional halos help Evolve under this protocol but do
not by themselves establish that prediction compression is resolved.

## 13. Computational scaling

For k=8 and N greater than k, neighbor selection grows linearly as `8N` per
snapshot: 800, 1,600, and 4,000 selections at Top100, Top200, and Top500.
Dense adjacency storage and dense message passing scale approximately as
N²: 10,000, 40,000, and 250,000 matrix entries per snapshot. Evolve processes
five snapshots; Static processes one.

Logs do not record epoch duration, total wall time, GPU/CPU peak memory, or
prediction time. The resource CSV reports these explicitly as `not recorded`
and includes checkpoint byte sizes without opening checkpoints. Computational
cost therefore cannot be balanced quantitatively against predictive gain from
the existing lightweight evidence alone.

## 14. Decision matrix

- Canonical reusable rows: 18.
- Required missing rows: 0.
- Partial rows: 0.
- Replacement rows: 0.
- Optional five-seed extension: eight missing Top100/Top200 cells for seeds
  777 and 999 across the two models; not recommended solely for symmetry.
- Graph rebuilding: not required.
- New training: not required.

## 15. Strongest valid scientific question

How does increasing the retained halo population from the 100 most massive
halos to 200 and 500 halos affect Omega_m regression under fixed,
model-stratified U500 EvolveGCN-H and Static GCN protocols?

The comparison must acknowledge that per-snapshot minmax statistics change
with Top-N.

## 16. Recommended future analysis

Use identifier `controlled_static_vs_evolvegcn_topn_halo_scaling_500u`.
Build the future analysis from the 18 verified rows, with within-model paired
comparisons and descriptive cross-model presentation. The highest-value next
action is final analysis generation, not training.

## 17. Scientific caveats

- Three matched seeds support descriptive paired evidence, not fragile
  significance testing.
- Sample-local minmax statistics change with N.
- Selected halo membership is logically expected to nest but is not stored as
  auditable IDs; equal-mass ties lack an explicit stable secondary key.
- Graph topology is rebuilt and is not nested.
- Static and Evolve protocols differ and require stratification.
- Top1000 changes universe count and normalization.
- Resource timing and peak-memory evidence is absent.

## 18. Final Git state

At audit completion, `git diff --check` passes. The only files created by this
audit are the eight requested `topn_halo_*` outputs. All other untracked paths
below were pre-existing and remain untouched. There are no tracked
modifications. Exact `git status --short`:

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
?? reports/experiment_registry/topn_halo_scaling_candidate_matrix.csv
?? reports/experiment_registry/topn_halo_scaling_control_audit.md
?? reports/experiment_registry/topn_halo_scaling_dataset_matrix.csv
?? reports/experiment_registry/topn_halo_scaling_family_summary.csv
?? reports/experiment_registry/topn_halo_scaling_missing_run_decisions.csv
?? reports/experiment_registry/topn_halo_scaling_reporting_plan.md
?? reports/experiment_registry/topn_halo_scaling_resource_audit.csv
?? reports/experiment_registry/topn_halo_selection_implementation_audit.md
```
