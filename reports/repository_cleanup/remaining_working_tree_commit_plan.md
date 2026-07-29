# Remaining Working-Tree Commit Plan

Inspection snapshot: `thesis-controlled-scaling` at
`ab5552aa9a21c8995acde0f004305ddd74602f2a`, equal to its upstream. The initial
porcelain snapshot had 4 modified and 54 untracked entries. This document is a
plan only: nothing was staged, restored, deleted, trained, or committed.

## Static GCN focused audit

The exact HEAD diff is 125 insertions/7 deletions in
`src/models/static_gcn.py` and 193 insertions/9 deletions in
`src/training/train_static_gcn.py`. For reproducible identification, the
individual `git diff -- <file>` SHA-256 values are respectively
`2ccd264711b9f68d5588fbf4ef537572fb709fdd2b45fa7ad4a04556de38eb29` and
`a262637dbb5332af766322528769125d1575c303590cc2b15e2a519e287ef9df`.

The model diff adds `DenseGraphSAGELayer`, selects `gcn` or `graphsage` through
`conv_type`, computes a masked neighbor mean for dense adjacency, concatenates
self and neighbor features, and retains the old dense-GCN path as the default.
The trainer diff adds `dataset_format=temporal_final_snapshot`, converts the
last temporal snapshot for static training, can read and validate a
`split_config_path` to reuse train/validation/test simulation IDs, exposes
`conv_type`, and records dataset format, convolution, and split provenance.

Defaults preserve the former CLI behavior (`static`, `gcn`, generated splits).
Backward compatibility is not yet unconditional: new parameters were inserted
before `seed` in the trainer's Python function and before `add_self_loops` in
the regressor constructor. Positional callers that supplied later arguments can
silently bind to the wrong parameter. The CLI's old named options remain
compatible.

Sixteen existing configurations use the new behavior: the fourteen
`static_gcn_500u_top500_final_snapshot_h32_seed{42,123,777,999,2025}_{none,minmax,zscore}`
configurations that exist (all combinations except seed42/minmax),
`static_gcn_top500_final_snapshot_same_split_h32_seed42`, and
`static_graphsage_500u_top500_final_snapshot_h32_seed42`.

Inspection tests passed:

- in-memory Python compilation of both changed files;
- imports in `envs/camels-gnn`;
- `python -m src.models.static_gcn` GCN smoke test, output shape `(4, 1)` and
  4,161 parameters;
- `python -m src.training.train_static_gcn --help`;
- no repository test suite currently covers this code.

Before staging, add or run focused tests for both GCN and GraphSAGE forward
paths, zero-neighbor masking, temporal-final-snapshot conversion, split-ID
validation/reordering, config provenance, and old named-argument behavior.
Review whether public parameters should be appended or made keyword-only.
Also profile the temporal conversion: loading a 15 GB temporal dataset and
building a second static structure can exceed RAM. These two source files
should form one separate commit only after that review.

## Evaluation classification and groups

All 20 untracked evaluation scripts compile in memory. In the project
environment every module responds to `python -m src.evaluation.<module>
--help`. Running files directly is not the supported mode because package
imports expect the repository root. Torch and NumPy are common dependencies;
scikit-learn is additionally used by distribution-shift,
graph-versus-summary, summary-versus-embedding, and summary-feature baseline
code. `diagnose_predictions.py` uses pandas. The runners and diagnostics that
load models share the EvolveGCN-H training/model code.

### Baseline runners

- `src/evaluation/run_summary_feature_baseline.py`
- `src/evaluation/run_summary_mlp_baseline.py`

Both have CLIs and write configs, metrics, and predictions into a newly named
experiment directory. Their current `exist_ok` behavior can reuse/overwrite a
same-name directory, so test refusal or completeness checks before commit.
They are reusable baseline runners, not diagnostics.

### Representation and split diagnostics

- `src/evaluation/diagnose_evolvegcn_h_representations.py` — shared extraction
  core plus CLI.
- `src/evaluation/diagnose_embedding_distribution_shift.py`
- `src/evaluation/diagnose_embedding_feature_stability.py`
- `src/evaluation/diagnose_embedding_neighborhood_consistency.py`
- `src/evaluation/diagnose_embedding_probe_splits.py`
- `src/evaluation/diagnose_embedding_target_relationship.py`
- `src/evaluation/diagnose_split_target_distribution.py`

These are reusable diagnostics. They emit JSON; model-based tools read a
dataset/checkpoint and default to an `experiment_dir/diagnostics/` destination,
so they modify the selected experiment unless `--output_path` is redirected.
Test the shared extractor once with a tiny fixture, then test each statistic
and explicit-output behavior.

### EvolveGCN-H collapse investigation

- `src/evaluation/diagnose_evolvegcn_h_feature_variance.py`
- `src/evaluation/diagnose_evolvegcn_h_head_analysis.py`
- `src/evaluation/diagnose_evolvegcn_h_layer1_activations.py`
- `src/evaluation/diagnose_evolvegcn_h_layer_variance.py`
- `src/evaluation/diagnose_evolvegcn_h_regressor_head.py`
- `src/evaluation/diagnose_evolvegcn_h_variance_flow.py`
- `src/evaluation/diagnose_graph_vs_summary_baseline.py`
- `src/evaluation/diagnose_head_vs_optimal_linear_solution.py`
- `src/evaluation/diagnose_summary_vs_embedding_combined.py`

These are one-off exploratory diagnostics implemented as reusable CLIs. They
read checkpoints/datasets, produce JSON, and generally default to changing the
experiment's diagnostics directory. Commit them together only if the
investigation is thesis evidence; otherwise archive them outside Git. Tests
should use a tiny synthetic fixture and verify that explicit output paths do
not touch an experiment.

### Lightweight prediction diagnostics

`src/evaluation/diagnose_prediction_collapse_summary.py` is the canonical
reusable, explicit-output prediction-CSV summarizer.
`src/evaluation/diagnose_predictions.py` is an older pandas implementation
superseded by it; exclude the latter as category G unless a missing statistic
is demonstrated.

The already tracked `run_mean_baseline.py` and `summarize_experiments.py` are
reusable baseline/summary tools, and `__init__.py` is package infrastructure;
they have no working-tree change.

## Ordered commits

### 1. Record the repository-cleanup decision

Files:

- `reports/repository_cleanup/remaining_working_tree_inventory.csv`
- `reports/repository_cleanup/remaining_working_tree_commit_plan.md`
- `reports/repository_cleanup/proposed_gitignore_policy.md`
- `reports/repository_cleanup/notebook_and_report_audit.md`

Explicitly exclude all other current changes. Validate CSV parsing, Markdown
links/paths, and a fresh porcelain count. Suggested message:
`Document remaining working-tree classification`. Do this before the five
canonical runs.

### 2. Add reviewed Static GCN temporal-snapshot and GraphSAGE support

Files:

- `src/models/static_gcn.py`
- `src/training/train_static_gcn.py`

Exclude experiments, notebooks, and diagnostics. Run the focused tests listed
above and resolve positional API/RAM concerns first. Suggested message:
`Add controlled static graph convolution variants`. Do this before the five
canonical runs so source provenance is clean, although those EvolveGCN-H runs
do not directly depend on it.

### 3. Add reusable summary-feature baselines

Files:

- `src/evaluation/run_summary_feature_baseline.py`
- `src/evaluation/run_summary_mlp_baseline.py`

Exclude diagnostic scripts and experiment outputs. Test tiny inputs, fixed
splits, deterministic seeds, schema compatibility, and same-name overwrite
protection. Suggested message: `Add reusable summary baseline runners`. This
can occur before the canonical runs.

### 4. Add representation and split diagnostics

Files are exactly the seven files in the representation/split group above.
Exclude collapse probes and generated diagnostic JSON. Test tiny-fixture model
loading, all split modes, numerical edge cases, CLI help, and output
redirection. Suggested message: `Add EvolveGCN representation diagnostics`.
This can occur before the canonical runs.

### 5. Preserve the collapse investigation tools

Files are exactly the nine files in the collapse-investigation group above.
Exclude `diagnose_predictions.py`, checkpoints, and generated diagnostics.
Test tiny-fixture execution and no-write behavior when an explicit external
output is selected. Suggested message:
`Add EvolveGCN prediction-collapse diagnostics`. Do this only if these probes
are cited by the thesis; timing is independent of the canonical runs.

### 6. Add the canonical lightweight prediction summary

File: `src/evaluation/diagnose_prediction_collapse_summary.py`. Explicitly
exclude `src/evaluation/diagnose_predictions.py`. Test empty, constant, NaN,
and ordinary prediction CSVs plus JSON schema. Suggested message:
`Add prediction collapse summary diagnostic`. This can occur before the runs.

### 7. Preserve historical kNN ablation runners

Files:

- `scripts/run_knn_ablation_500u_training.sh`
- `scripts/run_knn_ablation_k8_h64.sh`

Exclude logs and experiment directories. `bash -n` passes; additionally review
`set -o pipefail`, quoting in the first script, completeness checks stronger
than `metrics.json`, and a dry-run mode. Suggested message:
`Add reproducible 500U kNN ablation runners`. These document completed
historical runs and need not block the canonical runs.

### 8. Add stable experimental documentation

Files:

- `docs/experiment_naming_convention.md`
- `reports/experiment_registry/professor_table_blueprint.md`
- `reports/experiment_registry/baseline_validation_audit.md` only after its
  stated confirmation

Exclude live/stale registry products. Check terminology against the committed
pipeline and registry builder. Suggested message:
`Document experiment naming and reporting rules`. Do this before the runs.

### 9. Add validated diagnostic and ablation notebooks

First commit `notebooks/visualization/05_evolvegcn_h_diagnostic_investigation.ipynb`
with the diagnostic source it documents. In a later notebook-only commit,
include:

- `notebooks/validation/01_raw_dataset_integrity_validation.ipynb`
- `notebooks/visualization/06_normalization_ablation_top500.ipynb`
- `notebooks/visualization/08_evolvegcn_750u_top1000_head_ablation_results.ipynb`
- `notebooks/visualization/09_evolvegcn_750u_top1000_mean_vs_meanmax_pooling_results.ipynb`

Exclude 03, 04, both 07 copies, and presentation exports. Before staging,
remove or sanitize server paths and sensitive output, clear excessive output
where it is not scientific evidence, restart/run against raw registry evidence,
and verify reported values independently. Suggested messages:
`Add EvolveGCN diagnostic investigation notebook` and
`Add validated controlled-ablation notebooks`. These need not block the five
runs.

### 10. Adopt the artifact policy

File: `.gitignore`. Do not stage `experiments/` in this commit. Review the exact
rules in `proposed_gitignore_policy.md`, confirm no tracked file becomes
accidentally ignored, and verify representative checkpoint/data/cache paths
with `git check-ignore -v`. Suggested message:
`Clarify experiment artifact ignore policy`. Do this before staging evidence.

### 11. Track selected lightweight experiment evidence

No commit should be staged yet: an exact allowlist has not been approved.
After approval, generate and review a manifest selecting completed-run
`config.json`, `metrics.json`, `test_predictions.csv`, and compact diagnostics;
exclude every checkpoint, partial run, large output, and log. Suggested message:
`Record lightweight experiment evidence`. Prefer after the five runs so the
selection is stable.

### 12. Refresh the professor-ready registry

After all five canonical runs finish, regenerate and validate this minimum set:

- `reports/experiment_registry/master_experiment_registry.csv`
- `reports/experiment_registry/master_experiment_registry.md`
- `reports/experiment_registry/registry_build_metadata.json`
- `reports/experiment_registry/registry_validation_report.md`
- `reports/experiment_registry/controlled_experiment_matrix.csv`
- `reports/experiment_registry/final_table_readiness_status.md`
- `reports/experiment_registry/professor_request_experiment_inventory.md`
- `reports/experiment_registry/professor_table_blueprint.md`

Also include `canonical_scaling_live_status.csv` only if it is converted from a
transient live snapshot into the final completed-run record. Exclude temporary
plans, stale snapshots, and redundant generated summaries. Run the registry
builder/validator, require current branch/commit and clean provenance, verify
15/15 canonical completion, recompute metrics from predictions, and compare
table aggregates. Suggested message:
`Refresh professor-ready experiment registry`. This is strictly after the five
runs.

## Files that must not be committed

Never commit `data/`, `envs/`, any checkpoint or `.pt/.pth/.ckpt`, `logs/`,
`outputs/`, Python caches, `.ipynb_checkpoints/`,
`graph_sanity_outputs.tar.gz`, stale presentation exports, temporary live
snapshots, or stale registry builds. Restore the two kernelspec-only notebook
changes after approval. Exclude the root notebook-07 duplicate and
`diagnose_predictions.py` unless review overturns the duplicate finding.

The highest-value immediate source commit is number 2, but only after its
focused tests and compatibility review. Until then, number 1 is the only
immediately stageable, self-contained commit.
