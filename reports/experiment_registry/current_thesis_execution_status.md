# Current Thesis Execution Status

Inspection date: 2026-07-29 UTC.

## Executive finding

The controlled EvolveGCN-H scaling family is **10/15 complete**. The canonical
20U seed-42 pilot completed successfully and joins nine compatible historical
`final32` runs. Five replacement folders remain absent:

- 20U seed 2025
- 50U seed 42
- 50U seed 2025
- 100U seed 42
- 100U seed 2025

No training or canonical runner process is active. A tmux session named
`canonical-scaling` exists, but its only pane is an idle `bash`.

The runner passes `bash -n`, retains exactly six sequential `run_one` calls,
and is safe by inspection. Its optional `/usr/bin/time` fallback is present
only as an unstaged working-tree edit: it is not committed or pushed.

## Inspection boundaries

- No `.pt` dataset was loaded.
- No checkpoint was loaded.
- No training or full canonical verification was launched.
- Checkpoints and datasets were checked only for filesystem presence.
- Test R² values below were recomputed directly from finite
  `true_omega_m`/`pred_omega_m` CSV columns because `metrics.json` does not save
  R².

## Inventory system

### Script purposes

- `scripts/build_experiment_registry.py`: scans experiment folders, lightweight
  metadata, configs, metrics, predictions, logs, reports/notebooks/outputs, and
  Git metadata; recomputes prediction metrics; groups experiment families;
  identifies missing/duplicate/stale evidence; and writes a registry package.
  It does not deserialize datasets or checkpoints.
- `scripts/verify_canonical_evolvegcn_scaling.py`: read-only final gate for the
  exact 15-row canonical EvolveGCN-H family. It checks dataset and metadata
  presence/settings, required artifacts, config protocol, seed/dataset paths,
  expected split counts and non-overlap, prediction columns/metrics, complete
  universe-seed coverage, and seed-level/aggregate results.
- `scripts/run_canonical_evolvegcn_scaling_missing.sh`: sequentially launches
  the six originally planned low-U replacements, skips a folder only when all
  five required artifacts exist, and stops safely if a planned folder is
  partial.

### Registry-generated reports

`registry_build_metadata.json` declares these generated report files:

- `master_experiment_registry.csv`
- `master_experiment_registry.md`
- `experiment_family_summary.csv`
- `incomplete_or_duplicate_experiments.csv`
- `analysis_artifact_index.csv`
- `registry_validation_report.md`
- `professor_request_experiment_inventory.md`
- `registry_build_metadata.json`

It also declares `docs/experiment_naming_convention.md`, outside the report
directory.

### Separately produced inventory/control reports

The following are not in the registry generator's output allow-list and were
therefore produced separately (their exact manual/tool provenance is not saved):

- `baseline_validation_audit.md`
- `canonical_evolvegcn_scaling_execution_plan.md`
- `canonical_evolvegcn_scaling_run_matrix.csv`
- `canonical_scaling_family.csv`
- `controlled_experiment_matrix.csv`
- `family_compatibility_audit.md`
- `missing_run_decision_matrix.csv`
- `professor_table_blueprint.md`
- `source_code_provenance_audit.md`
- `thesis_completion_blueprint.md`
- `thesis_seed_completion_plan.md`

### Authoritative roles

- Master experiment inventory:
  `master_experiment_registry.csv`; `master_experiment_registry.md` is its
  human-readable rendering.
- Controlled-family compatibility:
  `canonical_scaling_family.csv`, `controlled_experiment_matrix.csv`, and
  `family_compatibility_audit.md`.
- Final professor-table plan:
  `professor_table_blueprint.md`, supported by
  `thesis_completion_blueprint.md`.
- Canonical execution manifest:
  `canonical_evolvegcn_scaling_run_matrix.csv` and
  `canonical_evolvegcn_scaling_execution_plan.md`.

### Known stale evidence

- The generated registry is a 2026-07-11 snapshot of branch `main` at
  `f260a56`; it predates commit `f2ccae3` and the completed 20U seed-42 pilot.
  Its generated metadata reports 160 scanned folders and must not be treated as
  the live canonical status.
- The canonical execution plan/run matrix and `canonical_scaling_family.csv`
  still describe 20U seed-42 as planned/missing; raw artifacts now prove it is
  complete.
- Registry validation records 50 metric mismatches in two known stale summary
  CSVs:
  `outputs/evolvegcn_500top500_vs_750top1000_5seed_table.csv` and
  `outputs/evolvegcn_500u_vs_750u_results/seed_level_results_clean.csv`.
- Raw configs, metrics, predictions, and artifact presence take precedence over
  all of these summaries.

## Git and source state

- Branch: `thesis-controlled-scaling`
- HEAD: `f2ccae39303a3982b35290260184b6a0cbceec5d`
- Upstream: `origin/thesis-controlled-scaling`
- Ahead/behind: `0/0`
- Configured remote:
  `https://github.com/Farid-Nowrouzi/camels-simba-gnn-thesis.git`

Latest five commits:

| Commit | Date (UTC) | Subject |
|---|---|---|
| `f2ccae39303a3982b35290260184b6a0cbceec5d` | 2026-07-23 09:37:31 | Add reproducible canonical EvolveGCN-H scaling workflow |
| `f260a5664049c81980ef7efa84ebfacad286152c` | 2026-05-19 00:36:15 | Update CAMELS-SIMBA notebooks and downloader script |
| `7d6bf12812d00e9cb4910013c068fbbb8405a7b4` | 2026-05-14 02:29:11 | Add CAMELS-SIMBA downloader and graph scaling notebook |
| `8e99d6b84456049830d64efea2ec8da37b2a09d2` | 2026-05-13 16:47:24 | Improve EvolveGCN-H temporal training pipeline |
| `49ce7c63404aaa9cce1b62a7db3bd2fab9eb2d3d` | 2026-05-13 16:18:42 | Clean EvolveGCN-H model implementation |

Commit `f2ccae3` exists locally and is the current HEAD. A live read-only
`git ls-remote` confirms that
`refs/heads/thesis-controlled-scaling` on `origin` points to the same full
commit.

### Runner fallback state

- Edited: **yes**
- Staged: **no**
- Committed: **no**
- Pushed: **no**
- Commit containing the fallback: **none**

Commit `f2ccae3` added the runner and verifier, but its runner still hard-fails
when `/usr/bin/time` is unavailable. The working-tree diff introduces
`TIME_PREFIX=(/usr/bin/time -v)` when available and an empty array otherwise,
then uses that array for logging and execution.

Relevant-file uncommitted audit:

| File | Worktree state |
|---|---|
| `src/models/evolvegcn_h.py` | clean |
| `src/training/train_evolvegcn_h.py` | clean |
| `scripts/run_canonical_evolvegcn_scaling_missing.sh` | modified, unstaged |
| `scripts/verify_canonical_evolvegcn_scaling.py` | clean |

## Live process state

- `tmux ls`: `canonical-scaling: 1 windows (created Thu Jul 23 09:43:51 2026)`
- Pane state: session `canonical-scaling`, pane `0.0`, PID 18125, current
  command `bash`, pane not dead.
- `train_evolvegcn_h` processes: none.
- `run_canonical_evolvegcn_scaling_missing` processes: none.
- Training active: **no**.
- Canonical/similar tmux session: **yes, idle**.

## Canonical protocol

The execution plan, runner, verifier, completed pilot config, reusable configs,
and lightweight dataset metadata agree on:

| Setting | Canonical value |
|---|---|
| Model | EvolveGCN-H (`EvolveGCNHRegressor`) |
| Universes | 20, 50, 100, 200, 500 |
| Seeds | 42, 123, 2025 |
| Top-N / snapshots | 100 / 5 |
| Normalization | minmax |
| Graph | periodic kNN, k=8, box size 25 |
| Architecture | hidden 32, 2 layers, dropout 0.2, ReLU |
| Readout | graph mean, temporal mean, MLP head |
| Optimization | batch 4, maximum 300 epochs, patience 40, LR 0.001, weight decay 0.00001, gradient clipping 1.0 |
| Self-loops | enabled |
| Splits | 0.70 / 0.15 / 0.15 |
| Summary features / target normalization | disabled / disabled |

No scientific-setting discrepancy was found in the ten complete canonical
rows. The three legacy 20U/50U/100U seed-123 configs omit explicit
`activation`, `head_type`, `use_summary_features`, and `normalize_target`;
the verifier deliberately resolves these to the training source defaults
`relu`, `mlp`, `false`, and `false`. This is a serialization difference, not a
protocol difference.

## All 15 required rows

Full machine-readable details are in `canonical_scaling_live_status.csv`.

| U | Seed | Exact experiment | Origin | Artifacts | Compatible | Status | MAE | RMSE | R² | Best epoch | Test n |
|---:|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|
| 20 | 42 | `evolvegcn_h_u20_top100_norm-minmax_h32_l2_gpool-mean_tpool-mean_head-mlp_canonical300_seed42` | replacement | 5/5 | yes | complete/reusable | 0.0946318010489146 | 0.10297306709756288 | 0.048888086725889845 | 28 | 3 |
| 20 | 123 | `evolvegcn_h_20u_seed123_final32` | historical | 5/5 | yes | complete/reusable | 0.10589082042376201 | 0.11243401456713151 | 0.33716093292584237 | 19 | 3 |
| 20 | 2025 | `evolvegcn_h_u20_top100_norm-minmax_h32_l2_gpool-mean_tpool-mean_head-mlp_canonical300_seed2025` | replacement | 0/5 | not verifiable | incomplete/excluded | — | — | — | — | — |
| 50 | 42 | `evolvegcn_h_u50_top100_norm-minmax_h32_l2_gpool-mean_tpool-mean_head-mlp_canonical300_seed42` | replacement | 0/5 | not verifiable | incomplete/excluded | — | — | — | — | — |
| 50 | 123 | `evolvegcn_h_50u_seed123_final32` | historical | 5/5 | yes | complete/reusable | 0.10247249715030193 | 0.12100125069644337 | 0.05712601547643881 | 8 | 8 |
| 50 | 2025 | `evolvegcn_h_u50_top100_norm-minmax_h32_l2_gpool-mean_tpool-mean_head-mlp_canonical300_seed2025` | replacement | 0/5 | not verifiable | incomplete/excluded | — | — | — | — | — |
| 100 | 42 | `evolvegcn_h_u100_top100_norm-minmax_h32_l2_gpool-mean_tpool-mean_head-mlp_canonical300_seed42` | replacement | 0/5 | not verifiable | incomplete/excluded | — | — | — | — | — |
| 100 | 123 | `evolvegcn_h_100u_seed123_final32` | historical | 5/5 | yes | complete/reusable | 0.1262190709511439 | 0.14484999299933063 | -0.4455000601187815 | 8 | 15 |
| 100 | 2025 | `evolvegcn_h_u100_top100_norm-minmax_h32_l2_gpool-mean_tpool-mean_head-mlp_canonical300_seed2025` | replacement | 0/5 | not verifiable | incomplete/excluded | — | — | — | — | — |
| 200 | 42 | `evolvegcn_h_200u_seed42_final32` | historical | 5/5 | yes | complete/reusable | 0.09758373474081357 | 0.11099666591965886 | -0.009786346018163083 | 8 | 30 |
| 200 | 123 | `evolvegcn_h_200u_seed123_final32` | historical | 5/5 | yes | complete/reusable | 0.0880318875114123 | 0.10571293071466246 | 0.005183752891004589 | 5 | 30 |
| 200 | 2025 | `evolvegcn_h_200u_seed2025_final32` | historical | 5/5 | yes | complete/reusable | 0.11484164421757062 | 0.12910925014056243 | -0.015509614338739874 | 5 | 30 |
| 500 | 42 | `evolvegcn_h_500u_seed42_final32` | historical | 5/5 | yes | complete/reusable | 0.08653705775737762 | 0.10205947521376181 | -0.014195702403178245 | 8 | 75 |
| 500 | 123 | `evolvegcn_h_500u_seed123_final32` | historical | 5/5 | yes | complete/reusable | 0.09924482007821402 | 0.11675728841112959 | -0.05300625110331092 | 65 | 75 |
| 500 | 2025 | `evolvegcn_h_500u_seed2025_final32` | historical | 5/5 | yes | complete/reusable | 0.10510393093029657 | 0.12090740337632608 | -0.029850955944742852 | 5 | 75 |

For every complete row, all five required artifacts exist, saved test counts
match prediction-row counts, prediction target/value columns are finite, and
split counts match `floor(0.70U) / floor(0.15U) / remainder`: 14/3/3,
35/7/8, 70/15/15, 140/30/30, and 350/75/75.

## Pilot status

Folder:
`experiments/evolvegcn_h_u20_top100_norm-minmax_h32_l2_gpool-mean_tpool-mean_head-mlp_canonical300_seed42`

- Folder exists: yes.
- All five required artifacts: yes.
- Successful completion: yes; its log ends with
  `EVOLVEGCN-H TRAINING COMPLETE`.
- Finite predictions: yes, all three targets and predictions.
- Test MAE: `0.0946318010489146`.
- Test RMSE: `0.10297306709756288`.
- Test R² recomputed: `0.048888086725889845`.
- Best epoch: `28`.
- Counts: train 14, validation 3, test 3.
- Canonical config match: yes.
- Runner behavior: it will skip this folder because all five artifacts exist.

Relevant logs:

- `logs/canonical_evolvegcn_scaling/source_provenance.txt` records branch
  `thesis-controlled-scaling`, commit `f2ccae3`, and hashes for the clean
  EvolveGCN-H model/training sources at 2026-07-23 09:43:41 UTC.
- `logs/canonical_evolvegcn_scaling/u20_seed42_pilot.log` records the complete
  68-epoch training trace, early stopping after 40 non-improving epochs, final
  metrics, and completion marker.
- There is no log showing that the six-run canonical runner itself was
  launched; only the pilot and provenance logs exist.

## Runner safety status

- `/usr/bin/time` fallback present: yes, in the unstaged edit.
- Shell syntax: `bash -n` passes.
- `run_one` invocations: exactly six.
- Run order:
  1. 20U seed 42
  2. 20U seed 2025
  3. 50U seed 42
  4. 50U seed 2025
  5. 100U seed 42
  6. 100U seed 2025
- Complete folders: skipped only after all five required files pass.
- Partial folders: cause exit status 4 before any training for that row.
- Execution: sequential; a failed run stops subsequent runs.
- Scientific settings: unchanged by the fallback diff and match the protocol.
- Safety conclusion: **safe to execute after the working-tree fallback is
  committed/pushed for provenance**. The existing pilot will be skipped; the
  remaining five rows will run sequentially. This conclusion is static
  inspection only; the script was not executed.

## Verifier status

Only this permitted command was run:

```text
python3 scripts/verify_canonical_evolvegcn_scaling.py --help
```

It succeeded and documents `--repo-root` plus optional `--output`. The full
verification was correctly not run because five required folders are absent.

Expected path logic:

- Planned replacement paths for 20U/50U/100U seeds 42 and 2025.
- Historical `*_seed123_final32` paths for 20U/50U/100U seed 123.
- Historical `evolvegcn_h_{200,500}u_seed{42,123,2025}_final32` paths.

Therefore it correctly combines six named replacements with nine historical
runs and excludes the old 200-epoch low-U seed-42/2025 folders. Once all rows
exist, it prints seed-level coverage and per-universe mean/sample-SD for MAE,
RMSE, R², and Pearson, plus best seed. With `--output`, it writes a JSON object
containing protocol, experiments, summaries, errors, and validity. Without
`--output`, it writes nothing.

Verifier readiness: **the verifier code/help is ready, but full verification is
not yet actionable as a passing final gate**.

## Exact last completed actions

1. **Last committed action:** commit `f2ccae3` on 2026-07-23 09:37:31 UTC,
   “Add reproducible canonical EvolveGCN-H scaling workflow.”
2. **Last pushed action:** the same commit to
   `origin/thesis-controlled-scaling`; live remote inspection confirms the
   branch currently points to `f2ccae3`.
3. **Last confirmed completed training run:** canonical replacement 20U seed
   42. The experiment folder has all artifacts, the dedicated pilot log ends
   with the completion marker, and the metrics/predictions agree. File times
   place completion near 2026-07-23 10:24:16 UTC, but the conclusion rests on
   the log marker and raw artifacts, not the timestamp alone.
4. **Last script modification:** the unstaged optional `/usr/bin/time` fallback
   in `scripts/run_canonical_evolvegcn_scaling_missing.sh`. The content diff
   proves the current change; its filesystem time is 2026-07-23 11:36:20 UTC.
5. **Last recorded substantive verification:** registry generation/validation
   at 2026-07-11 09:07:07 UTC (0 fatal errors, 50 stale-output mismatches).
   There is no artifact showing a full canonical verifier run. During this
   inspection, verifier `--help` and runner `bash -n` succeeded; neither is a
   full result verification.
6. **Currently running:** no training and no runner; only an idle
   `canonical-scaling` tmux shell.

## Immediate next action

Exactly one immediate action: commit and push the runner fallback before
launching any remaining experiment, so future runs have a clean, remotely
recoverable source state.

Exact command (do not run as part of this inspection):

```bash
git add scripts/run_canonical_evolvegcn_scaling_missing.sh && git commit -m "Allow canonical scaling without GNU time" && git push origin thesis-controlled-scaling
```

## Git status

Final `git status --short` after creating and validating only the three
requested reports:

```text
 M notebooks/visualization/01_graph_sanity_checks.ipynb
 M notebooks/visualization/02_graph_models_20u_to_200u_scaling_results.ipynb
 M scripts/run_canonical_evolvegcn_scaling_missing.sh
 M src/models/static_gcn.py
 M src/training/train_static_gcn.py
?? docs/
?? experiments/
?? notebooks/07_evolvegcn_500u_top500_vs_750u_top1000_results.ipynb
?? notebooks/validation/
?? notebooks/visualization/03_graph_models_20u_to_500u_clean_results.ipynb
?? notebooks/visualization/04_knn_ablation_500u_h64_results.ipynb
?? notebooks/visualization/05_evolvegcn_h_diagnostic_investigation.ipynb
?? notebooks/visualization/06_normalization_ablation_top500.ipynb
?? notebooks/visualization/07_evolvegcn_500u_top500_vs_750u_top1000_results.ipynb
?? notebooks/visualization/08_evolvegcn_750u_top1000_head_ablation_results.ipynb
?? notebooks/visualization/09_evolvegcn_750u_top1000_mean_vs_meanmax_pooling_results.ipynb
?? presentation_assets/
?? reports/experiment_registry/analysis_artifact_index.csv
?? reports/experiment_registry/baseline_validation_audit.md
?? reports/experiment_registry/canonical_scaling_family.csv
?? reports/experiment_registry/canonical_scaling_live_status.csv
?? reports/experiment_registry/controlled_experiment_matrix.csv
?? reports/experiment_registry/current_thesis_execution_status.md
?? reports/experiment_registry/experiment_family_summary.csv
?? reports/experiment_registry/family_compatibility_audit.md
?? reports/experiment_registry/final_table_readiness_status.md
?? reports/experiment_registry/incomplete_or_duplicate_experiments.csv
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
?? scripts/build_experiment_registry.py
?? scripts/run_knn_ablation_500u_training.sh
?? scripts/run_knn_ablation_k8_h64.sh
?? src/evaluation/diagnose_embedding_distribution_shift.py
?? src/evaluation/diagnose_embedding_feature_stability.py
?? src/evaluation/diagnose_embedding_neighborhood_consistency.py
?? src/evaluation/diagnose_embedding_probe_splits.py
?? src/evaluation/diagnose_embedding_target_relationship.py
?? src/evaluation/diagnose_evolvegcn_h_feature_variance.py
?? src/evaluation/diagnose_evolvegcn_h_head_analysis.py
?? src/evaluation/diagnose_evolvegcn_h_layer1_activations.py
?? src/evaluation/diagnose_evolvegcn_h_layer_variance.py
?? src/evaluation/diagnose_evolvegcn_h_regressor_head.py
?? src/evaluation/diagnose_evolvegcn_h_representations.py
?? src/evaluation/diagnose_evolvegcn_h_variance_flow.py
?? src/evaluation/diagnose_graph_vs_summary_baseline.py
?? src/evaluation/diagnose_head_vs_optimal_linear_solution.py
?? src/evaluation/diagnose_prediction_collapse_summary.py
?? src/evaluation/diagnose_predictions.py
?? src/evaluation/diagnose_split_target_distribution.py
?? src/evaluation/diagnose_summary_vs_embedding_combined.py
?? src/evaluation/run_summary_feature_baseline.py
?? src/evaluation/run_summary_mlp_baseline.py
```
