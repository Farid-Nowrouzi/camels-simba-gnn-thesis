# Notebook 17 — final Set Transformer provenance

Closure record created: 2026-09-10T08:31:57.700524+00:00. Scientific finalization: 2026-09-10T08:14:55.888764+00:00.

## Scientific question and frozen protocol

Can learned permutation-invariant set attention without an explicit graph recover the performance of explicit graph message passing? The primary comparison is DeepSets vs Set Transformer vs Static GCN, all Top1500, U1000, raw7, final a=1.0, seeds 42/123/2025 and exact 700/99/201 splits.

Protocol: `reports/experiment_registry/u1000_top1500_set_transformer_protocol_freeze.json`; SHA256 `a344737fcc06197d0ecde0fc3f7bc37980bbff665b5bb0022eb726c933c4e051`. Dataset SHA256 `ff6f6a89517c0b67a96a8733ce5778dba1524df441cc23c27fbfa4e2f5cdb113`. Exact split paths/hashes and architecture are in the [result freeze](../experiment_registry/u1000_top1500_set_transformer_final_result_freeze.json).

Architecture: 7→16 projection, two ISABs, two heads, 32 inducing points, one PMA seed, 16→32→16→1 regression head; dropout 0.2; 11,313 parameters. Training configuration remains frozen.

## Implementation and scientific runs

Production implementation commit: `ff6b88f465bd0291980713fb7c595e745423839c`; frozen ancestor: `9b515aa70d785f691ccfa2f0ec4e7833875a4359`. All production implementation Git blobs match the protocol hashes. Attribution comes from the verified source/protocol chain; the run config does not itself record a Git commit.

Set Transformer:

- `experiments/set_transformer_u1000_top1500_raw7_final_train700_seed42_none_d16_h2_m32_isab2_pma1/`
- `experiments/set_transformer_u1000_top1500_raw7_final_train700_seed123_none_d16_h2_m32_isab2_pma1/`
- `experiments/set_transformer_u1000_top1500_raw7_final_train700_seed2025_none_d16_h2_m32_isab2_pma1/`

DeepSets:

- `experiments/deepsets_u1000_top1500_raw7_final_train700_seed42_none_h32_phi3_mean_statichead/`
- `experiments/deepsets_u1000_top1500_raw7_final_train700_seed123_none_h32_phi3_mean_statichead/`
- `experiments/deepsets_u1000_top1500_raw7_final_train700_seed2025_none_h32_phi3_mean_statichead/`

Static GCN:

- `experiments/static_gcn_u1000_top1500_sparse_train700_seed42_none_h32_l3_mean_mlp_final/`
- `experiments/static_gcn_u1000_top1500_sparse_train700_seed123_none_h32_l3_mean_mlp_final/`
- `experiments/static_gcn_u1000_top1500_sparse_train700_seed2025_none_h32_l3_mean_mlp_final/`

## Integrity verification

All required files exist and are nonempty. Checkpoint tensors are finite and give exact parameter counts 11,313 / 4,161 / 5,281. Set Transformer checkpoint configs and strict minimum validation-MSE epochs match saved configs/logs. No forward pass was performed. Full dataset SHA256, all split files, all production implementation blobs, and all 42 frozen control-manifest entries were verified.

Every controlled train/validation/test CSV has unique IDs in exact frozen manifest order and 700/99/201 rows. MAE, MSE, RMSE and R² were recomputed in float64, with absolute tolerance 1e-10 and zero relative tolerance. Maximum absolute disagreement: 1.1102230246251565e-15.

All three metadata states are `finalized` with `test_status=evaluated_once`, finite test metrics and 201 test predictions. Persistent `test_evaluation_started.json` files are expected anti-retry claims, not interrupted states; their timestamps precede finalization.

Provenance qualifications:

- The control-manifest hash frozen under CRLF is `b329b1995facd81d4c38ba815d90faf3f8f66297fb7ed51e076eb523bac82ec3`; the committed LF hash is `cfb16e0241942752075e2ddc90499c41a5aff93b3a61d295abfbaec6bccad831`. LF→CRLF conversion in memory reproduces the frozen hash exactly. No original file was changed.
- The only production-source edit is the necessary Set Transformer registry parser correction. Production and closure parser hashes are recorded separately. Model, runner, family config and protocol remain unchanged. The frozen runner’s whole-tree hash guard would reject this reporting-only parser change; the production parser remains available at the implementation commit. No training/finalization rerun is needed or authorized.
- All analysis artifact paths resolve inside the canonical repository; no sibling worktree is needed. Existing sibling worktrees are retained.

## Final metrics

Equal-weight mean ± sample SD (ddof=1); raw saved prediction values, not pooled predictions.

| Seed | Best epoch | Val MAE | Test MAE | Test MSE | Test RMSE | Test R² |
|---|---|---|---|---|---|---|
| 42 | 51 | 0.048183334 | 0.049112603 | 0.003724772 | 0.061030911 | 0.745902033 |
| 123 | 45 | 0.048932389 | 0.048834171 | 0.003926225 | 0.062659593 | 0.698497431 |
| 2025 | 49 | 0.049232820 | 0.051357242 | 0.004477932 | 0.066917354 | 0.690980211 |

| Model | Parameters | Val MAE | Test MAE | Test MSE | Test RMSE | Test R² |
|---|---|---|---|---|---|---|
| DeepSets | 4161 | 0.046316332 ± 0.001693555 | 0.048050625 ± 0.000985758 | 0.003803657 ± 0.000111976 | 0.061669346 ± 0.000907344 | 0.728927384 ± 0.012366442 |
| Set Transformer | 11313 | 0.048782848 ± 0.000540488 | 0.049768005 ± 0.001383342 | 0.004042976 ± 0.000389918 | 0.063535953 ± 0.003039500 | 0.711793225 ± 0.029777260 |
| Static GCN | 5281 | 0.038022503 ± 0.002814522 | 0.037886619 ± 0.000941551 | 0.002435524 ± 0.000061803 | 0.049348397 ± 0.000623944 | 0.826322570 ± 0.010565686 |

| Seed | Set Transformer − DeepSets Test MAE | Set Transformer − Static GCN Test MAE |
|---|---|---|
| 42 | -0.000058069 | +0.010916602 |
| 123 | +0.001167958 | +0.012004867 |
| 2025 | +0.004042251 | +0.012722688 |
| Mean ± sample SD | +0.001717380 ± 0.002104651 | +0.011881386 ± 0.000909353 |

## Interpretation and limitations

Under the tested fixed architecture and matched data protocol, Set Transformer did not improve mean test MAE over DeepSets and remained behind Static GCN on every seed. Generic learned set attention in this configuration was insufficient to reproduce the advantage observed with explicit periodic kNN graph message passing. Set Transformer has more parameters than both controls, so additional parameter count alone does not explain the Static GCN gain.

Set Transformer narrowly beats DeepSets on seed 42 but is worse on seeds 123 and 2025; its average is worse. Three seeds support a descriptive comparison, not a general superiority theorem or a formal significance claim. This is not a pure causal effect of topology: capacity, depth and optimization differ. No claim is made about all Set Transformers.

## Registry and final selections

Existing supported registry workflow: `bash scripts/refresh_experiment_registry.sh`. Registry 394→397 rows, with all 394 historical rows field-for-field identical and exactly three completed Set Transformer rows; raw7, Top1500, U1000, final snapshot, correct metrics, seeds and split counts. No smoke scientific rows or new duplicates. Existing 50 stale-output mismatches and one legacy duplicate group remain documented; zero metric disagreements and zero scan errors.

Policy A: `thesis_final_selection.csv` is byte-for-byte unchanged. Gradient Boosting Top1500, graph selections and DeepSets Top1000 headline are retained. This result freeze is the additive Notebook17 Top1500 comparison record.

## Broad thesis comparison availability

All families below have three final runs (42/123/2025), matching ordered split IDs and validation/test metrics verified independently. Targets agree within 1e-7 across representations. Exact directories and per-run metrics are in the result freeze. “Safe” assumes the stated representation qualifications.

| Model | Available / final | Seeds | Top-N | Input | Safe for Notebook17? |
|---|---|---|---|---|---|
| Target mean | Yes / yes | 42,123,2025 | N/A | target only | Yes |
| Ridge | Yes / yes | 42,123,2025 | 1500 | static summary20 | Yes |
| Random Forest | Yes / yes | 42,123,2025 | 1500 | static summary20 | Yes |
| Gradient Boosting | Yes / yes | 42,123,2025 | 1500 | static summary20 | Yes |
| Summary MLP | Yes / yes | 42,123,2025 | 1500 | static summary20 | Yes |
| DeepSets | Yes / yes | 42,123,2025 | 1500 | static raw7 | Yes |
| Set Transformer | Yes / yes | 42,123,2025 | 1500 | static raw7 | Yes |
| Static GCN | Yes / yes | 42,123,2025 | 1500 | static raw7 graph | Yes |
| PNA | Yes / yes | 42,123,2025 | 1500 | static raw7 graph | Yes |
| EvolveGCN-H | Yes / yes | 42,123,2025 | 1500 | temporal raw7 graph, 5 snapshots | Yes |
| EvolveGCN-O | Yes / yes | 42,123,2025 | 1500 | temporal raw7 graph, 5 snapshots | Yes |
| GCN-GRU | Yes / yes | 42,123,2025 | 1500 | temporal raw7 graph, 5 snapshots | Yes |
| GCN Temporal Transformer | Yes / yes | 42,123,2025 | 1500 | temporal raw7 graph, 5 snapshots | Yes |

Target mean is trained only on training targets and is Top-N independent; its stored Top1000 dataset binding has identical split IDs/targets. Engineered summary20 models are contextual baselines. Temporal graph models use five snapshots. Neither is a matched-input architecture ablation against raw7 final-snapshot sets. EvolveGCN-O seed 123 retains its documented inference-only recovery provenance. Use canonical finalized temporal runs, not interrupted directories.

## Notebook17 readiness

| Item | Ready? | Evidence |
|---|---|---|
| A. Primary controlled comparison | Yes | Nine verified Top1500 runs in this result freeze. |
| B. True-vs-predicted plots | Yes | Saved universe IDs, targets and predictions for all nine runs. |
| C. Residual plots | Yes | Residual = prediction minus target from saved CSVs. |
| D. Prediction range/compression | Yes | Saved targets and predictions; registry prediction ranges and standard-deviation ratios. |
| E. Validation/test tables | Yes | Float64 recomputation and seed-level means/sample SDs in this record. |
| F. Parameter comparison | Yes | Finite checkpoint tensor counts: 4161, 11313, 5281; no forward pass. |
| G. Broad model leaderboard | Yes | 13 families, 39 runs with matched split IDs and independently verified validation/test metrics; representation and temporal qualifications retained. |
| H. Historical context / experiment progression | Yes | All 394 historical registry rows preserved; Notebooks 1-16, selection freezes, final-thesis manifest and authoritative replacement tables retained. |

## Validation and closure policy

Four existing protocol/control/smoke tests passed before the reporting edit; four focused registry regression tests passed after it. Existing final-thesis artifact validation passed: 24 headline artifacts, eight selections and ten authoritative deprecated-table replacement rows. Changed Python compiles. Registry validation reports zero errors and zero saved/recomputed metric disagreements. Final file/schema/hash checks are performed against this record and its manifest.

Training and scientific test evaluation are complete. No retuning, model selection, architecture changes, or hyperparameter changes from these test results is permitted. Do not repeat training, inference, or test evaluation for model selection.

No checkpoints, predictions, metrics, existing notebooks, or final selections were changed. No training, inference or test evaluation was run. No notebook was created. No files were staged, committed or pushed. Next: create Notebook17 from these frozen artifacts only after a separate request.
