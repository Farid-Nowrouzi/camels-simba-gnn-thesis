# Notebook and Experiment-Registry Audit

This is a read-only content/metadata audit. Notebook outputs were measured from
their JSON without execution. Reports were not regenerated.

## Notebook findings

Embedded-output sizes below are serialized JSON estimates and can differ
slightly from the notebook file's byte contribution.

| Notebook | Purpose and experiment family | File / output bytes | Paths and source state | Recommendation |
|---|---|---:|---|---|
| `notebooks/visualization/01_graph_sanity_checks.ipynb` | Graph construction and sanity validation | 8,125,817 / 8,072,908 | Absolute `/home` paths occur in source and output. The only HEAD change is kernelspec display name. Output cell 11 contains an email-like string and a password/secret-marker pattern; no value is reproduced here. No stale summary CSV dependency was found. | Category H: restore the metadata-only change after approval; separately sanitize the already tracked output before any future notebook refresh. |
| `notebooks/visualization/02_graph_models_20u_to_200u_scaling_results.ipynb` | Historical Static GCN/EvolveGCN-H universe scaling, 20U–200U | 423,406 / 363,380 | Absolute paths occur in output. Only HEAD change is kernelspec display name. Source reads ignored `outputs/experiment_summary_clean_final32_complete_clean.csv`, which predates the controlled family. | Category H for the current metadata change; regenerate the scientific content from the registry rather than recommit old metrics. |
| `notebooks/07_evolvegcn_500u_top500_vs_750u_top1000_results.ipynb` | Top-N/universe comparison | 1,003,266 / 902,354 | Absolute paths occur in output. Reads stale ignored `evolvegcn_500top500_vs_750top1000_5seed_table.csv`. Cell-source hash and serialized outputs match visualization/07. | Category G: duplicate; exclude this root copy and remove only after approval. |
| `notebooks/validation/01_raw_dataset_integrity_validation.ipynb` | Raw CAMELS dataset integrity | 37,573 / 12,921 | Absolute paths occur only in output; no stale summary CSV reference. Validation output points to ignored local data/output state. | Category B after clearing/sanitizing output and checking portable root discovery. |
| `notebooks/visualization/03_graph_models_20u_to_500u_clean_results.ipynb` | Historical graph-model scaling through 500U | 996,362 / 944,533 | Absolute paths occur in output. It aggregates historical protocol-mixed families; metrics are stale for the controlled thesis table even without one detected stale filename. | Category E: regenerate from the final controlled registry. |
| `notebooks/visualization/04_knn_ablation_500u_h64_results.ipynb` | Controlled k=4/6/8/12 ablation at 500U h64 | 677,219 / 619,395 | Source hard-codes `Path("/home/ml/thesis-camels")`; output also contains absolute paths. No stale summary CSV reference was detected, but protocol/value validation is still required. | Category I: focused review and portability fix before a notebook commit. |
| `notebooks/visualization/05_evolvegcn_h_diagnostic_investigation.ipynb` | EvolveGCN-H representation/prediction-collapse investigation | 120,672 / 66,957 | No absolute path or stale summary CSV reference detected. Consumes experiment diagnostic JSON. | Category B: commit with the tested diagnostic scripts after verifying the represented runs. |
| `notebooks/visualization/06_normalization_ablation_top500.ipynb` | Top-500 normalization ablation | 1,379,348 / 1,077,083 | Absolute paths occur only in output. No stale summary CSV reference detected; embedded metrics appear to represent the controlled family but were not recomputed during inspection. | Category B after output sanitation and independent metric check. |
| `notebooks/visualization/07_evolvegcn_500u_top500_vs_750u_top1000_results.ipynb` | Top-N/universe comparison | 1,003,161 / 902,354 | Absolute paths occur in output and source reads the stale ignored five-seed table. It is the canonical-location copy of root notebook 07. | Category E: retain this location, regenerate after the canonical runs, then reconsider category B. |
| `notebooks/visualization/08_evolvegcn_750u_top1000_head_ablation_results.ipynb` | Five-seed MLP-versus-linear head ablation | 1,130,954 / 1,077,243 | Absolute paths occur only in output. No stale summary CSV reference detected; metrics were not independently recomputed here. | Category B after output sanitation and registry cross-check. |
| `notebooks/visualization/09_evolvegcn_750u_top1000_mean_vs_meanmax_pooling_results.ipynb` | Five-seed mean-versus-meanmax pooling ablation | 1,307,392 / 1,245,368 | Absolute paths occur only in output. No stale summary CSV reference detected; metrics were not independently recomputed here. | Category B after output sanitation and registry cross-check. |

The two notebook-07 files are not intentionally different analyses. Their
concatenated cell sources have the same SHA-256
(`6965670189af…`), their serialized cell outputs have the same byte count, and
comparison found no cell-output difference. Their whole-file sizes differ by
105 bytes because of notebook-level metadata/container differences. The
visualization-directory path is the canonical location.

For notebooks 01 and 02, `git diff HEAD` changes exactly one value each:
`kernelspec.display_name` from `camels-gnn` to
`camels-gnn (3.10.12.final.0)`. There is no scientific source or output change
in the working-tree diff.

## Registry report audit

“Safe now” means scientifically useful and not expected to change merely
because the five outstanding canonical runs finish. It does not authorize
staging.

| File | Report role | Currency and decision |
|---|---|---|
| `analysis_artifact_index.csv` | Generated summary/index | 2026-07-11 builder output; stale and regenerate after experiments (E). |
| `baseline_validation_audit.md` | Scientific audit | 2026-07-23; safe now only after resolving its explicit “needs one confirmation” condition (D). |
| `canonical_evolvegcn_scaling_execution_plan.md` | Source-of-truth execution plan | Tracked and clean; durable pre-run rationale, already committed. |
| `canonical_evolvegcn_scaling_run_matrix.csv` | Source-of-truth execution matrix | Tracked and clean; exact planned commands/status baseline, already committed. |
| `canonical_scaling_family.csv` | Generated canonical-family extract | Predates the replacement runs; stale, regenerate after experiments (E). |
| `canonical_scaling_live_status.csv` | Live status snapshot | Current snapshot reports the pilot plus historical compatible runs and 10/15 completion; wait, then regenerate (E). |
| `controlled_experiment_matrix.csv` | Generated scientific comparison matrix | Pre-completion matrix; wait and regenerate (E). |
| `current_thesis_execution_status.md` | Live status snapshot | Dated 2026-07-29 and changes as runs finish; wait (E). |
| `experiment_family_summary.csv` | Generated summary | 2026-07-11 registry build; stale and regenerate (E). |
| `family_compatibility_audit.md` | Scientific audit generated from registry | Its method is valuable, but inputs are stale; regenerate after experiments (E). |
| `final_table_readiness_status.md` | Live status snapshot | Explicitly a readiness snapshot; wait for five runs (E). |
| `incomplete_or_duplicate_experiments.csv` | Generated registry exception list | 2026-07-11 state; regenerate (E). |
| `master_experiment_registry.csv` | Source-of-truth registry output | Current file is stale; build metadata identifies branch `main`, commit `f260a566…`, dirty state. Regenerate after runs (E). |
| `master_experiment_registry.md` | Human-readable source-of-truth summary | Derived from the stale master CSV; regenerate (E). |
| `missing_run_decision_matrix.csv` | Temporary planning report | Useful until execution completes, then obsolete; do not include in final professor set (H). |
| `professor_request_experiment_inventory.md` | Professor-facing generated summary | Derived from stale 2026-07-11 registry; regenerate after runs (E). |
| `professor_table_blueprint.md` | Scientific audit/reporting policy | Stable rule: report compatible seed means ± sample SD and never mix protocols. Safe now (D). |
| `registry_build_metadata.json` | Source-of-truth build provenance | Explicitly stale branch/commit/dirty provenance; regenerate (E). |
| `registry_validation_report.md` | Generated validation summary | Validates the stale build; regenerate after runs (E). |
| `source_code_provenance_audit.md` | Scientific/source audit | Says the tree had six tracked files and 883 insertions/56 deletions, no longer the current diff; regenerate after source commits (E). |
| `thesis_completion_blueprint.md` | Temporary planning report | Valuable planning context but not final evidence; keep local until decisions are executed (H). |
| `thesis_seed_completion_plan.md` | Generated/stale plan | Superseded by the committed canonical runner/matrix and current live status; regenerate or exclude (E). |

The minimum professor-ready tracked set after the five runs is:

1. `master_experiment_registry.csv` and `.md`;
2. `registry_build_metadata.json`;
3. `registry_validation_report.md`;
4. `controlled_experiment_matrix.csv`;
5. `final_table_readiness_status.md`;
6. `professor_request_experiment_inventory.md`;
7. `professor_table_blueprint.md`.

Add the live-status CSV only if the final generated version is explicitly
frozen as the completed 15/15 execution record. The artifact index, broad
family summary, and compatibility audit are useful appendices but not the
minimum professor-facing set.

## Privacy and security scan

The scan covered lightweight text under `docs/`, `experiments/` excluding
checkpoints, `notebooks/`, `reports/`, `scripts/`, and `src/`. It found:

- Absolute private/server paths in
  `docs/experiment_pipeline.md`, the notebooks identified in the table,
  `reports/experiment_registry/canonical_evolvegcn_scaling_execution_plan.md`,
  and `reports/experiment_registry/registry_build_metadata.json`.
- One email-like value and one password/secret-marker pattern in embedded
  output cell 11 of `notebooks/visualization/01_graph_sanity_checks.ipynb`.
  Treat the cell as sensitive until manually inspected and sanitized. No
  suspected value is printed in this report.
- No private-network IP address match.
- No confirmed API token, SSH private key, or credential was identified by the
  lightweight pattern scan.

Absolute dataset paths disclose local storage layout even when they contain no
credential. Prefer repository-relative paths and record logical dataset IDs
plus hashes in tracked evidence.
