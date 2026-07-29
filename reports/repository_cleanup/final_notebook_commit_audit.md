# Final Thesis Notebook Commit Audit

Audit date: 2026-07-29 UTC

Branch: `thesis-controlled-scaling`
Pre-edit HEAD: `e6bd7f0`

No notebook cell was executed. No model was trained, and no `.pt` dataset,
checkpoint, experiment directory, metric, seed, or scientific setting was
loaded or changed.

## Preflight

- Branch: `thesis-controlled-scaling`.
- Upstream: `origin/thesis-controlled-scaling`.
- Local/remote divergence: `0 0`.
- Staged paths before editing: none.
- Matching training processes: none.
- Existing unrelated untracked experiment and registry-report paths were left
  untouched.

## Notebook decisions

| Notebook | Purpose / experiment family | Metric provenance | Decision |
|---|---|---|---|
| `notebooks/validation/01_raw_dataset_integrity_validation.ipynb` | Raw CAMELS-SIMBA storage and integrity validation | Validation evidence comes from raw-file metadata and saved validation tables; it is not a model-performance notebook. | Sanitize saved path text, then commit. |
| `notebooks/visualization/01_graph_sanity_checks.ipynb` | Graph construction, target inspection, and graph sanity checks | Uses saved graph-validation artifacts and scientific visualizations; no final model comparison is asserted. | Remove one sensitive embedded HTML output, sanitize paths, and commit. |
| `notebooks/visualization/02_graph_models_20u_to_200u_scaling_results.ipynb` | Historical Static GCN and EvolveGCN-H universe scaling, 20U–200U | Displayed metrics are indirectly traceable to an older generated experiment-summary CSV, not the current registry. | Retain as a dated research snapshot and commit. Regenerate final tables later. |
| `notebooks/visualization/03_graph_models_20u_to_500u_clean_results.ipynb` | Historical graph-model scaling through 500U | Code documents summary generation from experiment `metrics.json`, `test_predictions.csv`, and `train_log.csv`, but the saved view includes historical protocol families. | Retain as a dated research snapshot and commit. Regenerate final tables later. |
| `notebooks/visualization/04_knn_ablation_500u_h64_results.ipynb` | Controlled 500U h64 kNN ablation for Static GCN and EvolveGCN-H | Reads experiment configs, metrics, predictions, and graph-connectivity metadata directly. Dataset paths are present in code but no `.pt` file was loaded during this audit. | Replace fixed repository root, sanitize saved paths, and commit. |
| `notebooks/visualization/05_evolvegcn_h_diagnostic_investigation.ipynb` | EvolveGCN-H representation and prediction-collapse investigation | Reads named diagnostic JSON artifacts directly from experiment diagnostics. | Commit unchanged. |
| `notebooks/visualization/06_normalization_ablation_top500.ipynb` | Five-seed Top500 normalization ablation for Static GCN and EvolveGCN-H | Reads experiment configs, metrics, and test predictions directly and writes derived tables. | Sanitize saved path text and commit. |
| `notebooks/visualization/07_evolvegcn_500u_top500_vs_750u_top1000_results.ipynb` | Five-seed 500U Top500 versus 750U Top1000 EvolveGCN-H comparison | Reads saved seed/group summary CSVs and test predictions. The summary CSVs predate the latest canonical registry. | Keep the canonical copy as a dated snapshot, fix one missing newline, remove its failed traceback, and commit. Regenerate final tables later. |
| `notebooks/visualization/08_evolvegcn_750u_top1000_head_ablation_results.ipynb` | Five-seed MLP-versus-linear head ablation | Reads experiment configs, metrics, and predictions directly and records missing-run warnings. | Sanitize saved path text and commit. |
| `notebooks/visualization/09_evolvegcn_750u_top1000_mean_vs_meanmax_pooling_results.ipynb` | Five-seed mean-versus-meanmax pooling ablation | Reads experiment configs, metrics, and predictions directly and derives paired-seed evidence. | Sanitize saved path text and commit. |

## Duplicate notebook 07

Compared:

- A: `notebooks/07_evolvegcn_500u_top500_vs_750u_top1000_results.ipynb`
- B: `notebooks/visualization/07_evolvegcn_500u_top500_vs_750u_top1000_results.ipynb`

Before deletion, A and B had different whole-file SHA-256 hashes
(`84f7f34b…` and `585f42cc…`) and differed by 105 bytes. Detailed comparison
showed:

- identical notebook-level metadata and nbformat;
- 28 cells in each: 16 Markdown and 12 code;
- identical Markdown source, code source, execution counts, output objects,
  image/plot payloads, captions, attachments, data-source paths, and
  metric-bearing output;
- identical source SHA-256 `6965670189af…`;
- identical output SHA-256 `3054398c1f86…`;
- only five generated cell IDs differed.

A contained no unique scientific content and was an older container-level
duplicate. It was untracked and ignored, so the exact file
`notebooks/07_evolvegcn_500u_top500_vs_750u_top1000_results.ipynb` was removed
with a normal single-file deletion. No directory was deleted. B remains at the
required canonical location.

## Changes made

- Notebook 01: removed exactly one `execute_result` item in code cell 11 whose
  embedded interactive HTML matched email-like and secret-marker patterns. The
  output source cell, existence/path message, and all other scientific plots
  and analysis remain. No removed value was printed.
- Notebook 01: replaced a machine-specific repository path in an explanatory
  source comment.
- Notebook 04: replaced `Path("/home/ml/thesis-camels")` with upward repository
  discovery using `pathlib.Path` and `.git`/`notebooks` markers.
- All notebooks: replaced the exact machine-specific repository root in saved
  output strings with `<REPO_ROOT>`. Numeric results, plots, tables, dataset
  names, experiment names, metrics, and settings were unchanged.
- Notebook 07: split `display(output_artifacts)print(...)` into two valid source
  lines, then removed the obsolete saved `SyntaxError` traceback and cleared
  that cell's execution count. The cell was not executed.
- Added a `Research snapshot status` Markdown cell near the start of notebooks
  02, 03, and 07. Each states the represented historical experiment state and
  that final thesis tables will be regenerated from verified artifacts.

At preflight, notebooks 01 and 02 were already clean relative to HEAD; their
earlier kernelspec-only working-tree changes had already been restored. No
additional environment-only metadata was retained.

## Final file-size and structure table

Embedded output is the serialized size of all output objects after sanitation.

| Notebook | File bytes | Cells (Markdown/code) | Output cells | Embedded output bytes | Valid JSON |
|---|---:|---:|---:|---:|---|
| validation/01 | 37,518 | 15 (9/6) | 5 | 12,846 | yes |
| visualization/01 | 3,050,993 | 29 (15/14) | 14 | 3,025,369 | yes |
| visualization/02 | 423,740 | 53 (26/27) | 25 | 363,205 | yes |
| visualization/03 | 996,604 | 33 (17/16) | 16 | 944,292 | yes |
| visualization/04 | 677,513 | 34 (15/19) | 18 | 619,231 | yes |
| visualization/05 | 120,672 | 56 (42/14) | 14 | 66,873 | yes |
| visualization/06 | 1,379,249 | 36 (21/15) | 14 | 1,076,940 | yes |
| visualization/07 | 1,002,930 | 29 (17/12) | 10 | 901,678 | yes |
| visualization/08 | 1,130,162 | 33 (19/14) | 13 | 1,076,411 | yes |
| visualization/09 | 1,306,545 | 37 (21/16) | 15 | 1,244,477 | yes |

Notebook 01 was initially 8,125,799 bytes (over 5 MiB) because the sensitive
interactive HTML output alone occupied about 5.05 MB. After its required
removal, no notebook exceeds 5 MiB or 10 MiB.

## Validation

- All ten canonical notebooks parse as JSON.
- All contain `cells`, `metadata`, `nbformat`, and `nbformat_minor`.
- No merge-conflict marker was found.
- No notebook contains `/home/ml/thesis-camels` or another `/home/ml` path
  after sanitation.
- No scanned email, credential, private-key, password, or secret-marker pattern
  remains.
- No saved error/traceback output remains.
- Canonical visualization notebook 07 exists.
- Root-level duplicate notebook 07 is absent.
- Notebooks 02, 03, and 07 each contain exactly one dated-snapshot notice.
- `git diff --check` passes.
- Notebook cells were not executed.

## Commit scope

Approved for this notebook snapshot:

- all `.ipynb` files directly under `notebooks/validation/`;
- notebooks 01 through 09 directly under `notebooks/visualization/`;
- this audit report.

Excluded and untouched: `experiments/`, `data/`, `logs/`, `outputs/`,
`presentation_assets/`, `reports/experiment_registry/`, source code, model
files, evaluation scripts, checkpoints, and binary datasets.

Future regeneration is required for final-result versions of notebooks 02, 03,
and 07. Their saved results are deliberately retained as labelled historical
research snapshots and must not be cited as final thesis tables.
