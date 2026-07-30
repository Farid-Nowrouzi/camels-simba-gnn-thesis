# Analysis Reporting Pipeline

## Purpose

The analysis-reporting pipeline turns verified experiment artifacts into
reproducible scientific tables, plots, compatibility evidence, and narrative
summaries. It is separate from training:

- training creates experiment folders and model artifacts;
- family verification checks one controlled experiment family;
- analysis reporting compares one or more already-complete families.

The reporting pipeline reads only lightweight `config.json`, `metrics.json`,
dataset metadata sidecars, and `predictions/test_predictions.csv`. It does not
load `.pt` datasets or checkpoints, run notebooks, or start training.

## Commands

Validate without writing a report:

```bash
python3 scripts/validate_analysis_report.py \
  --repo-root /home/ml/thesis-camels \
  --spec configs/analysis_reports/controlled_static_vs_evolvegcn_universe_scaling_top100.json
```

Build after validation:

```bash
python3 scripts/build_analysis_report.py \
  --repo-root /home/ml/thesis-camels \
  --spec configs/analysis_reports/controlled_static_vs_evolvegcn_universe_scaling_top100.json
```

The build command performs the same validation first and writes nothing when
validation fails.

## Creating an analysis specification

Copy `configs/analysis_reports/TEMPLATE.json`, give it a unique
`analysis_name`, and set:

- a title and scientific question;
- the analysis type;
- labeled experiment-family specification paths;
- grouping field and complete ordered grouping values;
- required seeds and metrics;
- required fixed protocol fields as dotted family-spec paths;
- every intentionally different protocol field with a scientific reason;
- a repository-relative output directory;
- prediction-plot setting, representative-seed policy, and figure formats.

Supported analysis types are `single_family`, `paired_family`,
`multi_family`, `ablation`, and `scaling`. The type describes scientific
intent; validation is always driven by the listed families and protocol
requirements.

An optional `reference_family` can document a baseline or reference
specification. It is provenance metadata unless it is also listed in
`families`, in which case it participates in verification and aggregation.

## Compatibility and scientific safety

Each listed family is passed through the existing generic family verifier.
This requires every non-excluded row to be artifact-complete and compatible
with its family specification. Saved MAE, RMSE, and MSE are compared with
values recomputed from raw prediction CSVs at the configured tolerance. R² and
Pearson correlation are recomputed, including negative R² values.

The cross-family layer additionally requires:

- the exact grouping-value/seed product for every family;
- unique experiment paths, preventing duplicate reproduction from being
  counted as another independent seed;
- all required fixed protocol fields to match;
- every observed cross-family difference to be declared explicitly;
- identical ordered split signatures for matching grouping values and seeds.

Any unexpected difference produces a non-zero validation exit. The builder
does not create final figures or tables after such a failure. Notebook values
and manually copied summary tables are never inputs.

## Output package

Each package contains:

- `analysis_manifest.json` with the specification, UTC generation timestamp,
  Git state, SHA-256 source hashes, experiment paths, exclusions, row counts,
  and representative runs;
- `compatibility_report.md`;
- protocol tables in CSV and Markdown;
- raw-derived seed-level, aggregate, and paired-difference CSV files;
- main results tables in CSV, Markdown, and LaTeX;
- `scientific_summary.md`;
- PNG/PDF figures under `figures/`;
- exact plotting CSVs under `plot_data/`.

Aggregate uncertainty is the sample standard deviation across seed-level
metrics (`n-1`). Test samples are not pooled.

## Representative-seed policies

Prediction and residual panels support:

- `median_test_mae` (default): sort seeds by test MAE and select the median;
- `explicit_seed`: select the configured integer `seed`;
- `best_test_mae`;
- `worst_test_mae`;
- `all_seeds`.

The default deliberately avoids automatically presenting the best seed.
Selected runs are recorded in the manifest and scientific summary.

## Adding a plot

Add a plotting function or a configuration-controlled branch in
`scripts/analysis_reporting/common.py`. Use matplotlib only, consume the
validated seed/aggregate/prediction structures, and save the exact plotted
rows to `plot_data/<figure_name>.csv`. Preserve negative metrics, full error
bars, meaningful reference lines, readable labels, and both configured output
formats.

## Regenerating after new runs

First verify or finalize each experiment family with the generic experiment
pipeline. Then rerun the analysis validator and builder. The report is derived
again from raw configs, metrics, and predictions, and its manifest records new
hashes and Git state.

Never add an incomplete family merely to fill a table. If a protocol differs,
either declare and justify the intentional difference or create a genuinely
matched experiment family. Do not hide failed rows, negative R², incompatible
settings, or non-monotonic trends.

## Static GCN versus EvolveGCN-H example

The supplied universe-scaling analysis matches universe counts, Top100/minmax
periodic-kNN graph construction, features, seeds, and exact split IDs. It
explicitly documents that Static GCN uses a native final snapshot, its
established MLP head, and batch size 8, while EvolveGCN-H uses five snapshots,
its canonical MLP head, and batch size 4.

The resulting comparison describes performance of the complete canonical
protocols. It cannot isolate a causal effect of temporal modeling, head
architecture, or batch size.
