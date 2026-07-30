# Notebook 10 presentation-repair validation

Final status: **PASS**

## Repair objective

Repair Notebook 10 for direct supervisor-meeting screen sharing without changing
scientific results. The original presentation risks were theme-sensitive
low-contrast tables, overly wide result tables, transparent figure canvases,
small figure rendering, incomplete protocol captions, and a prediction-collapse
section that did not show the progression from the controlled U500 Top-N family
to the stronger finalized U750 Top1000 protocol.

## Implemented presentation changes

- Replaced theme-sensitive pandas Styler output with deterministic HTML whose
  table, caption, header, body, row, and cell elements carry explicit inline
  styles. Tables use a white background, `#111827` body text, `#1f4e78` headers
  with white text, white/`#f3f6fa` zebra rows, borders, numeric right alignment,
  text left alignment, 13 px type, and padded cells. Critical colors use
  `!important`.
- Reduced central performance tables to factor/model, seed count, mean MAE, MAE
  SD, mean RMSE, mean R², and prediction-SD ratio. Prediction-repeat diagnostics
  are separate. The 16-cell factorial is split by model and hidden dimension.
  The prediction-compression protocol definition and outcome diagnostics are
  split into two keyed three-row tables so all required fields remain visible.
- Added report-level light-theme CSS with `color-scheme: light`, white page and
  output backgrounds, dark Markdown text and headings, accessible links, light
  code blocks, white figure cards, and print-safe rules.
- Replaced direct source-image display with a helper that reads finalized PNGs
  through Pillow, composites RGBA content over opaque white, writes only the
  selected presentation-safe copies, and displays them at meeting-readable
  width with full protocol captions.
- Retained all 25 analysis-source cells with the `hide-input` tag and VS Code
  source-collapsed metadata.

## Prediction-compression comparison

All displayed representatives follow the finalized `median_test_mae` policy;
no best seed was selected.

| Protocol | Finalized package | Representative | MAE | R² | Prediction-SD ratio | Repeat fraction |
|---|---|---:|---:|---:|---:|---:|
| U500 Top100, minmax, k8, h32, mean graph pooling, mean temporal pooling, linear head | `controlled_static_vs_evolvegcn_topn_halo_scaling_500u` | seed 123 | 0.099245 | -0.053006 | 0.139946 | 0.000000 |
| U500 Top500, minmax, k8, h32, mean graph pooling, mean temporal pooling, linear head | `controlled_static_vs_evolvegcn_topn_halo_scaling_500u` | seed 123 | 0.088085 | 0.138769 | 0.561369 | 0.000000 |
| U750 Top1000, none, k8, h32, two layers, mean graph pooling, mean temporal pooling, linear head | `controlled_evolvegcn_regression_head_ablation_750u_top1000` | seed 123 | 0.052324 | 0.656075 | 0.979391 | 0.000000 |

The U500 comparison is a controlled Top-N comparison within EvolveGCN-H. The
U750 panel is descriptive cross-protocol context because universe count and
normalization differ. The notebook explicitly states that the stronger U750
Top1000 result is not proof that Top1000 alone caused the improvement.

Search result: **no finalized controlled U500 Top1000 package is available**.
The finalized registry states that Top1000 is U750-only, so no U500 Top1000
panel was invented or sourced from an unverified experiment.

The comparison panel was deterministically redrawn from exact finalized
representative plot-data CSVs because the source PNGs combine the required
panels with other experimental cells. It retains the exact predictions,
identity line, representative seeds, and finalized diagnostics. It is a
presentation composition, not a new experiment.

## Execution and HTML checks

| Check | Result |
|---|---|
| Notebook parses and validates as nbformat | PASS |
| Notebook cells | 70 |
| Code cells executed | 25 / 25 |
| Code-cell execution counts | contiguous 1–25 |
| Error outputs | 0 |
| Traceback text in HTML | 0 |
| Required finalized sources present | PASS |
| Visible HTML tables | 28 |
| Tables with explicit white table background | 28 / 28 |
| Table header cells with explicit dark-blue background and white text | 155 / 155 |
| Maximum columns in a central table | 10 |
| Visible HTML images | 18 |
| Figure captions | 18 |
| Presentation-safe source-figure copies | 17 PNG |
| Prediction-comparison outputs | 1 PNG and 1 PDF |
| PNG files with nonzero size and valid dimensions | 18 / 18 |
| PNG files with any transparent pixel | 0 / 18 |
| Mandatory headings and protocol labels | PASS |
| Controlled-versus-descriptive qualification | PASS |
| Broken or missing image assets | 0 |
| External HTTP(S) image/script/stylesheet references | 0 |
| `color-scheme: light` present | PASS |
| Offline self-contained HTML | PASS |

Spot inspection covered the prediction-compression composite, the k × hidden
dimension main plot, and the graph-pooling paired plot. Axes, legends, protocol
labels, identity lines, diagnostic annotations, and captions are readable; no
axis labels or legends are clipped. Pillow inspection found dimensions from
1114×772 through 3079×1092 pixels, nonzero file sizes, opaque canvases, and
substantial white-background coverage.

Notebook size: **2,151,640 bytes**.

Offline HTML size: **2,398,134 bytes**.

## Source and safety checks

- Source-map rows identify the U500 Top100, U500 Top500, and U750 Top1000
  representative CSVs, diagnostics, finalized figures, presentation assets,
  specifications, and compatibility qualifications.
- Every finalized package used by the notebook has its manifest and scientific
  summary.
- No trainer process matching `train_evolvegcn_h`, `train_static_gcn`, or
  `run_experiment_family.py` was active during final validation.
- Existing tmux sessions were listed read-only; no tmux session was launched.
- No training command was executed.
- No graph dataset (`.pt`) was opened or deserialized.
- No checkpoint was loaded.
- No experiment artifact or finalized scientific analysis package was modified.

## Final decision

**PASS.** Notebook 10 and its offline HTML meet the requested contrast, width,
figure-readability, scientific-labeling, prediction-collapse, reproducibility,
and safety requirements.
