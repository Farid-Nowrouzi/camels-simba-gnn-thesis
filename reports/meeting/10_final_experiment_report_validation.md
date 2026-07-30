# Final experiment report validation

## Result

**PASS**

The final notebook is valid nbformat 4.5, executed from beginning to end, and
exported as a self-contained offline HTML report. All scientific displays come
from finalized analysis packages except the explicitly qualified baseline
context.

## Commands

Generation:

```text
envs/camels-gnn/bin/python /tmp/build_final_experiment_report.py
```

Execution:

```text
env JUPYTER_CONFIG_DIR=/tmp/final-report-jupyter-config \
  JUPYTER_DATA_DIR=/tmp/final-report-jupyter-data \
  JUPYTER_RUNTIME_DIR=/tmp/final-report-jupyter-runtime \
  IPYTHONDIR=/tmp/final-report-ipython \
  MPLCONFIGDIR=/tmp/final-report-matplotlib-cache \
  envs/camels-gnn/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 \
  notebooks/visualization/10_final_experiment_report.ipynb
```

HTML export:

```text
env JUPYTER_CONFIG_DIR=/tmp/final-report-jupyter-config \
  JUPYTER_DATA_DIR=/tmp/final-report-jupyter-data \
  JUPYTER_RUNTIME_DIR=/tmp/final-report-jupyter-runtime \
  IPYTHONDIR=/tmp/final-report-ipython \
  envs/camels-gnn/bin/jupyter nbconvert \
  --to html --embed-images \
  --HTMLExporter.mathjax_url='' \
  --HTMLExporter.require_js_url='' \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_input_tags='{"hide-input"}' \
  --output 10_final_experiment_report.html \
  --output-dir reports/meeting \
  notebooks/visualization/10_final_experiment_report.ipynb
```

## Structural checks

| Check | Result |
|---|---:|
| Notebook cell count | 70 |
| Markdown-cell count | 45 |
| Code-cell count | 25 |
| Executed code-cell count | 25 |
| Unexecuted code-cell count | 0 |
| Error-output count | 0 |
| Empty-cell count | 0 |
| Visible table count | 18 |
| Visible image count | 18 |
| Collapsible meeting-note blocks | 6 |
| Explicit experiment meeting takeaways | 6 |
| Source-map rows | 8 |

## Content and source checks

- All mandatory title, Sections 1–20, and reproducibility appendix headings:
  **PASS**.
- Seven required analysis manifests parsed: **PASS**.
- Selected figure paths registered by manifests and present: **PASS**.
- Source-map package and displayed-figure paths: **PASS**.
- Notebook output scan for `Traceback`, `FileNotFoundError`, `KeyError`,
  `ERROR`, and `WARNING`: **PASS**, none found.
- Broken HTML image references: **PASS**, none found.
- HTML external script, stylesheet, font, or image dependencies: **PASS**,
  none found.
- Embedded HTML images: **18/18**.
- Required tables fit through compact, index-free presentation styling:
  **PASS**.
- Title, scientific conclusion, next-decision, meeting-summary, and source-map
  sections present in exported HTML: **PASS**.

## Artifact sizes

| Artifact | Size |
|---|---:|
| `notebooks/visualization/10_final_experiment_report.ipynb` | 2,382,037 bytes |
| `reports/meeting/10_final_experiment_report.html` | 2,625,014 bytes |
| `reports/meeting/10_final_experiment_report_source_map.csv` | 2,459 bytes |

## Optional PDF

Not created. The existing environment provides none of `xelatex`, `pdflatex`,
`wkhtmltopdf`, or `weasyprint`; no dependency was installed.

## Safety confirmation

- No training occurred.
- No graph `.pt` dataset was loaded or rebuilt.
- No model checkpoint was loaded.
- No notebook from 01 through 09 was modified or executed.
- No experiment result or artifact was modified.
