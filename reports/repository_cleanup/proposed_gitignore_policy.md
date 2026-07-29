# Proposed Gitignore and Experiment-Artifact Policy

Inspection snapshot: branch `thesis-controlled-scaling`, HEAD
`ab5552aa9a21c8995acde0f004305ddd74602f2a`. No ignore file was changed.

## Recommendation

Use policy **B: track selected lightweight experiment evidence while ignoring
checkpoints and binary model files**.

`experiments/` is not itself ignored. This is useful: its 927 non-checkpoint
JSON/CSV files occupy only 7,985,764 bytes and preserve configurations, split
IDs, metrics, predictions, and diagnostics. The 139 checkpoint files occupy
13,850,921,078 bytes out of the directory's 13,858,906,842 bytes. Ignoring the
whole directory would discard the compact evidence; committing it wholesale
would be impractical.

Before adding the lightweight evidence, define an allowlist contract. Track
only stable, completed-run artifacts such as `config.json`, `metrics.json`,
`test_predictions.csv`, compact diagnostic JSON, and a short manifest. Do not
track partial-run state or generated plots merely because they are small.

## Exact proposed rules

Append the following rules after review; do not remove the existing general
rules:

```gitignore
# Local datasets and environments
/data/
/envs/

# Experiment binaries and training state: never commit
/experiments/**/checkpoints/
/experiments/**/*.pt
/experiments/**/*.pth
/experiments/**/*.ckpt
/experiments/**/events.out.tfevents.*
/experiments/**/wandb/

# Runtime and generated analysis products
/logs/
/outputs/
/presentation_assets/
/graph_sanity_outputs.tar.gz

# Notebook/runtime caches
**/__pycache__/
**/.ipynb_checkpoints/
*.py[cod]
```

The repository already ignores most of these through broader patterns
(`data/raw/`, `data/processed/`, `data/splits/`, `*.pt`, `checkpoints/`,
`logs/`, `outputs/`, `*.tar.gz`, `__pycache__/`, and
`.ipynb_checkpoints/`). The anchored entries make repository policy explicit
and prevent a future exception from exposing large local state.

Do **not** add `/experiments/` as a blanket rule if policy B is adopted.
Instead, add a tracked `experiments/README.md` or manifest explaining the
allowlist. If the project later decides that raw experiment directories are too
mutable, migrate to policy C by exporting the same evidence to a tracked
`reports/experiment_evidence/` directory, then ignore `/experiments/` entirely.

## Size and hosting assessment

Filesystem-name counts in the audited data, experiment, log, output,
presentation, notebook, and archive areas are:

- 105 files over 10 MiB, totaling 37,782,617,324 bytes.
- 28 files over 50 MiB, totaling 34,884,879,716 bytes.
- 27 files over 100 MiB, totaling 34,800,890,642 bytes.
- Within `experiments/` alone: 83 over 10 MiB, 18 over 50 MiB, and 18 over
  100 MiB. All 18 files over 100 MiB are checkpoints.

GitHub blocks normal Git objects over 100 MB and warns around 50 MB. Even the
roughly 40.9 MB checkpoints below those thresholds are unsuitable because they
are numerous, regenerated, and binary. No checkpoint or `.pt`, `.pth`, or
`.ckpt` file should be committed or moved into ordinary Git history.

`data/` uses about 46.09 GB of allocated space and includes processed datasets
up to 15,125,161,534 bytes. Store datasets and checkpoints on managed research
storage, with hashes and retrieval/build instructions in tracked manifests.
`outputs/` (about 101 MB), `logs/` (about 0.73 MB), Python caches, notebook
checkpoints, and the 13.59 MB tar archive are generated local state.

`presentation_assets/` contains ten PNGs totaling 1.69 MB. They are small
enough for Git, but the current set is derived from the stale notebook-07
comparison, so regenerate it after the canonical runs before considering a
curated export commit.

## Eventual clean-tree sequence

After explicit approval, the tree can become clean by:

1. Commit reviewed source, documentation, and selected lightweight evidence in
   the ordered groups in `remaining_working_tree_commit_plan.md`.
2. Commit the reviewed `.gitignore` policy and the experiment-evidence
   allowlist/manifest.
3. Restore only the accidental kernelspec-only changes in notebooks 01 and 02.
4. Remove the root-level duplicate notebook and superseded prediction
   diagnostic only after approval; retain their canonical replacements.
5. Keep datasets, environments, checkpoints, logs, caches, outputs, archives,
   and stale presentation exports ignored.
6. Finish the five canonical runs, regenerate registry/report outputs, validate
   them, and commit the minimum professor-ready set.

This inspection performed none of those cleanup actions.
