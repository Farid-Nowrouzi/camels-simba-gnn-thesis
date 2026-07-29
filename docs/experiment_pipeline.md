# Controlled Experiment-Family Pipeline

## Purpose

An **experiment family** is a set of runs that answers one scientific question
while holding a declared protocol fixed. One field changes across groups, and
each group is evaluated with the same required independent seeds. The family
JSON is the auditable contract connecting experiment paths, training commands,
verification, aggregation, and reporting.

**Canonical** means that the protocol and exact rows have been explicitly
selected as the authoritative controlled family. It does not mean that every
historical experiment with a similar name is compatible. Historical rows may
be reused only after their raw configs and artifacts pass the canonical
checks.

The planning matrix is not live status. A row marked “planned” may now be
complete, and a row marked “existing” may have become partial. Always derive
completion from the live experiment folder and its required artifacts.

## Scientific family types

- **Universe scaling** changes the number of independent simulated universes
  while Top-N, graph construction, preprocessing, architecture, optimization,
  and seed set stay fixed.
- **Top-N scaling** fixes the universe population and changes the number of
  retained halos per graph. It is not universe scaling.
- **kNN ablation** fixes the dataset population and model protocol while
  changing the neighborhood size `k`. It tests graph-construction sensitivity,
  not sample-size scaling.

Normalization, graph pooling, temporal pooling, and regression head are other
valid grouping fields. Static GCN requires its own audited specification:
model architecture and training behavior must not be assumed to match
EvolveGCN-H.

## Pipeline architecture

The stages are independent commands:

1. A JSON family specification defines scientific settings and exact runs.
2. `status_experiment_family.py` reports live artifact state without training.
3. `run_experiment_family.py` performs preflight and, only with `--execute`,
   runs missing jobs sequentially.
4. `verify_experiment_family.py` independently checks configs, metadata,
   splits, predictions, and saved metrics.
5. `refresh_experiment_registry.sh` refreshes the master cross-family registry.
6. `build_experiment_family_results.py` extracts verified seed rows, aggregates
   by the configured grouping field, writes Markdown and plotting data, and
   creates a figure when matplotlib is available.
7. `finalize_experiment_family.sh` runs verification, registry refresh, and
   result building in fail-fast order. It never trains.
8. `launch_experiment_family_tmux.sh` dry-runs first, activates
   `envs/camels-gnn`, and launches exactly one detached sequential runner.

Shared read-only behavior lives in `scripts/experiment_pipeline/common.py`.
Scientific stages remain separate so each can be tested and rerun
independently.

## Family specification

Specifications use JSON and schema version `1.0`; no YAML dependency is
required. Important sections are:

- family identity, scientific question, model, training module, and output root;
- required artifacts;
- fixed config fields, legacy defaults, and lightweight dataset-metadata fields;
- grouping field, exact grouping values, and required seeds;
- prediction file and configured target/prediction column aliases;
- ordered CLI arguments, preserving exact text such as `0.70`;
- result filenames and figure metadata;
- one exact run definition for every grouping-value/seed pair.

Each run declares:

- `group_value` and `seed`;
- audited dataset, experiment name, and experiment path;
- `origin`: `reusable_existing`, `canonical_replacement`, or `planned_new`;
- `action`: `reuse`, `run_if_missing`, or `exclude`;
- canonical reference, exclusion reason, optional argument overrides, and notes.

`TEMPLATE.json` uses descriptive placeholders. Replace every placeholder with
audited evidence before using a new specification. Never copy unverified
experiment paths from another family.

## Artifact states and safety

The default complete set is:

- `config.json`
- `metrics.json`
- `train_log.csv`
- `predictions/test_predictions.csv`
- `checkpoints/best_model.pt`

The tools classify a row as:

- **complete**: all required files exist;
- **missing**: the experiment path does not exist;
- **partial**: the path exists but at least one required file is absent;
- **excluded**: the specification explicitly excludes it with a reason.

A missing `run_if_missing` row is runnable. A complete compatible row is
skipped. A partial folder is a hard preflight failure: the runner never
overwrites or silently resumes it. A missing row declared `reuse` is also a
failure because the specification did not authorize training it.

Preflight checks dataset paths only with filesystem metadata. It never
deserializes `.pt` files or checkpoints.

## Status and dry-run usage

Current canonical status:

```bash
python3 scripts/status_experiment_family.py \
  --repo-root /home/ml/thesis-camels \
  --spec configs/experiment_families/canonical_evolvegcn_universe_scaling.json
```

Safe runner preview:

```bash
python3 scripts/run_experiment_family.py \
  --repo-root /home/ml/thesis-camels \
  --spec configs/experiment_families/canonical_evolvegcn_universe_scaling.json \
  --dry-run
```

Without `--execute`, the runner is always a dry run. `--only-group VALUE` and
`--only-seed SEED` filter the live selection without changing the JSON.

The historical command remains valid and unchanged:

```bash
bash scripts/run_canonical_evolvegcn_scaling_missing.sh
```

The legacy shell runner remains an independent canonical-specific path. The
generic runner's tokenized commands, six replacement definitions, order, skip
set, and partial-folder refusal were tested against it; all six generated
training commands match.

Execution is deliberately explicit:

```bash
python3 scripts/run_experiment_family.py \
  --repo-root /home/ml/thesis-camels \
  --spec configs/experiment_families/canonical_evolvegcn_universe_scaling.json \
  --execute
```

Jobs run sequentially, never concurrently. Each gets one log with its exact
command, start time, finish time, and exit status. `/usr/bin/time -v` is used
when executable; otherwise execution continues safely without GNU resource
statistics.

## Independent verification

Final verification:

```bash
python3 scripts/verify_experiment_family.py \
  --repo-root /home/ml/thesis-camels \
  --spec configs/experiment_families/canonical_evolvegcn_universe_scaling.json
```

During an incomplete campaign:

```bash
python3 scripts/verify_experiment_family.py \
  --repo-root /home/ml/thesis-camels \
  --spec configs/experiment_families/canonical_evolvegcn_universe_scaling.json \
  --allow-incomplete
```

The verifier checks the specification's exact group/seed coverage, required
artifacts, configured scientific fields, lightweight dataset metadata, split
IDs and overlap, finite prediction values, and saved-versus-recomputed MAE,
RMSE, and MSE. It independently recomputes R² and Pearson correlation. Default
verification fails on any required missing, partial, incompatible, or
numerically invalid row. `--allow-incomplete` reports missing/partial rows
without claiming final validity, but incompatible completed rows still fail.

The legacy `verify_canonical_evolvegcn_scaling.py` remains available as an
independent canonical-specific verifier. It was intentionally not rewritten:
keeping it provides a useful second implementation of the final scientific
gate and avoids risking its established interface.

## Registry, results, and finalization

The registry wrapper is:

```bash
bash scripts/refresh_experiment_registry.sh
```

It invokes the existing registry builder with the repository's standard roots.
It should run only after family verification succeeds.

The results builder is:

```bash
python3 scripts/build_experiment_family_results.py \
  --repo-root /home/ml/thesis-camels \
  --spec configs/experiment_families/canonical_evolvegcn_universe_scaling.json \
  --output-dir reports/experiment_registry/canonical_scaling_results
```

It verifies first and writes nothing when the family is incomplete. When all
rows pass, it creates seed-level CSV, grouped CSV, plotting-data CSV,
scientific Markdown, and—if matplotlib is installed—a figure.

The fail-fast post-training command is:

```bash
bash scripts/finalize_experiment_family.sh \
  configs/experiment_families/canonical_evolvegcn_universe_scaling.json
```

The canonical convenience wrapper is:

```bash
bash scripts/finalize_canonical_evolvegcn_scaling.sh
```

Finalization never trains. It stops if verification fails, then refreshes the
registry, builds results, and prints the generated locations.

## Optional tmux launch

After reviewing a dry run, launch one detached sequential runner with:

```bash
bash scripts/launch_experiment_family_tmux.sh \
  --spec configs/experiment_families/canonical_evolvegcn_universe_scaling.json \
  --session canonical-scaling-run
```

The launcher performs its own dry run, refuses an existing session, verifies
`envs/camels-gnn/bin/activate`, and then creates exactly one session. Monitor
with `tmux list-panes -t canonical-scaling-run` or attach with
`tmux attach -t canonical-scaling-run`.

## Seed aggregation

The primary reported result for each grouping value is the arithmetic mean of
the independently evaluated seed-level metric. Uncertainty is the **sample
standard deviation**, with denominator `n-1`. Test examples from different
seeds are not pooled into one error calculation.

The best and worst seeds are useful diagnostics, but the best seed is not the
primary result. Selecting only the best seed is optimistic selection and hides
training/split variability.

## Canonical EvolveGCN-H worked example

`canonical_evolvegcn_universe_scaling.json` defines:

- grouping field `universes` with values 20, 50, 100, 200, and 500;
- seeds 42, 123, and 2025;
- Top100, minmax, periodic kNN k=8, five snapshots;
- EvolveGCN-H h32/L2, graph mean, temporal mean, MLP;
- batch 4, maximum 300 epochs, patience 40, LR 0.001, weight decay 0.00001;
- six canonical low-universe replacements and nine audited reusable runs.

Live artifacts—not the old planning label—show that the 20U seed-42
replacement pilot is complete. The generic runner therefore skips it and
selects only the five genuinely missing replacements.

The figure contract uses universe count on the x-axis and mean Test MAE on the
y-axis, with sample-SD error bars. It excludes 750U, Top200/Top500/Top1000, and
Static GCN.

## Adding a future family safely

1. Copy `TEMPLATE.json` to a new stable family ID.
2. State one scientific question and one grouping field.
3. Audit raw configs and dataset metadata to establish all fixed settings.
4. Populate the exact Cartesian product of grouping values and required seeds.
5. Use `reuse` only for artifact-complete, config-compatible experiments.
6. Give every excluded row a scientific reason.
7. Preserve exact CLI text and argument ordering from the audited training CLI.
8. Run the status command and review every row.
9. Run the generic runner with `--dry-run`; compare every printed command with
   the intended protocol.
10. Use `--allow-incomplete` during collection and default verification only
    at the final gate.
11. Run the finalizer only after default verification passes.

Future examples:

- **500U Top-N scaling:** set `grouping_field` to `top_n`, grouping values to
  100, 200, and 500, and fix universe count at 500.
- **kNN ablation:** set `grouping_field` to `k` and grouping values to 4, 6, 8,
  and 12; keep all other graph/model settings fixed.
- **Static GCN universe scaling:** use the Static GCN training module and its
  separately audited architecture/batch settings. Do not assume EvolveGCN-H
  parity.

These are schema examples only. No unverified experiment paths are supplied.
