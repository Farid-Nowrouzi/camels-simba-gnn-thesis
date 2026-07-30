# Canonical Static GCN Universe-Scaling Execution Plan

Prepared: 2026-07-30 UTC  
Family specification:
`configs/experiment_families/canonical_static_gcn_universe_scaling.json`

## Scope and scientific protocol

This plan defines the controlled 20U–500U Static GCN Top100 universe-scaling
family. It does not launch training. The exact 15-row product is five universe
counts (`20, 50, 100, 200, 500`) by three seeds (`42, 123, 2025`).

The fixed protocol is a native Static GCN final-snapshot graph at exactly
`a=1.0`, Top100 halos selected by raw `Mvir`, seven node features
(`log10_Mvir, X, Y, Z, VX, VY, VZ`), minmax normalization, periodic kNN with
`k=8` and box size 25, h32, two GCN layers, ReLU, dropout 0.2, masked mean
graph pooling, the established Static GCN MLP head, and self-loops. Training
uses batch size 8, at most 300 epochs, patience 40, learning rate 0.001,
weight decay 0.00001, gradient clipping 1.0, and train/validation/test ratios
0.70/0.15/0.15. No summary features or target normalization are used.

The datasets are native static datasets. Temporal-final-snapshot conversion,
Top200/Top500/Top1000, none/zscore normalization, h64 kNN datasets,
GraphSAGE, and 750U data are outside this family.

## Reuse and replacement decision

Six artifact-complete historical experiments are reused:

| Universes | Seed | Reused experiment |
|---:|---:|---|
| 20 | 123 | `static_gcn_20u_seed123_final32` |
| 50 | 123 | `static_gcn_50u_seed123_final32` |
| 100 | 123 | `static_gcn_100u_seed123_final32` |
| 200 | 42 | `static_gcn_200u_seed42_final32` |
| 200 | 123 | `static_gcn_200u_seed123_final32` |
| 200 | 2025 | `static_gcn_200u_seed2025_final32` |

Nine new experiment names are reserved with `run_if_missing`. They cannot
overwrite the historical candidates whose batch size and/or epoch budget was
incompatible. The exact names, dataset paths, and split sources are in
`canonical_static_gcn_scaling_run_matrix.csv`.

## Exact split reuse

Every replacement command passes `--split_config_path` explicitly. Each path
points to the completed EvolveGCN-H canonical run for the same universe count
and seed. The Static audit verified that these saved train, validation, and
test ID sequences exactly match the deterministic Static split assignments.
The Static trainer validates completeness, disjointness, and dataset
membership before training.

No split is regenerated silently. The replacement output config records the
source path as `split_source`, and the family verifier expects that exact
value.

## Dataset evidence

Filesystem checks confirmed that all five `.pt` paths and all five lightweight
metadata sidecars exist. The sidecars agree on:

- `dataset_type=static_final_snapshot_graphs`
- `preferred_snapshot=1.0`
- `preprocessing_version=v2_logmass_minmax_top100_periodic_knn`
- `num_nodes=100`, normalization `minmax`, graph mode `knn`
- periodic boundary flags enabled, `k=8`, box size `25.0`
- the seven required feature names
- `node_selection=top_num_nodes_by_raw_Mvir_descending`

No `.pt` file was loaded.

## Generic-pipeline compatibility

The existing generic pipeline supports this family without code changes:

- training module selection through `training_module`;
- native input through `--dataset_format static`;
- GCN selection through `--conv_type gcn`;
- h32/L2, batch size 8, mean pooling, and all optimizer settings through the
  ordered runner arguments;
- exact per-row split reuse through `argument_overrides.split_config_path`;
- deterministic sequential execution by `scripts/run_experiment_family.py`;
- refusal to overwrite partial folders and skipping of compatible completed
  folders;
- incomplete-family verification through `--allow-incomplete`.

Historical reusable configs omit several fields that predate their explicit
CLI/config representation. The family records the audited source-compatible
defaults (`static`, GCN, ReLU, established Static MLP head, self-loops, no
summary features, and no target normalization) under
`legacy_config_defaults`.

## Lightweight validation result

The authorized non-training checks passed:

- JSON parsing: pass.
- `src.training.train_static_gcn` import: pass.
- trainer `--help`: pass; it exposes `dataset_format`, `split_config_path`,
  `batch_size`, `epochs`, `hidden_dim`, `num_layers`, `graph_pooling`, and
  `conv_type`.
- generic status: 15 required, 6 complete, 9 missing, 0 partial, 0 excluded,
  and 9 runnable.
- generic runner dry run: exactly nine jobs selected and the six reusable
  rows skipped.
- command audit: all nine commands use native `static` data, batch size 8,
  300 epochs, h32, L2, mean graph pooling, and GCN; none uses GraphSAGE or
  temporal-final-snapshot conversion.
- split audit: every selected Evolve split source has train/validation/test
  ID sequences exactly equal to the audited historical Static split for the
  same universe count and seed.
- overwrite check: all nine new output paths are absent.
- generic verifier with `--allow-incomplete`: `INCOMPLETE`, with 6/15
  required rows verified and only the nine planned rows reported missing.

No dataset or checkpoint was loaded and no training was executed.

## Safety and execution

The generic runner is sequential. Before any future execution, inspect family
status and confirm the expected `6 complete / 9 missing / 0 partial / 9
runnable` result. A partial replacement directory is a hard stop.

Future execution command (not run during preparation):

```bash
python3 scripts/run_experiment_family.py \
  --spec configs/experiment_families/canonical_static_gcn_universe_scaling.json \
  --execute
```

The generic tmux launcher may wrap the same family later if separately
authorized; no launcher or training process was started while creating this
plan.
