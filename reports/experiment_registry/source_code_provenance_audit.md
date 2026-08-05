# Source-Code Provenance Audit

## Scope and status

`git diff` shows 883 insertions and 56 deletions across the six tracked files
requested. The changes are not recorded in experiment configs as a git commit
or dirty-state fingerprint consistently enough to reconstruct code provenance.
They must be reviewed and committed as a coherent, tested revision before any
final runs. Existing artifacts must not retroactively be claimed to use the
current working tree merely because their configs expose the same options.

## `src/models/evolvegcn_h.py`

Exact functional changes:

- replaces hard-coded functional ReLU with selectable `relu`,
  `leaky_relu`, or `elu`;
- adds masked `mean_max` graph pooling;
- adds optional concatenated summary features;
- adds selectable MLP or linear regression head;
- adjusts regressor input dimension for mean-max/summary features;
- adds shape/value validation for the new paths.

Likely reason: support the activation, head, graph-pooling, and hybrid-summary
investigations. The 500U leaky-ReLU, linear-head, mean-max and hybrid runs and
all 750U head/pooling runs depend on these capabilities. The intent is clear
and scientifically motivated.

Risks: masked max uses the dtype minimum for fully masked snapshots without an
explicit all-masked replacement; summary features change the model input and
must use train-only scaling; checkpoint compatibility depends on selected
head/pooling; the MLP head activation remains ReLU even when graph activation
changes. Review and unit/smoke-test every option before training. Commit first.

## `src/training/train_evolvegcn_h.py`

Exact functional changes:

- computes 20 temporal summary statistics per snapshot (100 for five
  snapshots), optionally returns them in batches, and fits their mean/std on
  train IDs only;
- optionally normalizes targets using train-only mean/std and denormalizes
  predictions;
- extends collate, batch movement, training, loss evaluation, and prediction
  collection for optional summary tensors;
- exposes and records activation, head type, summary-feature and
  target-normalization settings;
- persists scaler values/source in config;
- threads the new model arguments through training.

Likely reason: run hybrid graph+summary, target-normalization, activation, and
linear-head diagnostics while preserving the original five-item batch path.
Those named experiments depend on it; ordinary ReLU/MLP/no-summary/no-target-
normalization runs should follow the legacy path.

Risks: this is the largest change and affects core batch/loss code even when
features are disabled; NumPy and Torch standard-deviation conventions were
made consistent via population SD, but should be regression-tested; target
normalization must never mix normalized predictions with original targets;
saved scaler lists enlarge configs; defaults missing from old configs impede
provenance grouping. Review against a legacy deterministic smoke fixture and
all optional branches. Commit first.

## `src/models/static_gcn.py`

Exact functional changes:

- adds a dense GraphSAGE layer using neighbor mean, concatenation with the
  central node, linear projection, activation, dropout, layer norm, residual,
  and masking;
- adds `conv_type={gcn,graphsage}`;
- keeps normalized adjacency for GCN but sends raw adjacency to GraphSAGE;
- routes layer calls by convolution type.

Likely reason: enable the single-seed GCN-versus-GraphSAGE ablation
`static_graphsage_500u_top500_final_snapshot_h32_seed42`. Intent is clear.

Risks: GraphSAGE ignores isolated-node self information only in the neighbor
aggregate but retains it through concatenation; `add_self_loops` is not applied
on the GraphSAGE path; layer normalization/residual make this more than a bare
convolution substitution; only one seed was exercised. Review semantics and a
masked/isolated-node test. Commit before more GraphSAGE or Static training.

## `src/training/train_static_gcn.py`

Exact functional changes:

- adds in-memory conversion of temporal datasets to final-snapshot static
  samples via `A_list[-1]`, `Nodes_list[-1]`, and `mask_list[-1]`;
- adds `dataset_format={static,temporal_final_snapshot}`;
- adds external split-config loading and overlap/membership validation;
- threads and saves `conv_type`, dataset format, and split source;
- adds CLI options for those capabilities.

Likely reason: reuse large saved temporal graphs for matched Static comparisons
without rebuilding data and support GraphSAGE. The 500U final-snapshot,
GraphSAGE, and proposed 750U Static experiments depend on these paths.

Risks: the full 15.13-GB 750U temporal object is loaded, then a second dictionary
is built; tensors are referenced rather than cloned, but dictionary/object
overhead and deserialization can still exceed RAM. Split validation checks
membership and overlap but, unlike the baseline validator, the shown change
does not explicitly require exact coverage of every dataset ID. Temporal
lists must be nonempty and aligned, but equal list lengths are not explicitly
checked before selecting `[-1]`. Review memory ownership, exact split coverage,
and final-snapshot alignment. Commit before the proposed 750U run.

## `notebooks/visualization/01_graph_sanity_checks.ipynb`

Exact change: only kernelspec `display_name` changes from `camels-gnn` to
`camels-gnn (3.10.12.final.0)`. No cell source or output change. Likely an
automatic environment metadata rewrite. No experiments depend on it. It is
benign and intentional-looking, but should be committed separately or reverted
by the owner to avoid noisy provenance. It does not block training.

## `notebooks/visualization/02_graph_models_20u_to_200u_scaling_results.ipynb`

Exact change: the same kernelspec display-name rewrite only. No cell source or
output change. It has no effect on experiment results. Treat as environment
metadata and keep it out of the functional source commit unless deliberately
documenting the environment.

## Gate before final training

Yes: review and commit the four source files first. Minimum review checks are:
legacy-default regression behavior, all-masked mean-max pooling, optional
six-item batch handling, target denormalization, exact external split coverage,
GraphSAGE self-loop semantics, and peak RAM behavior of temporal-final-snapshot
conversion. Notebook metadata is not a blocker.
