# Canonical EvolveGCN-H Scaling Execution Plan

## Decision

Complete a three-seed EvolveGCN-H universe-scaling family at 20, 50,
100, 200, and 500 universes. Reuse nine compatible experiments and run only
six uniquely named compatibility replacements. This package does not execute
training.

**Hypothesis:** increasing the number of simulated universes improves
generalization for Omega_m regression when graph construction, preprocessing,
architecture, optimization, and seed set are held fixed.

Three seeds (42, 123, 2025) are sufficient for this Bachelor-thesis question:
they expose seed sensitivity, support a sample standard deviation, match the
existing 200U/500U evidence, and require only six new runs. Adding 777 and 999
would require ten more low-U runs for a modest precision gain and is not
necessary under the supervisor deadline.

## Verified canonical protocol

Values marked “metadata” come from each dataset's lightweight
`.metadata.json`; values marked “config” come from the compatible seed-123
`final32` configs and are confirmed at 200U/500U. The legacy configs omit
`activation` and `head_type`; their exact effective values are verified from
the then/current CLI/model defaults as ReLU and MLP. New commands state both
explicitly.

|Field|Canonical value|Evidence|
|---|---|---|
|dataset_path|universe-specific `temporal_{U}u_minmax/...top100_periodic_knn.pt`|config|
|universes|20, 50, 100, 200, 500|metadata|
|snapshots|5|config + metadata|
|Top-N|100|metadata `num_nodes` and dataset name|
|normalization|minmax|metadata|
|graph construction|kNN, periodic true|metadata|
|k|8|metadata|
|box size|25.0|metadata|
|node features|7: log10_Mvir, X, Y, Z, VX, VY, VZ|config + metadata|
|model|EvolveGCNHRegressor|config|
|hidden_dim / layers|32 / 2|config|
|dropout / activation|0.2 / relu|config/default verified from source|
|graph / temporal pooling|mean / mean|config|
|head|mlp|default verified from source; explicit in new commands|
|batch / epochs / patience|4 / 300 / 40|config|
|learning rate / weight decay|0.001 / 0.00001|config|
|gradient clipping|1.0|config|
|self-loops|true|config|
|train / validation / test|0.70 / 0.15 / 0.15|config|
|split generation|sort IDs numerically, copy list, `random.Random(seed).shuffle`, then integer train and validation cuts; remainder is test|training source|
|external split config|none; unsupported and unnecessary for this family|training source|
|summary features / target normalization|false / false|legacy behavior and explicit absence of flags|

Only universe count, universe-specific dataset/IDs, and seed may vary.

## Dataset availability

|U|Exact processed dataset|Bytes|Approx.|Graphs present|Rebuild|
|---:|---|---:|---:|---|---|
|20|`data/processed/temporal_20u_minmax/camels_20u_temporal_logmass_minmax_top100_periodic_knn.pt`|4,447,214|4.24 MiB|20 successful five-snapshot temporal graphs|no|
|50|`data/processed/temporal_50u_minmax/camels_50u_temporal_logmass_minmax_top100_periodic_knn.pt`|11,118,862|10.60 MiB|50 successful five-snapshot temporal graphs|no|
|100|`data/processed/temporal_100u_minmax/camels_100u_temporal_logmass_minmax_top100_periodic_knn.pt`|22,242,122|21.21 MiB|100 successful five-snapshot temporal graphs|no|

Existence, sizes, graph parameters, and successful counts were checked using
filesystem metadata and JSON only. No `.pt` file was deserialized. The 200U
and 500U canonical datasets also exist and have matching metadata.

## Split protocol and expected counts

The training module creates splits internally; no external split file is
needed or accepted by this CLI. For each U, seed 42/123/2025 applies the same
algorithm used by compatible 200U/500U runs. The seed controls a local Python
shuffle; the IDs naturally differ with U.

|U|train|validation|test|
|---:|---:|---:|---:|
|20|14|3|3|
|50|35|7|8|
|100|70|15|15|
|200|140|30|30|
|500|350|75|75|

Do not reuse a config or split from another universe count.

## Nine reusable experiments

- `evolvegcn_h_20u_seed123_final32`
- `evolvegcn_h_50u_seed123_final32`
- `evolvegcn_h_100u_seed123_final32`
- `evolvegcn_h_200u_seed42_final32`
- `evolvegcn_h_200u_seed123_final32`
- `evolvegcn_h_200u_seed2025_final32`
- `evolvegcn_h_500u_seed42_final32`
- `evolvegcn_h_500u_seed123_final32`
- `evolvegcn_h_500u_seed2025_final32`

## Six missing replacements

The historical 20U, 50U, and 100U seed-42/2025 experiments used 200 epochs.
The canonical protocol uses 300. All six are excluded automatically by the
manifest-based verifier and remain untouched.

Execution order:

1. 20U seed 42
2. 20U seed 2025
3. 50U seed 42
4. 50U seed 2025
5. 100U seed 42
6. 100U seed 2025

The exact unique names and commands are in
`canonical_evolvegcn_scaling_run_matrix.csv`. The sequential runner stops on a
partial directory and skips only a folder containing all five required
artifacts.

## Resource expectations

- Graph rebuilding: none.
- Storage: low for datasets and six experiment artifacts; logs/checkpoints are
  expected to remain modest relative to existing 500U datasets.
- RAM: low at 20U, low-to-moderate at 50U/100U; all are far below existing
  500U runs.
- Runtime: low at 20U, low-to-moderate at 50U, moderate at 100U. Runs are
  strictly sequential.

## Source-code gate

Current commit: `f260a5664049c81980ef7efa84ebfacad286152c`.

Relevant working-tree hashes:

- `src/training/train_evolvegcn_h.py`:
  `50cc32d8246dd8905d365c566cad261efeff3c47d70fcb91fe51b637e98dd53b`
- `src/models/evolvegcn_h.py`:
  `a4370a16ee9bb64b73352768a4b6298b2447a90ff53f6f20eb7dbc0fef6e5236`

The repository is dirty and both relevant source files are modified. The
current implementation supports module execution and every planned option;
import/help and syntax checks pass. The six runs use the legacy-compatible
path: ReLU, MLP, no summary features, no target normalization. Nevertheless,
review and commit the source first so every final artifact has an immutable
provenance anchor. Script generation is not blocked by the dirty state.

## Later execution and verification

After source review and environment activation:

```bash
bash scripts/run_canonical_evolvegcn_scaling_missing.sh
```

Then verify all 15 rows without loading datasets/checkpoints:

```bash
python3 scripts/verify_canonical_evolvegcn_scaling.py \
  --repo-root /home/ml/thesis-camels
```

Refresh the registry only after verification passes:

```bash
python3 scripts/build_experiment_registry.py \
  --repo-root /home/ml/thesis-camels
```

## Final thesis table and figure

Table columns:

`universes`, `seeds`, `mean Test MAE`, `standard deviation Test MAE`,
`mean Test RMSE`, `standard deviation Test RMSE`, `mean Test R²`,
`standard deviation Test R²`, `best seed`, `target standard deviation`,
`prediction standard deviation ratio`, `conclusion`.

Use sample standard deviation across the three seeds. The figure is a
five-point curve with universe count on x, mean Test MAE on y, and sample-SD
error bars at 20, 50, 100, 200, and 500. Label it
“EvolveGCN-H, Top100, minmax, periodic kNN k=8, h32/L2.” Do not include 750U
Top1000.
