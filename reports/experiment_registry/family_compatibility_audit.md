# Family Compatibility Audit

Generated from `master_experiment_registry.csv`, `experiment_family_summary.csv`, and raw `experiments/*/config.json` / `metrics.json`. No training was run.

## Executive Finding

The historical 20U-500U scaling rows are **not one clean three-seed controlled family**. For both EvolveGCN-H and Static GCN, seed 123 uses the later `final32` protocol at 20U/50U/100U, while seeds 42 and 2025 at those same universe counts use older training settings. At 200U and 500U, all three seeds use the later `final32` protocol.

## Exact Cause of Split Historical Families

- EvolveGCN-H 20U/50U/100U: seed 42 and seed 2025 use `epochs=200`; seed 123 uses `epochs=300`.
- Static GCN 20U/50U/100U: seed 42 and seed 2025 use `batch_size=4` and `epochs=200`; seed 123 uses `batch_size=8` and `epochs=300`.
- The registry split these rows because `epochs` and `batch_size` are scientifically relevant training settings in the seed-family fingerprint.

## Field-by-Field Difference Tables

### EvolveGCN-H Top100 minmax h32, seeds 42/123/2025

|Universe|Differing field|Seed 42|Seed 123|Seed 2025|
|---:|---|---|---|---|
|20|`epochs`|200|300|200|
|50|`epochs`|200|300|200|
|100|`epochs`|200|300|200|

All other audited fields matched within each universe count. Split ratios and split-counts matched for all three seeds; split IDs differ by seed as expected.

### Static GCN Top100 minmax h32, seeds 42/123/2025

|Universe|Differing field|Seed 42|Seed 123|Seed 2025|
|---:|---|---|---|---|
|20|`batch_size`|4|8|4|
|20|`epochs`|200|300|200|
|50|`batch_size`|4|8|4|
|50|`epochs`|200|300|200|
|100|`batch_size`|4|8|4|
|100|`epochs`|200|300|200|

All other audited fields matched within each universe count. Split ratios and split-counts matched for all three seeds; split IDs differ by seed as expected.

## Canonical Scaling Family Decision

A clean existing all-universe scaling line exists only for **seed 123** across 20U, 50U, 100U, 200U, and 500U. A clean three-seed family does **not** currently exist across all five universe counts. Seeds 42 and 2025 are compatible with seed 123 at 200U and 500U, but not at 20U/50U/100U.

## Canonical-Compatible Existing Rows

See `canonical_scaling_family.csv`. Rows marked `include_in_canonical_final32_scaling=true` are compatible with the later `final32` protocol. Rows marked false are historical rows that should remain separate unless rerun under the final32 protocol.

## kNN Reporting Audit

|model|k|seeds|n|mae_mean|mae_std|rmse_mean|experiments|
|---|---|---|---|---|---|---|---|
|EvolveGCN-H|4|123,2025,42|3|0.097773|0.011783|0.114218|evolvegcn_h_500u_k4_seed123, evolvegcn_h_500u_k4_seed2025, evolvegcn_h_500u_k4_seed42|
|EvolveGCN-H|6|123,2025,42|3|0.096895|0.009886|0.113565|evolvegcn_h_500u_k6_seed123, evolvegcn_h_500u_k6_seed2025, evolvegcn_h_500u_k6_seed42|
|EvolveGCN-H|8|123,2025,42|3|0.096562|0.009954|0.113073|evolvegcn_h_500u_k8_h64_seed123, evolvegcn_h_500u_k8_h64_seed2025, evolvegcn_h_500u_k8_h64_seed42|
|EvolveGCN-H|12|123,2025,42|3|0.097453|0.011609|0.113686|evolvegcn_h_500u_k12_seed123, evolvegcn_h_500u_k12_seed2025, evolvegcn_h_500u_k12_seed42|
|Static GCN|4|123,2025,42|3|0.097318|0.010580|0.113640|static_gcn_500u_k4_seed123, static_gcn_500u_k4_seed2025, static_gcn_500u_k4_seed42|
|Static GCN|6|123,2025,42|3|0.096765|0.009289|0.112337|static_gcn_500u_k6_seed123, static_gcn_500u_k6_seed2025, static_gcn_500u_k6_seed42|
|Static GCN|8|123,2025,42|3|0.096295|0.009144|0.112393|static_gcn_500u_k8_h64_seed123, static_gcn_500u_k8_h64_seed2025, static_gcn_500u_k8_h64_seed42|
|Static GCN|12|123,2025,42|3|0.097433|0.010385|0.113094|static_gcn_500u_k12_seed123, static_gcn_500u_k12_seed2025, static_gcn_500u_k12_seed42|

The kNN grouping is scientifically correct: each k value has its own registry group. The Markdown registry display is only a presentation issue because `family_name` omits `k`, making k=4/6/8/12 rows look identical.

## 500U Top500 Minmax Seed-42 Audit

### EvolveGCN-H minmax

Seed-42 run: `evolvegcn_h_500u_top500_h32_seed42`. This run is scientifically compatible with the other four minmax seeds. It was separated because the older seed-42 config omits defaults that newer configs store explicitly.

|Differing field versus first newer seed|Seed 42 value|Newer seed value|Interpretation|
|---|---|---|---|
|`activation`|<missing>|relu|explicit default only; not a scientific difference|
|`head_type`|<missing>|mlp|explicit default only; not a scientific difference|
|`use_summary_features`|<missing>|False|explicit default only; not a scientific difference|
|`normalize_target`|<missing>|False|explicit default only; not a scientific difference|

No new seed-42 minmax run is needed for this family. The registry script should eventually normalize missing legacy defaults (`activation=relu`, `head_type=mlp`, `use_summary_features=False`, `normalize_target=False`, `conv_type=gcn`) before grouping.

### Static GCN minmax

Seed-42 run: `static_gcn_top500_final_snapshot_same_split_h32_seed42`. This run is scientifically compatible with the other four minmax seeds. It was separated because the older seed-42 config omits defaults that newer configs store explicitly.

|Differing field versus first newer seed|Seed 42 value|Newer seed value|Interpretation|
|---|---|---|---|
|`conv_type`|<missing>|gcn|explicit default only; not a scientific difference|
|`split_source`|experiments/evolvegcn_h_500u_top500_h32_seed42/config.json|experiments/evolvegcn_h_500u_top500_h32_seed123_minmax/config.json|expected different seed-specific split source|

No new seed-42 minmax run is needed for this family. The registry script should eventually normalize missing legacy defaults (`activation=relu`, `head_type=mlp`, `use_summary_features=False`, `normalize_target=False`, `conv_type=gcn`) before grouping.
