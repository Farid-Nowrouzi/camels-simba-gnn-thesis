# Static GCN 20U–500U Universe-Scaling Audit

Inspection date: 2026-07-29 UTC. Repository HEAD:
`775c50f0eb6d3cc413b80b69b993002f748515fe`.

This was a lightweight, read-only audit. No `.pt` dataset or checkpoint was
loaded, no notebook or registry was regenerated, and no training was run.

## Executive verdict

The repository does **not** already contain a scientifically clean 15-row
Static GCN universe-scaling family. It does contain the foundation of one.

The strongest historically established protocol is:

- native static final-snapshot dataset at exactly `a=1.0`;
- Top100, minmax, periodic kNN `k=8`, box size 25;
- seven features: `log10_Mvir, X, Y, Z, VX, VY, VZ`;
- Static dense GCN, h32, 2 graph layers, ReLU, dropout 0.2;
- masked mean graph pooling;
- established MLP head `32 -> 32 -> 16 -> 1` with ReLU/dropout;
- self-loops enabled;
- batch size 8, 300 epochs, patience 40;
- learning rate 0.001, weight decay 0.00001, gradient clipping 1.0;
- train/validation/test ratios 0.70/0.15/0.15;
- deterministic numeric-ID sort plus `random.Random(seed).shuffle`;
- seeds 42, 123, and 2025.

Six of the required fifteen rows match this strict protocol and are reusable:

- 20U seed123: `static_gcn_20u_seed123_final32`
- 50U seed123: `static_gcn_50u_seed123_final32`
- 100U seed123: `static_gcn_100u_seed123_final32`
- 200U seed42: `static_gcn_200u_seed42_final32`
- 200U seed123: `static_gcn_200u_seed123_final32`
- 200U seed2025: `static_gcn_200u_seed2025_final32`

Nine replacements are required for a strictly controlled family:

- 20U seeds 42 and 2025: existing runs used batch 4 and 200 epochs.
- 50U seeds 42 and 2025: existing runs used batch 4 and 200 epochs.
- 100U seeds 42 and 2025: existing runs used batch 4 and 200 epochs.
- 500U seeds 42, 123, and 2025: existing runs used batch 4.

Every selected historical row is artifact-complete. The incompatibility is
training protocol, not missing artifacts, graph construction, or split
leakage.

## Search result and exclusions

The registry contains 26 broad Static GCN rows at 20U–500U that report Top100,
minmax, and `k=8`. After excluding debug runs and the explicitly out-of-scope
500U h64 kNN-ablation family, 21 legitimate historical candidates remain:

- the 15 h32 primary scaling rows selected in the candidate matrix; and
- six archived h64 rows (20U/50U/100U seed123 and all three 200U seeds).

Top200, Top500, alternative-normalization, GraphSAGE, debug, h64 kNN-ablation,
and 750U candidates were not admitted to the 15-row matrix. Directory names
were not used as the scientific authority; every selected row was checked
against its `config.json`, dataset metadata, metrics, artifact presence,
prediction header/count, and registry record.

## Candidate matrix interpretation

The exact matrix, including complete saved train/validation/test ID lists, is
`static_gcn_universe_scaling_candidate_matrix.csv`.

All fifteen selected rows have:

- `config.json`, `metrics.json`, `train_log.csv`,
  `predictions/test_predictions.csv`, and checkpoint presence;
- prediction row counts equal to saved test split sizes;
- the expected prediction columns;
- disjoint train/validation/test IDs;
- exhaustive coverage of `LH_0` through `LH_(N-1)`;
- split order exactly reproduced by the current documented deterministic
  algorithm;
- saved MAE/RMSE consistent with the registry; and
- R² available as the registry's recomputation from saved predictions.

Historical configs predate explicit fields for `dataset_format`, `activation`,
`conv_type`, self-loops, summary features, and target normalization. Their
behavior is nevertheless compatible with documented historical/source
defaults: native static data, ReLU, GCN, self-loops enabled, no summary
features, and no target normalization. These defaults should be saved
explicitly by any replacement run.

## Protocol options

### Protocol A — h32, 2 layers, batch 8, 300 epochs

- Universe counts with at least one reusable row: 20, 50, 100, 200.
- Reusable rows: 6.
- Exact-protocol incompatible/missing rows: 9.
- Replacement runs: 9.
- Advantage: largest compatible h32 historical family; all three 200U seeds
  are already clean; matches the Evolve family's hidden dimension and layer
  count; uses the established final32 Static baseline.
- Disadvantage: all three 500U h32 runs must be repeated solely to remove the
  batch-size confound.
- Top-N compatibility: its h32/L2 architecture differs from the established
  500U Static Top200/Top500 h32/L3 family, so it is appropriate for universe
  scaling but not for pooling Top-N rows into one architectural ablation.

### Protocol B — h32, 3 layers, batch 8, 300 epochs

- Reusable exact Top100 rows: 0.
- Missing exact-protocol rows: 15.
- Replacement runs: 15.
- Advantage: architectural match to the existing 500U Static Top200/Top500
  experiments.
- Disadvantage: no clean historical Top100 universe-scaling anchor; replaces
  every required row and answers no stronger universe-scaling question than
  Protocol A.
- Verdict: scientifically possible but not justified as the universe-scaling
  canonical protocol.

### Protocol C — historical h64, 3 layers, batch 8, 300 epochs

- Reusable admitted rows: 50U seed123, 100U seed123, and all three 200U seeds
  (5 rows).
- Incompatible/missing exact-protocol rows: 10.
- Replacement runs: 10.
- The archived 20U seed123 h64 run used batch 4 and is not compatible.
- The three 500U h64/k8 runs are scientifically useful kNN-ablation evidence
  but were explicitly excluded from this family; admitting them would mix
  experiment roles and still leave incomplete smaller-U seed coverage.
- Advantage: historically established older architecture.
- Disadvantage: larger model, weaker coverage, mixed archival/ablation
  provenance, and less alignment with the completed h32 Evolve family.
- Verdict: inferior to Protocol A for this thesis question.

Two additional partial historical protocols exist but do not form clean
families: h32/L2/batch4/200 epochs covers six seed42/2025 rows at
20U/50U/100U, while h32/L2/batch4/300 epochs covers only the three 500U rows.

## Split fairness

For every one of the 15 selected candidates:

- ratios are exactly 0.70/0.15/0.15;
- IDs are unique within each split and disjoint across splits;
- the union covers every expected ID exactly once;
- split counts are 14/3/3, 35/7/8, 70/15/15, 140/30/30, and 350/75/75 for
  20U, 50U, 100U, 200U, and 500U respectively;
- the saved order equals numeric `LH_i` sorting followed by
  `random.Random(seed).shuffle`;
- each Static split is exactly identical to the completed Evolve split for the
  same universe count and seed.

There is no observed split leakage. Split IDs are not required to be identical
between different universe-count populations. Indeed, changing population
size changes the deterministic shuffle. The controlled requirement is the
same method, ratios, and seed semantics.

The 20U and 50U static metadata identify their raw population as prefixes of
the 100U source (`LH_0...`), while 200U and 500U have their own population
roots. This is transparent in metadata and does not invalidate sample-size
scaling, but it should be stated when interpreting statistical independence
between adjacent scale points.

## Model/training compatibility

All selected h32 candidates use two graph layers, dropout 0.2, mean pooling,
patience 40, learning rate 0.001, weight decay 0.00001, gradient clipping 1.0,
and 0.70/0.15/0.15 splits. Current source documents ReLU, dense GCN,
self-loops, layer normalization/residual blocks, and an MLP regression head.

The only strict Protocol-A conflicts are:

| Universe | Seeds | Existing values | Required values |
|---:|---|---|---|
| 20 | 42, 2025 | batch 4, 200 epochs | batch 8, 300 epochs |
| 50 | 42, 2025 | batch 4, 200 epochs | batch 8, 300 epochs |
| 100 | 42, 2025 | batch 4, 200 epochs | batch 8, 300 epochs |
| 500 | 42, 123, 2025 | batch 4, 300 epochs | batch 8, 300 epochs |

There is no selected Top200/Top500, none/zscore, GraphSAGE, debug, h64, or
750U contamination.

## Static versus completed EvolveGCN-H

Matched exactly:

- universe counts 20/50/100/200/500;
- Top100, minmax, periodic kNN `k=8`, box size 25;
- seven feature definitions and raw-mass Top-N selection;
- masking/padding and graph-building implementation;
- seeds 42/123/2025;
- all 15 exact split-ID assignments;
- learning rate, weight decay, epochs, patience, dropout, gradient clipping,
  graph pooling, and train/validation/test ratios under recommended Protocol A;
- hidden dimension 32 and two model layers.

Intentionally different:

- Static uses one native graph at `a=1.0`; Evolve uses five ordered snapshots.
- Static uses dense GCN layers; Evolve uses recurrently evolving graph
  convolutions.
- Static's established batch size is 8; Evolve's is 4.
- The regression-head implementations are model-specific even though both are
  MLP-style.

Because batch size and head implementation are not identical, the prospective
comparison is classified **mostly controlled**, not “fully matched except
architecture/temporal input.” It is nevertheless a strong and interpretable
model-family comparison if Protocol A is completed and the model-specific
batch/head choices are disclosed.

## 750U clarification

- **750U Top100/minmax Static GCN:** no dataset metadata, config, registry row,
  or completed run was found.
- **750U Top1000/none Static GCN:** no Static run was found. A temporal
  Top1000/none dataset exists, but only Evolve experiments use it.
- **Seed completeness:** neither Static variant has any completed seed.
- **New training:** required for either variant.
- **Scientific placement:** a future 750U Top100/minmax Static family would be
  an optional extension of universe scaling. A Static final-snapshot model
  derived from the 750U Top1000/none temporal dataset would belong to the
  matched 750U benchmark, not the 20U–500U Top100/minmax family.

## Decision and recommendation

Recommendation: **A. Complete a clean three-seed Static GCN universe-scaling
family using Protocol A.**

Nine additional Static runs are required. They reuse existing small native
static datasets and deterministic split IDs, so their cost is low at
20U–100U and moderate at 500U. The scientific value is high now that the
five-scale Evolve family is complete: the Static curve supplies the
architecture/temporal baseline needed to interpret whether scale or temporal
modeling drives performance.

If compute or deadline constraints prohibit all nine runs, a defensible
fallback is to run the six 20U–100U replacements and waive the 500U batch-size
difference with an explicit caveat. That produces a useful but not strictly
controlled table. Simply pooling the current 15 historical rows without the
caveat is not valid.

Highest-value next action: freeze a Static family specification for Protocol A
that reuses the exact saved splits, then dry-run it. This audit does not create
that specification and does not authorize training.
