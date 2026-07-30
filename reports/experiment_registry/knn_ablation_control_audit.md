# Controlled kNN Ablation Audit

Audit date: 2026-07-30 UTC. Repository branch:
`thesis-controlled-scaling`; inspected commit: `9f55561`.

This was a lightweight inspection. No `.pt` dataset or checkpoint was loaded,
no notebook was executed, no experiment or registry file was changed, and no
training was started.

## Executive verdict

The repository contains a scientifically controlled h64 kNN ablation at 500
universes for both EvolveGCN-H and Static GCN. The tested k values are exactly
`4, 6, 8, 12`; k=2 does not exist in the inspected experiment configs,
dataset metadata, registry records, or runner scripts.

The canonical grid contains 24 complete rows:

- 2 models;
- 4 k values;
- seeds 42, 123, and 2025.

Recommendation: **A. Use the existing h64 kNN family with no new training.**

## Candidate inventory

The candidate matrix contains 33 artifact-complete 500U Top100/minmax rows:

- 24 canonical h64 rows: 12 EvolveGCN-H and 12 Static GCN;
- 3 h32 EvolveGCN-H k=8 scaling anchors;
- 3 preferred h32 Static GCN k=8 batch-8 scaling anchors;
- 3 older h32 Static GCN k=8 batch-4 protocol alternatives.

Only the 24 h64 rows answer the kNN ablation question. The h32 rows supply
k=8 anchors but no k variation, so they cannot establish a kNN curve.

## Controlled h64 protocol

Within EvolveGCN-H, every k=4/6/8/12 row fixes:

- 500 temporal universes and five snapshots;
- Top100 halos selected by raw Mvir;
- minmax-normalized `log10_Mvir, X, Y, Z, VX, VY, VZ`;
- periodic kNN, box size 25;
- h64, 2 EvolveGCN-H layers, ReLU, dropout 0.2;
- mean graph pooling, mean temporal pooling, canonical MLP head;
- batch size 4, 300 epochs, patience 40;
- learning rate 0.001, weight decay 0.00001, gradient clipping 1.0;
- self-loops enabled;
- split ratios 0.70/0.15/0.15.

Within Static GCN, every k=4/6/8/12 row fixes:

- the same 500-universe population at native final snapshot `a=1.0`;
- the same Top100 feature, normalization, periodic-kNN, and box protocol;
- h64, 3 Static GCN layers, ReLU, dropout 0.2;
- mean graph pooling and the established Static MLP head;
- batch size 8, 300 epochs, patience 40;
- learning rate 0.001, weight decay 0.00001, gradient clipping 1.0;
- fixed trainer self-loops, layer normalization, and residual behavior;
- split ratios 0.70/0.15/0.15.

Legacy configs do not save every modern explicit field. Evolve configs save
self-loops but predate explicit activation/head fields; the audited training
source supplies ReLU and the MLP default. Static configs predate explicit
dataset-format/conv-type/self-loop fields; the runner and source show native
static GCN with self-loops. These are documented defaults, not filename
inferences.

Dataset metadata verifies that, within each model, k is the only graph-data
setting that changes. All datasets report 500 successful universes, Top100,
minmax, periodic boundary flags, `graph_mode=knn`, box size 25, identical
seven features, identical raw-Mvir node selection, the same
`CAMELS_SIMBA_500U` population root, and the same preprocessing version.

## Split validation

Every h64 config contains 350 train, 75 validation, and 75 test IDs. For all
24 rows:

- the three sets are internally unique and mutually disjoint;
- their union is exactly `LH_0` through `LH_499`;
- each seed has one ordered split signature shared by all four k values and
  both models.

Audited signatures:

| Seed | Split signature |
|---:|---|
| 42 | `0f963679cd284fca861fc2c59d88bdae8e8f1f21e2cbe1bb73bd593b49056748` |
| 123 | `853549f16ef8eb3d7f18ae850c94b13c0c8bf0e770bb99cfbffff48b03530266` |
| 2025 | `3ce48b66c11e30bec459c52ba7f4a900809dd2b45be0995b8b56aeaefc747951` |

Verdict: no split leakage and exact same-seed pairing across k and models.

## Artifact and metric validation

All 33 candidates have `config.json`, `metrics.json`, `train_log.csv`,
`predictions/test_predictions.csv`, and `checkpoints/best_model.pt`.
The 24 h64 train logs share the header
`epoch,train_mse,val_mse,best_val_mse,best_epoch,learning_rate`; their recorded
lengths are consistent with early stopping and do not exceed 300 epochs.

Prediction CSVs were used to recompute MAE, RMSE, R², Pearson correlation,
target SD, prediction SD, and prediction-SD ratio. Saved test MAE/RMSE/MSE
agree within `1e-6` for every candidate. Historical metrics files generally
do not save R² or Pearson, so those values were recomputed rather than
invented. Negative R² values were retained. One h32 Static batch-4 row has
constant predictions, hence Pearson is mathematically undefined (`nan`).

For the canonical h64 family, mean results across three seeds are:

| Model | k | Mean MAE | SD MAE | Mean RMSE | Mean R² | Mean Pearson |
|---|---:|---:|---:|---:|---:|---:|
| EvolveGCN-H | 4 | 0.097773 | 0.011783 | 0.114218 | -0.048775 | -0.021086 |
| EvolveGCN-H | 6 | 0.096895 | 0.009886 | 0.113565 | -0.038060 | -0.022979 |
| EvolveGCN-H | 8 | 0.096562 | 0.009954 | 0.113073 | -0.029003 | 0.002116 |
| EvolveGCN-H | 12 | 0.097453 | 0.011609 | 0.113686 | -0.038837 | -0.006665 |
| Static GCN | 4 | 0.097318 | 0.010580 | 0.113640 | -0.038502 | -0.117716 |
| Static GCN | 6 | 0.096765 | 0.009289 | 0.112337 | -0.016160 | -0.029337 |
| Static GCN | 8 | 0.096295 | 0.009144 | 0.112393 | -0.017001 | -0.066952 |
| Static GCN | 12 | 0.097433 | 0.010385 | 0.113094 | -0.029219 | -0.110541 |

Differences among k values are small relative to seed variability. k=8 has
the lowest mean MAE for both models, but the evidence does not support a
strong monotonic k effect.

Prediction behavior is a major caveat. Evolve rows have no exact repeated
predictions, but their prediction-SD ratios remain only about 0.066–0.125.
Several Static h64 rows contain many exact repeats: the most severe are k=4
seed42 (96.0%) and k=8 seed42 (97.3%), with prediction-SD ratios 0.0084 and
0.0034. This is strong evidence of near-constant or quantized prediction
behavior in some Static runs. It does not invalidate saved MAE comparison,
but it must be shown in any scientific report rather than presenting MAE
alone.

## h32 versus h64

### Protocol A — h64

- Expected rows: 24.
- Complete reusable rows: 24.
- Missing rows: 0.
- Incompatible rows: 0.
- Additional runs required: 0.
- Scientific value: directly answers the kNN question with four graph
  connectivities and three matched seeds for both models.

### Protocol B — h32

For a comparable k=4/6/8/12 two-model grid:

- Expected rows: 24.
- Reusable protocol cells: 6, all at k=8 (three Evolve and three preferred
  Static batch-8 anchors).
- Missing rows: 18 (both models at k=4,6,12).
- Incompatible alternatives: 3 older Static k=8 batch-4 rows.
- Additional runs required if selected: 18.
- Scientific value: a valid factorial extension, but redundant for the
  thesis kNN question because h64 already supplies a complete controlled
  family.

Only hidden dimensions 32 and 64 were found among the admitted candidates.
No other hidden dimension was present.

The main k=4/6/12 runner intentionally used h64 and omitted k=8. The separate
k=8 h64 runner explicitly repaired the compatibility gap and completed all
six missing k=8 rows. The current registry matrix and canonical notebook use
only h64 and explicitly exclude h32. Historical h32 k=8 baselines remained
nearby and created a mixing risk, but no current authoritative table mixes
h32 with h64.

## Duplicate and reproduction audit

The k=8 h64 experiments are replacements for the incompatible h32 anchors,
not duplicate independent seeds. They change hidden dimension (and Static
depth), have unique prediction-file hashes, and occupy unique experiment
folders. Across all 24 canonical h64 rows, no prediction CSV hash is
duplicated.

Multiple h32 Static experiments exist for each k=8 seed because batch-4
historical runs and batch-8 canonical scaling runs are different protocols.
They are not bitwise reproductions and must not be pooled as independent
seeds. The matrix selects the batch-8 rows only as preferred h32 anchors.

## Cross-model fairness

Matched:

- same 500-universe CAMELS-SIMBA population;
- Top100, minmax, periodic kNN, box size 25;
- k=4,6,8,12;
- h64;
- seeds 42/123/2025 and exact split IDs;
- feature definitions, node selection, optimizer settings, epoch budget,
  patience, dropout, and mean graph pooling.

Intentional differences:

- temporal five-snapshot versus native final-snapshot input;
- EvolveGCN-H versus Static GCN architecture;
- 2 Evolve layers versus 3 Static layers;
- canonical/model-specific regression heads;
- batch size 4 versus 8;
- temporal pooling applies only to EvolveGCN-H.

Verdict: **mostly controlled**. The paired curves compare complete canonical
model protocols under a matched data/graph/split design, but cannot isolate a
pure architecture or temporal-input causal effect.

## Recommendation

Choose **A: use the existing h64 kNN family with no new training**.

Do not add k=2 merely because it was suspected; it is absent and the existing
four-level grid is complete. Do not launch h32 reruns solely to align with the
newer universe-scaling family. The highest-value next action is to encode the
24 h64 rows as controlled family specifications and generate the paired
artifact-derived report, with prediction-collapse diagnostics prominent.
