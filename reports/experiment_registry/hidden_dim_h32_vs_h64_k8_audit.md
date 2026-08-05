# Fixed-k=8 Hidden-Dimension Audit

## Executive verdict

The repository contains all 12 requested artifact-complete rows, but the scientific verdict differs by model:

- **EvolveGCN-H:** **clean hidden-dimension ablation**. Within each seed, the h32 and h64 rows use the same dataset, two layers, batch size 4, optimization settings, architecture template, and exact ordered splits. Only `hidden_dim` changes.
- **Static GCN:** **mostly controlled capacity comparison**, not a pure hidden-dimension ablation. The canonical h32 rows use two graph layers, while the h64 kNN-family rows use three. Batch size and the other audited fields match.
- **Combined cross-model comparison:** **partly controlled** as a hidden-dimension study. The Evolve comparison is clean, but the Static width effect is confounded by depth. It must not be described as a pure 12-row width ablation without replacement runs.

No model was trained and no dataset or checkpoint was loaded for this audit.

## Candidate set

The authoritative candidate matrix contains exactly 12 canonical rows:

- EvolveGCN-H: h32 and h64 at seeds 42, 123, and 2025.
- Static GCN: canonical batch-size-8 h32 and h64 at seeds 42, 123, and 2025.

All 12 experiment folders contain `config.json`, `metrics.json`, `train_log.csv`, `predictions/test_predictions.csv`, and `checkpoints/best_model.pt`.

The three historical Static h32 alternatives `static_gcn_500u_seed{42,123,2025}_final32` are excluded. They use batch size 4 and two layers. Relative to the h64 rows, they would confound width with both depth and batch size; the canonical batch-size-8 h32 replacements remove the batch-size confound but not the depth confound.

## Field-by-field compatibility

| Field | EvolveGCN-H h32 versus h64 | Static GCN h32 versus h64 | Classification and evidence |
|---|---|---|---|
| Universe count | 500 versus 500 | 500 versus 500 | matched |
| Dataset population | CAMELS-SIMBA 500U in the same temporal dataset | CAMELS-SIMBA 500U in the same static dataset | matched within each model |
| Top-N | 100 versus 100 | 100 versus 100 | matched |
| Normalization | minmax versus minmax | minmax versus minmax | matched |
| Periodic graph flag | true versus true | true versus true | matched |
| k | 8 versus 8 | 8 versus 8 | matched |
| Box size | 25 versus 25 | 25 versus 25 | matched |
| Node features | log10(Mvir), X, Y, Z, VX, VY, VZ | same seven features | matched |
| Node ordering / selection | top 100 by raw Mvir descending | top 100 by raw Mvir descending | matched |
| Snapshot configuration | same five-snapshot temporal dataset | same exact final-snapshot dataset (`preferred_snapshot=1.0`) | matched within each model |
| `hidden_dim` | 32 versus 64 | 32 versus 64 | intended hidden-dimension difference |
| `num_layers` | 2 versus 2 | 2 versus 3 | **incompatible for a pure width ablation; Static depth confounder** |
| Dropout | 0.2 versus 0.2 | 0.2 versus 0.2 | matched |
| Activation | ReLU documented default in both | ReLU implemented in the shared Static model | compatible documented default |
| Graph pooling | mean versus mean | mean versus mean | matched |
| Temporal pooling | mean versus mean | not applicable | matched / not applicable |
| Convolution type | EvolveGCN-H in both | h32 records `gcn`; h64 inherits the documented GCN default | compatible documented default |
| Regression head | same MLP template; widths scale with `hidden_dim` | same established Static MLP template; widths scale with `hidden_dim` | matched architecture template; width scaling is an intended consequence |
| Batch size | 4 versus 4 | 8 versus 8 | matched for canonical rows |
| Epoch budget | 300 versus 300 | 300 versus 300 | matched |
| Patience | 40 versus 40 | 40 versus 40 | matched |
| Learning rate | 0.001 versus 0.001 | 0.001 versus 0.001 | matched |
| Weight decay | 0.00001 versus 0.00001 | 0.00001 versus 0.00001 | matched |
| Gradient clipping | 1.0 versus 1.0 | 1.0 versus 1.0 | matched |
| Self loops | true versus true | shared Static implementation passes true | matched / compatible documented default |
| Summary features | false documented default in both | not used by the Static training path | compatible documented behavior |
| Target normalization | false documented default in both | raw target is used by the Static training path | compatible documented behavior |
| Split ratios | 0.70/0.15/0.15 | 0.70/0.15/0.15 | matched |
| Ordered split IDs | identical by seed | identical by seed | matched |

The Static layer difference is not benign. A three-layer h64 result versus a two-layer h32 result measures a joint width-and-depth capacity change. It requires replacement runs before a pure Static hidden-dimension claim can be made.

The canonical Static batch sizes do match at 8. The excluded historical h32 alternatives use batch size 4 and are therefore less suitable.

The model-specific head templates match within each model. Internal layer widths scale with `hidden_dim`; this is intrinsic to changing the width and is not an independent head-type confounder.

## Split verification

For both hidden dimensions and both models:

- seed 42: `0f963679cd284fca861fc2c59d88bdae8e8f1f21e2cbe1bb73bd593b49056748`
- seed 123: `853549f16ef8eb3d7f18ae850c94b13c0c8bf0e770bb99cfbffff48b03530266`
- seed 2025: `3ce48b66c11e30bec459c52ba7f4a900809dd2b45be0995b8b56aeaefc747951`

Every split has 350 training, 75 validation, and 75 test IDs. The three sets are disjoint and their union is exactly `LH_0` through `LH_499`. Ordered IDs match h32 versus h64 within every model and also match Static versus Evolve for the same seed. Split verdict: **verified deterministic identity**.

## Prediction-only metric verification

All requested metrics were independently recomputed from the 12 `predictions/test_predictions.csv` files:

- all files contain 75 finite target/prediction pairs;
- universe IDs are unique and match the ordered test IDs;
- recomputed MAE agrees exactly with the saved value;
- the maximum RMSE discrepancy is `1.39e-17`, below the required `1e-6`;
- recomputed R² and Pearson values are finite;
- prediction means, sample standard deviations, prediction-SD ratios, and repeated-prediction fractions are recorded in the candidate matrix.

Poor or compressed predictions were retained. In particular, Static h64 seed 42 has a prediction-SD ratio of `0.003431` and repeated-prediction fraction of `0.973333`.

## Descriptive width differences

The paired difference is defined as `MAE_h64 - MAE_h32`; negative values favour h64.

| Model | Seed 42 | Seed 123 | Seed 2025 | Mean paired difference |
|---|---:|---:|---:|---:|
| EvolveGCN-H | -0.000210 | -0.002094 | +0.001104 | -0.000400 |
| Static GCN | -0.000605 | +0.000370 | -0.001529 | -0.000588 |

The Evolve differences are eligible for a clean width interpretation but remain small and seed-dependent. The Static values are descriptive capacity differences only because depth changes concurrently.

## Run and graph-construction decision

All 12 requested historical rows exist, so:

- a zero-training **EvolveGCN-H** hidden-dimension analysis is available now;
- a zero-training **Static capacity comparison** is available with a prominent layer caveat;
- a zero-training **clean combined hidden-dimension analysis is not available**.

The minimum repair is three Static GCN h64 runs at `num_layers=2`, batch size 8, k=8, and seeds 42/123/2025, reusing the exact existing static dataset and ordered splits. This choice preserves the canonical h32 two-layer protocol and also aligns layer count across the two model families. Rerunning three h32 rows at three layers is an alternative with the same run count, but is less compatible with the existing canonical h32 Static scaling protocol.

No graph rebuilding is required. Within each model, h32 and h64 already map to the exact same k=8 dataset; `hidden_dim` and `num_layers` are model settings rather than graph-construction settings.

## Final decision

Do not create the official combined analysis specification under the name `controlled_hidden_dim_ablation_500u_top100_k8_h32_vs_h64` until the three Static depth-matched replacements exist. If no additional training is desired, publish EvolveGCN-H as a clean width ablation and label the Static results explicitly as a two-layer-versus-three-layer capacity comparison.
