# Controlled regression-head ablation audit

## Executive summary

Phase B is scientifically allowed. A complete five-seed EvolveGCN-H U750
Top1000 family isolates `linear` versus `mlp` regression heads while holding
the dataset, ordered splits, encoder, graph pooling, temporal pooling, target
scale, optimizer, and training protocol fixed. All ten canonical rows are
artifact-complete, their prediction rows and ordered splits verify, and saved
primary metrics agree with recomputation within `1e-06`.

The linear head lowers MAE in four of five matched seeds. The paired convention
`linear - mlp` gives mean ΔMAE -0.005561 ±
0.005265. This is a modest, mostly consistent improvement,
not a universal result. No training, replacement, or graph rebuilding is
required.

## Search and candidate definition

All 176 parseable experiment configs were inspected. The matrix retains all
38 runs on the U750 Top1000 and U500 Top500 datasets where multiple head or
readout variants were actually exercised, including controlled candidates,
anchors, duplicates, target-normalized runs, summary hybrids, pooling
alternatives, and temporal-pooling alternatives.

| Compatibility class | Rows |
|---|---:|
| `canonical_head_compatible` | 10 |
| `compatible_anchor_only` | 14 |
| `duplicate_reproduction` | 1 |
| `graph_pooling_incompatible` | 5 |
| `hybrid_incompatible` | 2 |
| `target_normalization_incompatible` | 1 |
| `temporal_pooling_incompatible` | 5 |

All 38 candidates are artifact-complete. Maximum saved/recomputed primary
metric discrepancy is `8.3266726846886741e-17`.

## Implemented and trained heads

EvolveGCN-H accepts exactly `mlp` and `linear`; `mlp` is the model and CLI
default and is used for historical configs lacking `head_type`. No aliases are
accepted. The MLP is `Linear(D, H) → ReLU → Dropout(p) → Linear(H, 1)`;
the linear head is `Linear(D, 1)`. Both use biases, identity output activation,
PyTorch Linear default initialization, and receive the same representation
after graph and temporal pooling. All parameters are passed to AdamW.

At canonical `D=H=32`, linear has 33 trainable parameters and MLP has 1,089.
Static GCN uses a separate fixed three-linear-layer MLP readout
`32 → 32 → 16 → 1` with ReLU/dropout and 1,601 parameters; it exposes no
linear-head switch and is not mixed into the canonical family.

Implemented/trained evidence includes Evolve linear, Evolve MLP, and the Static
MLP readout. No residual, uncertainty-aware, bounded-output, or deeper
configurable Evolve head is implemented or trained.

## Canonical protocol and membership

- Model: EvolveGCN-H; temporal sequence input.
- Dataset: canonical U750 LH subset; Top1000; five snapshots.
- Features: `log10(Mvir), X, Y, Z, VX, VY, VZ`; normalization `none`.
- Periodic kNN: k=8, box size 25; hidden dimension 32; two layers.
- Graph pooling: mean; temporal pooling: mean.
- Batch size 4; MSE loss; AdamW; learning rate 0.001; weight decay 1e-5.
- Dropout 0.2; encoder activation ReLU; target unnormalized; no summaries.
- Seeds: 42, 123, 777, 999, 2025; split sizes 450/99/201.

- `evolvegcn_h_750u_top1000_h32_seed42_none_linear_head`
- `evolvegcn_h_750u_top1000_h32_seed123_none_linear_head`
- `evolvegcn_h_750u_top1000_h32_seed777_none_linear_head`
- `evolvegcn_h_750u_top1000_h32_seed999_none_linear_head`
- `evolvegcn_h_750u_top1000_h32_seed2025_none_linear_head`
- `evolvegcn_h_750u_top1000_h32_seed42_none`
- `evolvegcn_h_750u_top1000_h32_seed123_none`
- `evolvegcn_h_750u_top1000_h32_seed777_none`
- `evolvegcn_h_750u_top1000_h32_seed999_none`
- `evolvegcn_h_750u_top1000_h32_seed2025_none`

## Split, artifact, and prediction verification

Every canonical config, metrics JSON, CSV training log, prediction CSV, and
checkpoint path exists. Checkpoints were not opened. Every prediction file has
201 finite rows, unique IDs, exact ordered test IDs, and no missing IDs. Each
seed's linear and MLP runs share an exact ordered split signature. Each split
is disjoint and covers `LH_0` through `LH_749` exactly once.

The audit recomputed MAE, RMSE, MSE, R², Pearson where defined, target and
prediction means and sample SDs, prediction-SD ratio, exact and approximate
repeat fractions at tolerance `1e-12`, prediction range, residual mean/SD,
and maximum absolute residual. Undefined Pearson remains undefined.

## Controlled paired result

Convention: `difference = linear - mlp`. Negative MAE/RMSE favours linear;
positive R² favours linear.

| Seed | ΔMAE | ΔRMSE | ΔR² | Δprediction-SD ratio |
|---:|---:|---:|---:|---:|
| 42 | -0.011466 | -0.011725 | +0.108442 | +0.134420 |
| 123 | -0.002790 | -0.003252 | +0.034241 | +0.033801 |
| 777 | +0.000589 | +0.004947 | -0.059465 | +0.117133 |
| 999 | -0.010671 | -0.010934 | +0.115159 | +0.181659 |
| 2025 | -0.003466 | -0.007456 | +0.084208 | -0.051181 |

- Mean ΔMAE: -0.005561; SD 0.005265;
  four seeds favour linear and one favours MLP.
- Mean ΔRMSE: -0.005684; SD 0.006821;
  four seeds favour linear and one favours MLP.
- Mean ΔR²: +0.056517; SD 0.072197;
  four seeds favour linear and one favours MLP.
- Mean Δprediction-SD ratio: +0.083167 ±
  0.092151.

Linear aggregate MAE/RMSE/R² are 0.055843,
0.071136, and 0.627697.
MLP values are 0.061404,
0.076820, and 0.571180.

The effect is comparable to ordinary between-seed MAE variation, so evidence
supports a modest advantage rather than a decisive architectural law.

## Prediction compression and overfitting

Mean prediction-SD ratio is 0.862457 for
linear and 0.779290 for MLP. Linear improves
dispersion in four pairs, but the paired dispersion effect is variable. Linear
has no repeated predictions; MLP has repeated values in two seeds, with mean
exact-repeat fraction 0.061692.

Training and validation metrics are retained in the candidate matrix. Both
heads show train-to-validation gaps, but the existing evidence does not
establish that the MLP specifically overfits: test performance alone is not
used as an overfitting diagnosis.

## Duplicates and alternative protocols

One exact duplicate prediction pair exists:
`evolvegcn_h_750u_top1000_h32_seed42_none` and
`evolvegcn_h_750u_top1000_h32_seed42_none_mlp_relu_repeat`
(`e38c1d1e21829d266cb44bba53a06e558eae456594e4ec43d62a21374ebd8a34`).
The reproduction is retained but not counted independently.

Five temporal-last linear runs form a temporal-pooling ablation, and five
mean_max linear runs form the completed graph-pooling ablation. U500 includes
only a single-seed head pilot plus activation, target-normalized, summary
hybrid, Static GCN, and GraphSAGE alternatives. None is mixed into the primary
head comparison.

## Decision

- Selected family: `u750_top1000_mean_graph_mean_temporal_linear_vs_mlp`.
- Canonical reusable rows: 10.
- Missing, partial, and replacement rows: 0.
- Graph rebuild: not required.
- Training: not required.
- Phase B analysis identifier:
  `controlled_evolvegcn_regression_head_ablation_750u_top1000`.

## Caveats and repository state

The result applies only to the tested two-layer MLP versus direct linear head.
It does not resolve deeper, residual, attention-based, probabilistic, or
uncertainty-aware heads. The linear advantage does not prove that the pooled
signal is intrinsically linear.

At preflight the branch was `thesis-controlled-scaling`, `git diff --check`
passed, no trainer matched the requested process pattern, and only pre-existing
untracked experiment/registry work was present.
