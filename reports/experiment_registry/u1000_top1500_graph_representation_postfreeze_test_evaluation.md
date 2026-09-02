# U1000 Top1500 Graph-Representation Post-Freeze Test Evaluation

## Evaluation boundary

- Validation-only freeze commit: `34b9fba40ed7de512aeebecdc8c761c07d5b2f7b`
- Freeze commit timestamp: `2026-09-02T13:42:44Z`
- Test evaluation timestamp: `2026-09-02T13:45:13Z`
- Frozen representation: periodic sparse kNN(`k=8`)
- `test_metrics_used_for_selection = false`
- Test prediction values were parsed only after the freeze commit was created and verified.
- During the required pre-freeze training-log tail audit, saved test-summary fields were inadvertently visible. They were not used for selection, no test prediction values were parsed, and the validation-only result independently and consistently determines the frozen choice.

All metrics below were independently recomputed from the 201 stored rows in
each `predictions/test_predictions.csv`. Recomputed MAE, RMSE, and R2 agree
with `metrics.json` to floating-point precision. Standard deviations are
sample standard deviations. Calibration is the ordinary least-squares fit
`prediction = intercept + slope * target`.

## Six-cell test results

| Representation | Seed | N | MAE | RMSE | R2 | Pred mean | Target mean | Pred SD | Target SD | SD ratio | Bias | Slope | Intercept | Pearson r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kNN(k=8) | 42 | 201 | 0.038196 | 0.048952 | 0.836528 | 0.306515 | 0.308620 | 0.112008 | 0.121376 | 0.922818 | -0.002105 | 0.844211 | 0.045975 | 0.914819 |
| radius(r*) | 42 | 201 | 0.052257 | 0.064361 | 0.717418 | 0.296442 | 0.308620 | 0.097891 | 0.121376 | 0.806508 | -0.012178 | 0.688995 | 0.083805 | 0.854294 |
| kNN(k=8) | 123 | 201 | 0.036829 | 0.049026 | 0.815430 | 0.300336 | 0.295308 | 0.096504 | 0.114400 | 0.843571 | 0.005027 | 0.764491 | 0.074575 | 0.906256 |
| radius(r*) | 123 | 201 | 0.047746 | 0.062495 | 0.700083 | 0.298509 | 0.295308 | 0.094698 | 0.114400 | 0.827783 | 0.003200 | 0.693047 | 0.093846 | 0.837233 |
| kNN(k=8) | 2025 | 201 | 0.038635 | 0.050068 | 0.827009 | 0.309125 | 0.305581 | 0.102132 | 0.120678 | 0.846318 | 0.003544 | 0.772065 | 0.073196 | 0.912264 |
| radius(r*) | 2025 | 201 | 0.056549 | 0.072691 | 0.635352 | 0.291241 | 0.305581 | 0.089165 | 0.120678 | 0.738868 | -0.014340 | 0.597734 | 0.108585 | 0.808987 |

## Seed-matched test differences

Values are radius minus kNN. Positive MAE/RMSE and negative R2 favor kNN.

| Seed | Delta MAE | Delta RMSE | Delta R2 | Delta SD ratio | Delta slope | Ranking |
|---:|---:|---:|---:|---:|---:|---|
| 42 | +0.014061 | +0.015409 | -0.119110 | -0.116310 | -0.155217 | kNN first |
| 123 | +0.010917 | +0.013469 | -0.115347 | -0.015788 | -0.071444 | kNN first |
| 2025 | +0.017915 | +0.022624 | -0.191657 | -0.107450 | -0.174331 | kNN first |

## Per-representation test summary across seeds

| Representation | MAE mean +/- SD (range) | RMSE mean +/- SD (range) | R2 mean +/- SD (range) | SD ratio mean +/- SD | Slope mean +/- SD |
|---|---:|---:|---:|---:|---:|
| kNN(k=8) | 0.037887 +/- 0.000942 (0.036829--0.038635) | 0.049348 +/- 0.000624 (0.048952--0.050068) | 0.826323 +/- 0.010566 (0.815430--0.836528) | 0.870902 +/- 0.044981 | 0.793589 +/- 0.044003 |
| radius(r*) | 0.052184 +/- 0.004402 (0.047746--0.056549) | 0.066516 +/- 0.005429 (0.062495--0.072691) | 0.684284 +/- 0.043254 (0.635352--0.717418) | 0.791053 +/- 0.046429 | 0.659925 +/- 0.053897 |

## Validation-versus-test consistency

Validation classified the radius effect as `ROBUST DEGRADATION`: all three
matched seeds had higher MAE and RMSE and lower R2 under radius graphs. Test
repeats the same directional result for every seed and every primary metric.
The frozen kNN(k=8) choice is therefore supported by held-out behavior. It is
not changed or reconfirmed through test-driven selection; test is evaluation
evidence only.

## Prediction compression and calibration

Fixed-radius topology does not materially improve prediction dispersion. It
makes it worse in this regime:

- mean test prediction-SD/target-SD falls from 0.870902 for kNN to 0.791053 for radius;
- mean calibration slope falls from 0.793589 to 0.659925;
- every slope and SD ratio is below one, and radius is lower for both quantities in every matched seed;
- radius also raises mean MAE from 0.037887 to 0.052184 and lowers mean R2 from 0.826323 to 0.684284;
- biases vary by seed and are not the sole cause: radius has substantial negative bias for seeds 42 and 2025, but its reduced slope and dispersion persist as a separate range-compression failure.

Accuracy and calibration therefore tell the same directional story. Radius
neither trades accuracy for a wider calibrated range nor solves compression;
it degrades accuracy while generally narrowing the prediction range further.

## Scientific conclusion

Radius does not outperform kNN(k=8). Its degradation is consistent across the
three matched seeds on validation and test, and is large compared with the
observed across-seed MAE variation. Replacing fixed-neighbor connectivity with
a physically fixed scale therefore does not solve the Static GCN limitation in
this frozen U1000/Top1500/raw7/final-snapshot regime.

The controlled representation-family change is still informative: aggregate
edge-budget matching is insufficient to make kNN and radius graphs
interchangeable. Their different local degree distributions, possible
isolated-node behavior, and density sensitivity matter to this GCN, and the
radius form is less effective. Because neither the earlier k ablation nor this
family change resolves prediction compression, the remaining explanation may
involve the use of absolute vector coordinates, symmetry sensitivity, message
passing/pooling, model inductive bias, or optimization. This experiment does
not distinguish among those mechanisms.

## Artifact and provenance checks

- All three radius runs contain non-empty metrics, config, train log, best checkpoint, validation predictions, test predictions, and train predictions.
- Radius best epochs/final epochs are 17/57, 13/53, and 10/50 for seeds 42, 123, and 2025; all stopped normally after the frozen 40-epoch patience.
- No traceback, CUDA OOM, or NaN/Inf error signature was found; all logs contain the normal completion marker and saved-checkpoint path.
- Each checkpoint embeds the exact run config, its epoch equals the logged best epoch, and its best validation MSE equals the train-log minimum.
- Prediction row order equals the frozen split order for train, validation, and test in all six runs.
- Radius dataset SHA-256: `b719eed2b09de1e91e132a357c39100cae835489042639ccffb51aa91250db42`.
- Radius metadata SHA-256: `3c3d978e29ed7f86cdffaf39cb502896692eaa40c6f0e6dd44de87c2dc046891`.
- Radius completion-marker SHA-256: `05bda9e9cf9cfde7cd3619c4d696848064b24c4f63c0be7ebde43063ef18999b`.
- Canonical radius validator: `PASS`.
- Ordered splits are exactly equal between representations for every seed, with zero membership/order mismatch, overlap, or duplicate count.
- Recursive config comparison found zero unexpected scientific differences; all training and model parameters are equal.

## Symmetry readiness plan (read-only; not executed)

The current code already supplies the components needed for a separate
diagnostic: dataset loading/final-snapshot conversion and prediction collection
in `src/training/train_static_gcn.py`, deterministic periodic kNN construction
in `src/data/camels_graph_utils.py`, sparse batching in
`src/training/sparse_batch.py`, and the frozen forward pass in
`src/models/static_gcn.py`. A dedicated diagnostic script and tests are still
needed; no symmetry transformation or inference was run here.

Planned protocol:

1. Load each frozen kNN(k=8) best checkpoint, require exact config/state
   identity, call `eval()`, disable gradients, and never update weights.
2. Use a preregistered, target-independent set of 32 universe IDs shared across
   the three seed-specific evaluations for the implementation pilot, followed
   by all 201 frozen test IDs per seed if the pilot passes. Include each seed's
   full frozen test partition rather than selecting cases by prediction error.
3. Test deterministic periodic translations such as
   `(L/2,0,0)`, `(0,L/2,0)`, `(0,0,L/2)`, `(L/2,L/2,L/2)`, and a generic
   non-axis vector, applying `(position + shift) mod L`; mass and velocity are
   unchanged.
4. Test all 24 orientation-preserving signed permutation matrices of the cube
   (determinant `+1`), including identity as a control. Apply position modulo
   `L=25` and rotate velocity vectors with the same matrix; mass is unchanged.
5. For the primary model-symmetry test, reuse `edge_index`: both transformation
   families are exact isometries of the periodic cube and should preserve kNN
   connectivity. On the 32-universe pilot, independently rebuild edges with
   `build_sparse_knn_edge_index` and require exact edge-set equality; retain the
   rebuilt arm to diagnose any float/tie-boundary preprocessing effects.
6. For each checkpoint/transformation report signed and absolute prediction
   deltas, mean, median, RMS, 95th percentile, maximum, and fraction exceeding
   `atol + rtol*abs(reference)`. Establish the numerical floor with repeated
   untransformed inference, then use a preregistered provisional tolerance of
   `atol=1e-6`, `rtol=1e-5` unless the repeat-control floor requires a justified
   wider bound.
7. Record edge equality separately from prediction equality so preprocessing
   and model symmetry failures are not conflated. Translation and rotation
   results must also remain separate.

The pilot requires 3 checkpoints x 32 universes x 29 transformed conditions
(five translations plus 24 rotations, with identity included among the rotations),
or 2,784 graph forwards, plus small-subset edge rebuilding. The full 201-ID
follow-up is 17,487 graph
forwards. Reusing invariant edges keeps this an inference-scale job; rebuilding
all 1,500-node kNN graphs for every transformation would add avoidable
quadratic CPU work and should be limited to the implementation-control subset.

## Limitations

- Only three training seeds were evaluated; across-seed uncertainty is descriptive.
- The radius is one training-geometry-calibrated physical scale, not a scan over radii.
- Aggregate edge-budget matching does not match degree distributions or guarantee the absence of isolated nodes.
- Conclusions apply to the frozen Static GCN and data regime, not every graph architecture or feature representation.
- The pre-freeze saved-summary exposure weakens procedural blinding, although the recorded selection calculations and rationale use validation data only and the deterministic validation evidence is unanimous.
- Symmetry remains a plan only; no symmetry or PNA experiment was executed.
