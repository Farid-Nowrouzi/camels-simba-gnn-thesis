# U1000 Top1500 Static GCN vs Static PNA: post-freeze test evaluation

## Architecture freeze

- Validation-only freeze commit: `b9272294aeb1e3544ecfabe0a55677a3408852bf`
- Primary selection metric: validation MAE
- Frozen validation classification: **ROBUST DEGRADATION**
- Test metrics used for architecture classification: **NO**
- Test predictions parsed before the freeze: **NO**
- Capacity control required: **false**

The classification was committed before any held-out test prediction values or
saved test metric values were parsed. The post-freeze test evaluation below does
not alter that classification.

## Validation results used for the freeze

All metrics were independently recomputed from the 99 saved validation rows.
Saved MAE, RMSE, and R2 values agree to floating-point precision.

| Architecture | Seed | MAE | RMSE | R2 | Pred mean | Target mean | Pred SD | Target SD | SD ratio | Bias | Slope | Intercept | Pearson r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static GCN | 42 | 0.036552 | 0.049703 | 0.804669 | 0.309071 | 0.311933 | 0.100566 | 0.113032 | 0.889707 | -0.002862 | 0.798448 | 0.060009 | 0.897427 |
| Static PNA | 42 | 0.036958 | 0.047235 | 0.823583 | 0.298781 | 0.311933 | 0.090311 | 0.113032 | 0.798989 | -0.013152 | 0.737821 | 0.068630 | 0.923444 |
| Static GCN | 123 | 0.041268 | 0.049830 | 0.816708 | 0.308144 | 0.302984 | 0.098136 | 0.116984 | 0.838891 | 0.005160 | 0.761205 | 0.077511 | 0.907395 |
| Static PNA | 123 | 0.043447 | 0.055275 | 0.774464 | 0.285958 | 0.302984 | 0.093541 | 0.116984 | 0.799610 | -0.017025 | 0.717619 | 0.068531 | 0.897461 |
| Static GCN | 2025 | 0.036248 | 0.045682 | 0.851369 | 0.308155 | 0.303424 | 0.106138 | 0.119096 | 0.891204 | 0.004731 | 0.823604 | 0.058254 | 0.924148 |
| Static PNA | 2025 | 0.055840 | 0.067741 | 0.673166 | 0.269839 | 0.303424 | 0.071859 | 0.119096 | 0.603370 | -0.033585 | 0.558779 | 0.100292 | 0.926096 |

## Seed-matched validation differences

Differences are Static PNA minus Static GCN. Negative MAE and RMSE differences
favor PNA; positive R2 differences favor PNA.

| Seed | Delta MAE | Delta RMSE | Delta R2 | Delta SD ratio | Delta slope | Validation-MAE winner |
|---:|---:|---:|---:|---:|---:|---|
| 42 | +0.000406 | -0.002468 | +0.018914 | -0.090719 | -0.060626 | Static GCN |
| 123 | +0.002180 | +0.005445 | -0.042243 | -0.039280 | -0.043586 | Static GCN |
| 2025 | +0.019592 | +0.022059 | -0.178203 | -0.287834 | -0.264826 | Static GCN |

## Three-seed validation summary

Values are mean +/- sample SD, followed by the seed range in brackets.

| Architecture | MAE | RMSE | R2 | SD ratio | Calibration slope |
|---|---:|---:|---:|---:|---:|
| Static GCN | 0.038023 +/- 0.002815 [0.036248, 0.041268] | 0.048405 +/- 0.002359 [0.045682, 0.049830] | 0.824248 +/- 0.024246 [0.804669, 0.851369] | 0.873267 +/- 0.029780 [0.838891, 0.891204] | 0.794419 +/- 0.031394 [0.761205, 0.823604] |
| Static PNA | 0.045415 +/- 0.009593 [0.036958, 0.055840] | 0.056751 +/- 0.010332 [0.047235, 0.067741] | 0.757071 +/- 0.076702 [0.673166, 0.823583] | 0.733990 +/- 0.113120 [0.603370, 0.799610] | 0.671406 +/- 0.098060 [0.558779, 0.737821] |

With only three seeds, these descriptive summaries are not treated as formal
population-level uncertainty estimates. PNA nevertheless has higher validation
MAE in all three paired cells; its mean RMSE is higher and its mean R2,
prediction-SD ratio, and calibration slope are lower. Seed 42 has slightly
better PNA RMSE and R2 but still worse primary-metric MAE and more compression.
The frozen classification is therefore **ROBUST DEGRADATION**.

## Post-freeze held-out test results

All metrics were independently recomputed from the 201 saved test rows. Saved
MAE, RMSE, and R2 values agree to floating-point precision (maximum absolute
difference `1.11e-16`).

| Architecture | Seed | MAE | RMSE | R2 | Pred mean | Target mean | Pred SD | Target SD | SD ratio | Bias | Slope | Intercept | Pearson r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static GCN | 42 | 0.038196 | 0.048952 | 0.836528 | 0.306515 | 0.308620 | 0.112008 | 0.121376 | 0.922818 | -0.002105 | 0.844211 | 0.045975 | 0.914819 |
| Static PNA | 42 | 0.039983 | 0.050191 | 0.828149 | 0.295828 | 0.308620 | 0.101640 | 0.121376 | 0.837398 | -0.012792 | 0.770273 | 0.058106 | 0.919842 |
| Static GCN | 123 | 0.036829 | 0.049026 | 0.815430 | 0.300336 | 0.295308 | 0.096504 | 0.114400 | 0.843571 | 0.005027 | 0.764491 | 0.074575 | 0.906256 |
| Static PNA | 123 | 0.040117 | 0.050914 | 0.800935 | 0.280577 | 0.295308 | 0.091861 | 0.114400 | 0.802987 | -0.014731 | 0.731194 | 0.064650 | 0.910592 |
| Static GCN | 2025 | 0.038635 | 0.050068 | 0.827009 | 0.309125 | 0.305581 | 0.102132 | 0.120678 | 0.846318 | 0.003544 | 0.772065 | 0.073196 | 0.912264 |
| Static PNA | 2025 | 0.054435 | 0.069167 | 0.669855 | 0.274280 | 0.305581 | 0.071965 | 0.120678 | 0.596342 | -0.031301 | 0.546546 | 0.107266 | 0.916498 |

## Seed-matched test differences

| Seed | Delta MAE | Delta RMSE | Delta R2 | Delta SD ratio | Delta slope | Test-MAE winner |
|---:|---:|---:|---:|---:|---:|---|
| 42 | +0.001787 | +0.001239 | -0.008379 | -0.085420 | -0.073938 | Static GCN |
| 123 | +0.003287 | +0.001889 | -0.014495 | -0.040584 | -0.033298 | Static GCN |
| 2025 | +0.015800 | +0.019099 | -0.157154 | -0.249976 | -0.225519 | Static GCN |

## Three-seed test summary

| Architecture | MAE | RMSE | R2 | SD ratio | Calibration slope |
|---|---:|---:|---:|---:|---:|
| Static GCN | 0.037887 +/- 0.000942 [0.036829, 0.038635] | 0.049348 +/- 0.000624 [0.048952, 0.050068] | 0.826323 +/- 0.010566 [0.815430, 0.836528] | 0.870902 +/- 0.044981 [0.843571, 0.922818] | 0.793589 +/- 0.044003 [0.764491, 0.844211] |
| Static PNA | 0.044845 +/- 0.008305 [0.039983, 0.054435] | 0.056757 +/- 0.010753 [0.050191, 0.069167] | 0.766313 +/- 0.084636 [0.669855, 0.828149] | 0.745575 +/- 0.130380 [0.596342, 0.837398] | 0.682671 +/- 0.119496 [0.546546, 0.770273] |

## Validation-versus-test consistency

The held-out test behavior **supports** the validation-only classification. On
test, Static PNA has higher MAE and RMSE and lower R2 than Static GCN in every
seed-matched comparison. The large seed-2025 degradation and the smaller but
same-direction seed-42 and seed-123 effects reproduce the validation pattern.

## Prediction compression and calibration

PNA does not reduce prediction-range compression. Its SD ratio is lower than
GCN for every seed on both validation and test. Its calibration slope is also
lower for every seed on both partitions. Mean test SD ratio falls from 0.870902
for GCN to 0.745575 for PNA, while mean test slope falls from 0.793589 to
0.682671. PNA predictions are additionally negatively biased in every seed,
with the strongest underprediction and compression for seed 2025. Pearson
correlations can remain similar or slightly higher while MAE, dispersion, and
calibration worsen; correlation alone therefore does not indicate improved
regression quality.

## Parameter-count caveat and capacity-control decision

- Static GCN trainable parameters: **5,281**
- Static PNA trainable parameters: **51,553**
- PNA/GCN parameter ratio: **9.761976898314714** (approximately 9.76x)
- Parameter-count matched: **false**
- Capacity control required: **false**

This is a controlled architecture benchmark under matched data, graph, width,
depth, pooling, head, and optimization settings. It is not a parameter-count-
matched operator ablation, and the results do not isolate PNA aggregation as a
causal mechanism. However, a substantially larger PNA did not establish a
robust advantage. Parameter matching is therefore not required merely to
support the conclusion that increased PNA complexity did not solve the static
regression limitation. No capacity-control model was implemented or trained.

## Final scientific conclusion

1. **Predictive accuracy:** Static PNA does not outperform Static GCN. GCN wins
   validation MAE in all three seeds and wins test MAE, RMSE, and R2 in all
   three seeds.
2. **Seed robustness:** The primary MAE direction is consistent across all
   seeds and is independently supported by all held-out test cells. The effect
   size varies substantially, especially because seed 2025 is much worse under
   PNA.
3. **Calibration and dispersion:** PNA worsens rather than improves prediction
   dispersion and calibration slope in every matched validation and test cell.
4. **Simple-GCN limitation hypothesis:** Replacing GCN message passing with a
   more expressive, much larger PNA while holding the stated protocol fixed
   does not solve the problem. This weakens the claim that simple GCN message
   passing is the main limiting factor, although it does not rule out every
   possible architecture or training design.
5. **Capacity interpretation:** Because PNA has approximately 9.76x more
   parameters, no performance difference can be attributed to its aggregation
   mechanism alone. The extra capacity makes the absence of a robust advantage
   scientifically informative but does not convert this into an operator-only
   causal ablation.
6. **Capacity-control follow-up:** Not required under the preregistered rule,
   because PNA is robustly worse rather than robustly better.
7. **Relation to summary baselines:** The result weakens the hypothesis that
   architecture choice alone is the main reason these GNNs underperform summary
   baselines. Representation, feature symmetries, objective/training behavior,
   and information captured by the halo graph remain plausible limitations.

**STATIC ARCHITECTURE STUDY CLOSED.** Notebook 15 status: **READY TO PREPARE**.

## Notebook 15 read-only content plan

1. Final U1000/Top1500 static regime.
2. Static GCN reference.
3. kNN ablation for k = 4, 6, 8, and 12.
4. Validation-only k = 8 freeze.
5. Post-freeze k-ablation test evaluation.
6. Radius calibration and radius-graph rationale.
7. kNN(k = 8) versus radius comparison.
8. Final graph-representation freeze.
9. Translation and rotation symmetry diagnostic.
10. Static GCN versus Static PNA.
11. Prediction-versus-true comparison.
12. Residual analysis.
13. Calibration and prediction-SD compression.
14. Final static-model comparison table.
15. Scientific conclusions.
16. Limitations.
17. Transition to the later Notebook 16 temporal study.

Notebook 15 was not created in this task. The temporal study and Notebook 16
remain **NOT YET**.

## Limitations

- Only three seeds were evaluated; across-seed means and sample SDs are
  descriptive and do not justify strong inferential claims.
- This comparison covers one frozen PNA configuration and no PNA
  hyperparameter sweep.
- Parameter counts are not matched, so operator identity and model capacity are
  not separable causal factors.
- The comparison is specific to CAMELS-SIMBA U1000, Top1500 raw7 features, the
  final snapshot, no normalization, and periodic sparse kNN(k = 8).
- Neither this result nor the prior symmetry diagnostic establishes a causal
  explanation for prediction compression or performance relative to summary
  baselines.
