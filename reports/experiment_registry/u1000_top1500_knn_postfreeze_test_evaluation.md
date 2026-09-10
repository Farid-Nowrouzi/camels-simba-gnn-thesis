# U1000 Top1500 kNN Post-Freeze Test Evaluation

## Evaluation boundary

- Validation-only freeze commit: `cd3263efeb488d706c39fdfff76e12f6b6ec6601`
- Freeze commit timestamp: `2026-08-31T20:16:04Z`
- Test evaluation timestamp: `2026-08-31T20:17:11Z`
- Frozen selection: `k=8`
- The test prediction values were parsed only after the freeze commit was created and verified.
- Test results are post-freeze evaluation evidence and were not used for model selection.

All metrics below were independently recomputed from 201 rows in each
`predictions/test_predictions.csv`. The recomputed MAE, RMSE, and R2 values
agree with the corresponding `metrics.json` values (apart from harmless
floating-point display precision). Standard deviations are sample standard
deviations. Calibration is the ordinary least-squares regression
`prediction = intercept + slope * target`.

## Twelve-cell test results

| k | Seed | N | MAE | RMSE | R2 | Pred mean | Target mean | Pred SD | Target SD | SD ratio | Mean bias | Slope | Intercept | Pearson r |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 42 | 201 | 0.040479 | 0.051046 | 0.822241 | 0.302520 | 0.308620 | 0.108594 | 0.121376 | 0.894691 | -0.006100 | 0.812626 | 0.051727 | 0.908275 |
| 4 | 123 | 201 | 0.037443 | 0.049487 | 0.811937 | 0.297534 | 0.295308 | 0.101831 | 0.114400 | 0.890133 | 0.002226 | 0.802327 | 0.060601 | 0.901356 |
| 4 | 2025 | 201 | 0.037691 | 0.048353 | 0.838657 | 0.310699 | 0.305581 | 0.102273 | 0.120678 | 0.847482 | 0.005118 | 0.779345 | 0.072546 | 0.919601 |
| 6 | 42 | 201 | 0.038636 | 0.049164 | 0.835109 | 0.303031 | 0.308620 | 0.108863 | 0.121376 | 0.896907 | -0.005589 | 0.820841 | 0.049703 | 0.915191 |
| 6 | 123 | 201 | 0.036749 | 0.050119 | 0.807108 | 0.298123 | 0.295308 | 0.101774 | 0.114400 | 0.889632 | 0.002814 | 0.799581 | 0.061999 | 0.898777 |
| 6 | 2025 | 201 | 0.038482 | 0.049635 | 0.829987 | 0.307536 | 0.305581 | 0.098733 | 0.120678 | 0.818151 | 0.001955 | 0.749811 | 0.078408 | 0.916470 |
| 8 | 42 | 201 | 0.038196 | 0.048952 | 0.836528 | 0.306515 | 0.308620 | 0.112008 | 0.121376 | 0.922818 | -0.002105 | 0.844211 | 0.045975 | 0.914819 |
| 8 | 123 | 201 | 0.036829 | 0.049026 | 0.815430 | 0.300336 | 0.295308 | 0.096504 | 0.114400 | 0.843571 | 0.005027 | 0.764491 | 0.074575 | 0.906256 |
| 8 | 2025 | 201 | 0.038635 | 0.050068 | 0.827009 | 0.309125 | 0.305581 | 0.102132 | 0.120678 | 0.846318 | 0.003544 | 0.772065 | 0.073196 | 0.912264 |
| 12 | 42 | 201 | 0.038431 | 0.048731 | 0.838002 | 0.302452 | 0.308620 | 0.108701 | 0.121376 | 0.895576 | -0.006168 | 0.821327 | 0.048974 | 0.917094 |
| 12 | 123 | 201 | 0.034871 | 0.046939 | 0.830810 | 0.296244 | 0.295308 | 0.099726 | 0.114400 | 0.871731 | 0.000935 | 0.795396 | 0.061357 | 0.912433 |
| 12 | 2025 | 201 | 0.038710 | 0.049280 | 0.832406 | 0.305505 | 0.305581 | 0.101695 | 0.120678 | 0.842697 | -0.000076 | 0.771272 | 0.069818 | 0.915243 |

## Per-k test summary across seeds

| k | MAE mean +/- sample SD | RMSE mean +/- sample SD | R2 mean +/- sample SD | Mean SD ratio |
|---:|---:|---:|---:|---:|
| 4 | 0.038538 +/- 0.001686 | 0.049629 +/- 0.001352 | 0.824278 +/- 0.013476 | 0.877435 |
| 6 | 0.037956 +/- 0.001048 | 0.049639 +/- 0.000477 | 0.824068 +/- 0.014909 | 0.868230 |
| 8 | 0.037887 +/- 0.000942 | 0.049348 +/- 0.000624 | 0.826323 +/- 0.010566 | 0.870902 |
| 12 | 0.037337 +/- 0.002141 | 0.048317 +/- 0.001225 | 0.833740 +/- 0.003777 | 0.870001 |

The numerically lowest three-seed mean test MAE is at k=12, only 0.000549
below the frozen k=8 mean. This post-freeze observation does not change the
selection.

## Seed-matched candidate-minus-k8 test MAE differences

Positive values favor k=8; negative values favor the candidate.

| Candidate k | Seed 42 | Seed 123 | Seed 2025 |
|---:|---:|---:|---:|
| 4 | 0.002283 | 0.000614 | -0.000944 |
| 6 | 0.000440 | -0.000080 | -0.000152 |
| 12 | 0.000235 | -0.001959 | 0.000076 |

Test MAE rankings from best to worst are:

- seed 42: k=8, k=12, k=6, k=4;
- seed 123: k=12, k=6, k=8, k=4;
- seed 2025: k=4, k=6, k=8, k=12.

## Validation-versus-test consistency

Validation selected no robust winner: seed-level validation winners were k=8,
k=12, and k=6 for seeds 42, 123, and 2025, respectively. Test preserves the
seed-42 and seed-123 winners but changes the seed-2025 winner to k=4. The
three-seed mean ranks k=12 first on both validation and test, yet its advantage
over k=8 is small (about 0.000538 validation MAE and 0.000549 test MAE), and it
does not win consistently across matched seeds. Thus the post-freeze evidence
is qualitatively consistent with the validation classification
`SEED_DEPENDENT`, not evidence for test-driven reselection.

## Variance compression and calibration

| k | Mean bias mean +/- SD | Slope mean +/- SD | Intercept mean +/- SD | Pearson r mean +/- SD | SD ratio mean +/- SD |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.000415 +/- 0.005824 | 0.798099 +/- 0.017038 | 0.061625 +/- 0.010447 | 0.909744 +/- 0.009211 | 0.877435 +/- 0.026040 |
| 6 | -0.000273 +/- 0.004623 | 0.790078 +/- 0.036456 | 0.063370 +/- 0.014402 | 0.910146 +/- 0.009867 | 0.868230 +/- 0.043522 |
| 8 | 0.002156 +/- 0.003763 | 0.793589 +/- 0.044003 | 0.064582 +/- 0.016129 | 0.911113 +/- 0.004396 | 0.870902 +/- 0.044981 |
| 12 | -0.001770 +/- 0.003843 | 0.795998 +/- 0.025033 | 0.060050 +/- 0.010484 | 0.914923 +/- 0.002347 | 0.870001 +/- 0.026482 |

All 12 slopes are below one (0.7498--0.8442), and all prediction-to-target SD
ratios are below one (0.8182--0.9228). Across-k mean SD ratios occupy the very
narrow interval 0.8682--0.8774. Mean biases are small relative to prediction
errors, so the dominant calibration issue is range compression rather than a
uniform offset. Changing k within {4, 6, 8, 12} does not materially fix this
compression.

## Scientific interpretation

Validation evidence is the model-selection evidence: it shows small,
seed-dependent k effects and freezes k=8 to preserve the established control
and experimental continuity. Test evidence is strictly post-freeze: it shows
similarly small differences, unstable per-seed rankings, and no material
improvement in prediction dispersion or calibration as k changes.

Therefore, varying k in {4, 6, 8, 12} does not materially solve the Static GCN
problem. The test results support moving to a controlled representation change
from frozen kNN(k=8) to a periodic sparse physical-radius graph, rather than
continuing to tune k. The selected k remains 8 regardless of k=12's numerical
three-seed test mean.

## Limitations

- Only three seeds were evaluated.
- Per-seed k rankings are unstable.
- Differences among k values are small relative to their scientific import.
- This evaluation diagnoses association and calibration; it does not establish
  why the Static GCN compresses its prediction range.
