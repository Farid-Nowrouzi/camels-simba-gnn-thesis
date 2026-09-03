# U1000 Top1500 kNN(k=8) Static GCN full symmetry evaluation

## Protocol and frozen scientific inputs

This post-freeze evaluation executed the protocol in
`reports/experiment_registry/u1000_top1500_knn_k8_symmetry_protocol.json`
(SHA-256 `cf80146fa9c8af455e6b2ab5c5110edc3c24ce620f869295cabfd7f94474b665`)
at source commit `ab0bff431497c8da2a9118e1300ffb6e42da8c39`. The dataset is
CAMELS-SIMBA U1000, Top1500 halos, final snapshot, raw seven features
`[log10(Mvir), X, Y, Z, VX, VY, VZ]`, no normalization, a periodic
25 h^-1 Mpc box, and the frozen periodic sparse kNN graph with k=8. The
authoritative dataset SHA-256 is
`ff6f6a89517c0b67a96a8733ce5778dba1524df441cc23c27fbfa4e2f5cdb113`.

The three evaluated checkpoints were:

| Seed | Checkpoint SHA-256 |
|---:|---|
| 42 | `49ea8189bec30852bfa0879012144df8b7982291892b976368970c2c0926a487` |
| 123 | `dce2c8cac85ab8a3f8c3e6ae3247fcaa3cf818c9ae6364b3adf5598bd3ef50eb` |
| 2025 | `8be14a8dc8e7bb9a6ffb2f20ab5afe9df80e98740dc9db12584426d656b3809f` |

The frozen tolerance was `atol=1e-6`, `rtol=1e-5`; it was not changed.

## Artifact and workload audit

The full-run log is `logs/symmetry/full_20260902T214838Z.log`. It records a
successful pre-run audit and `FULL DIAGNOSTIC COMPLETE`, with no traceback,
numerical-control failure, or checkpoint mutation. The before/after checkpoint
digests are identical. Model audit records evaluation mode, disabled gradients,
and no parameter gradients for all three models.

The scientific workload comprised 201 held-out test universes for each of three
frozen seed models under 29 conditions: one separately retained baseline, four
periodic translations, and 24 proper cube rotations. This is 17,487 core
forwards. The prediction CSV intentionally contains only transformed conditions:
201 x 3 x 28 = 16,884 rows. All 16,884
`(seed, universe, family, transformation)` cells are unique and complete:
804 translation and 4,824 rotation rows per seed. There are no missing or
duplicate cells.

An independent recomputation from `symmetry_full_predictions.csv` reproduced
all 124 stored summary records exactly (maximum floating-point difference zero),
including every T1--T4 and R00--R23 summary and stored family/seed aggregates.
The stored rotation aggregate includes identity R03; the non-identity rotation
metrics below exclude R03 as required for scientific interpretation.

## Numerical floor

R03 is the identity rotation. For each seed, all 201 baseline-versus-R03 deltas
are exactly zero: mean absolute delta, RMS, p95, maximum, and failure fraction
are all zero. Five repeated baseline inferences per model also have exactly zero
maximum delta and zero tolerance failures. This establishes a zero-valued
observed numerical floor for this execution.

## Translation results

| Seed | N | Mean signed delta | Mean absolute delta | RMS delta | Median absolute delta | p95 absolute delta | Maximum absolute delta | Failures | Failure fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 804 | -0.000294840 | 0.013146552 | 0.018022541 | 0.009602338 | 0.037185730 | 0.097349942 | 804 | 1.000000000 |
| 123 | 804 | 0.001167452 | 0.011086360 | 0.015421124 | 0.007940263 | 0.033373262 | 0.089057982 | 804 | 1.000000000 |
| 2025 | 804 | -0.000279605 | 0.012070979 | 0.016485520 | 0.009064831 | 0.032260775 | 0.073789090 | 803 | 0.998756219 |
| All | 2,412 | 0.000197669 | 0.012101297 | 0.016677284 | 0.008828834 | 0.034784333 | 0.097349942 | 2,411 | 0.999585406 |

All four predeclared translations show nonzero effects across the three seed
models; their exact individual summaries are preserved in the machine-readable
full summary.

## Non-identity rotation results

| Seed | N | Mean signed delta | Mean absolute delta | RMS delta | Median absolute delta | p95 absolute delta | Maximum absolute delta | Failures | Failure fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 4,623 | -0.001350417 | 0.026057085 | 0.035240437 | 0.019291952 | 0.073991364 | 0.167753249 | 4,623 | 1.000000000 |
| 123 | 4,623 | 0.000793364 | 0.020305586 | 0.027134812 | 0.015327483 | 0.057451409 | 0.108550131 | 4,623 | 1.000000000 |
| 2025 | 4,623 | 0.001148363 | 0.022750214 | 0.030574296 | 0.016882300 | 0.063801390 | 0.139047325 | 4,622 | 0.999783690 |
| All | 13,869 | 0.000197103 | 0.023037628 | 0.031160734 | 0.017176956 | 0.065088689 | 0.167753249 | 13,868 | 0.999927897 |

All 23 non-identity proper rotations show nonzero effects across the three seed
models; exact R00--R23 summaries, including the R03 identity control, are in the
machine-readable full summary.

## Classification and scientific interpretation

**Translation invariance: SUBSTANTIAL SYSTEMATIC VIOLATION.** The observed
numerical floor is exactly zero, whereas the all-seed mean absolute delta is
0.01210, RMS is 0.01668, p95 is 0.03478, maximum is 0.09735, and 99.9585% of
transformed predictions fail the frozen tolerance. The conclusion is consistent
across all three independently trained seed models.

**Rotation invariance: SUBSTANTIAL SYSTEMATIC VIOLATION.** Excluding the exact
identity control, the all-seed mean absolute delta is 0.02304, RMS is 0.03116,
p95 is 0.06509, maximum is 0.16775, and 99.9928% fail the frozen tolerance.
The conclusion is consistent across all three independently trained seed models.

Thus, the frozen Static GCN is not invariant to the tested periodic translations
or proper cube rotations. The graph topology was unchanged during transformed
inference: the frozen periodic kNN edge representation was reused. Consequently,
the diagnostic demonstrates sensitivity of this trained raw-feature model under
fixed topology; it does not establish that graph topology itself is responsible.

This diagnostic also cannot prove that symmetry sensitivity causes prediction
compression. It establishes the statement “the frozen model is not invariant,”
not the distinct causal statement “this causes prediction compression.” Testing
the latter would require a controlled intervention and outcome comparison.

## Scope and limitations

This was a post-freeze inference diagnostic of three particular trained models,
not a training experiment or a universal claim about GCNs. It tests four fixed
translations and the 24-element proper cube-rotation group under the recorded
raw coordinate/velocity representation. No training occurred, no optimizer was
created, no weights changed, no checkpoint was written, the stored dataset was
not modified, and graph topology was unchanged.
