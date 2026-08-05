# Summary-Baseline Validation Audit

## Classification

**Likely valid but needs one confirmation.**

The saved result
`experiments/summary_features_500u_none_top500_matched_seed42/metrics.json`
is internally consistent and the implementation is explicitly ID-aligned and
split-safe. The remaining confirmation is external to the saved metrics:
verify from CAMELS/raw-catalog documentation or the target-generation source
that the raw halo mass/phase-space columns used by the baseline do not directly
encode, derive from, or accidentally include the target parameter. Because the
MAE is exceptionally low, that semantic check is required before the word
“validated” is used in Table 10.

## Artifact and protocol

- Dataset recorded:
  `data/processed/temporal_500u_none_top500_periodic_knn/camels_500u_temporal_logmass_none_top500_periodic_knn.pt`
  (exists; not loaded during this audit).
- Split source:
  `experiments/evolvegcn_h_500u_top500_h32_seed42_none_norm/config.json`.
- Sizes: 350 train, 75 validation, 75 test.
- Features: 100 = 20 statistics × five snapshots.
- Models: Ridge, Random Forest, Gradient Boosting; each is fitted on train
  arrays only.
- Test MAE/RMSE: Ridge .021594/.027277; Random Forest .013859/.018147;
  Gradient Boosting .012769/.016703.

## Validation checklist

|Question|Evidence|Finding|
|---|---|---|
|No target column in features?|`extract_temporal_summary` reads only node tensor and mask; `get_target` is separate.|Verified in code, assuming node-feature semantics are correct.|
|No train/test overlap?|Saved IDs have zero train/val, train/test, and val/test intersections; validator rejects overlap and duplicates.|Verified.|
|Preprocessing fit only on train?|Tree models have no preprocessing. Ridge in this script has no scaler. Model `.fit` receives `X_train,y_train` only.|Verified; no global fitted transform exists.|
|Exact split matching?|Saved `split_source`, IDs, and sizes match the Evolve seed-42 none config; validator also requires exact dataset coverage at run time.|Verified from lightweight JSON.|
|Universe-ID alignment?|Feature and target dictionaries share each `uid`; arrays are built by requested ID order.|Verified in code.|
|Target-prediction alignment?|Evaluation receives paired `X_split,y_split`; metrics are computed immediately on model output. No detached prediction CSV exists for this baseline.|Verified in code; saved aggregate metrics only.|
|Feature construction?|Per snapshot: valid-node count; mass mean/SD/min/max/median; position mean/SD; velocity mean/SD; speed mean/SD.|Verified in source.|
|Target leakage?|Target is not concatenated. Semantic leakage through a mislabeled node column cannot be excluded without raw-column provenance.|One confirmation required.|
|Dataset matching?|Path is the raw-scale, 500U, temporal, Top500, periodic-kNN dataset used by the matched graph experiment.|Verified from saved paths/configs.|
|Evaluation protocol?|One fixed seed-42 split; test is evaluated after fitting, but saved script does not perform validation-based model selection. Reporting the best test model would be post-hoc if framed as selection.|Metrics valid; Table 10 must show all three models or predeclare Gradient Boosting as the representative.|

## Important interpretive caveats

The “none” baseline exploits raw mass/position/velocity summary statistics.
Its dramatic advantage may be a genuine sufficient-statistics result rather
than software leakage; it also explains why per-graph normalization harms the
GNNs. However, selecting Gradient Boosting because it has the lowest *test*
MAE is test-set model selection. Present Ridge, Random Forest and Gradient
Boosting together, and call Gradient Boosting “best observed,” not a
validation-selected winner (validation actually slightly favors Random
Forest: .012918 versus .013164 MAE).

The minmax/zscore summary artifacts are incomplete as final baseline evidence
in the registry and must not be substituted for this matched raw-scale result.
No rerun is authorized or recommended in this audit.

## Confirmation required before Table 10

Trace the seven node columns used as mass + xyz position + xyz velocity from
the raw catalog parser and CAMELS column definitions, and document that
`Omega_m` is read only as the universe-level target. If that provenance check
passes, promote the classification to **validated**. If any feature is the
target or a deterministic copy, classify the result as invalid.
