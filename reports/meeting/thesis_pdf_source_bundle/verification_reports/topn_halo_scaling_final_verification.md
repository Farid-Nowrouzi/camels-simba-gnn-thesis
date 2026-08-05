# Final verification: controlled Top-N halo scaling

## Verifier result

PASS. Exactly 18 canonical experiments were verified: nine EvolveGCN-H and
nine Static GCN rows spanning Top100, Top200, Top500 and seeds 42, 123, 2025.
No duplicate experiment or prediction mapping is present.

## Exact membership and recomputed diagnostics

| model | Top-N | seed | experiment | split | MAE | R² | SD ratio | exact repeat |
|---|---|---|---|---|---|---|---|---|
| EvolveGCN-H | 100 | 42 | evolvegcn_h_500u_seed42_final32 | 0f963679cd28 | 0.086537058 | -0.014196 | 0.028083 | 0.000000 |
| EvolveGCN-H | 100 | 123 | evolvegcn_h_500u_seed123_final32 | 853549f16ef8 | 0.099244820 | -0.053006 | 0.139946 | 0.000000 |
| EvolveGCN-H | 100 | 2025 | evolvegcn_h_500u_seed2025_final32 | 3ce48b66c11e | 0.105103931 | -0.029851 | 0.020660 | 0.000000 |
| EvolveGCN-H | 200 | 42 | evolvegcn_h_500u_top200_h32_seed42 | 0f963679cd28 | 0.082909732 | 0.007177 | 0.309781 | 0.000000 |
| EvolveGCN-H | 200 | 123 | evolvegcn_h_500u_top200_h32_seed123 | 853549f16ef8 | 0.099043664 | -0.078225 | 0.456589 | 0.000000 |
| EvolveGCN-H | 200 | 2025 | evolvegcn_h_500u_top200_h32_seed2025 | 3ce48b66c11e | 0.094567859 | 0.148175 | 0.415636 | 0.000000 |
| EvolveGCN-H | 500 | 42 | evolvegcn_h_500u_top500_h32_seed42 | 0f963679cd28 | 0.081289283 | 0.048602 | 0.362423 | 0.000000 |
| EvolveGCN-H | 500 | 123 | evolvegcn_h_500u_top500_h32_seed123_minmax | 853549f16ef8 | 0.088085457 | 0.138769 | 0.561369 | 0.000000 |
| EvolveGCN-H | 500 | 2025 | evolvegcn_h_500u_top500_h32_seed2025_minmax | 3ce48b66c11e | 0.089123508 | 0.236532 | 0.558515 | 0.000000 |
| Static GCN | 100 | 42 | static_gcn_u500_top100_norm-minmax_k8_h32_l3_gpool-mean_conv-gcn_batch8_factorial_seed42 | 0f963679cd28 | 0.087339659 | -0.030030 | 0.057821 | 0.920000 |
| Static GCN | 100 | 123 | static_gcn_u500_top100_norm-minmax_k8_h32_l3_gpool-mean_conv-gcn_batch8_factorial_seed123 | 853549f16ef8 | 0.097586580 | -0.001440 | 0.011453 | 0.426667 |
| Static GCN | 100 | 2025 | static_gcn_u500_top100_norm-minmax_k8_h32_l3_gpool-mean_conv-gcn_batch8_factorial_seed2025 | 3ce48b66c11e | 0.103861143 | -0.008777 | 0.060640 | 0.000000 |
| Static GCN | 200 | 42 | static_gcn_500u_top200_h32_seed42 | 0f963679cd28 | 0.086566662 | -0.010303 | 0.000000 | 0.986667 |
| Static GCN | 200 | 123 | static_gcn_500u_top200_h32_seed123 | 853549f16ef8 | 0.098277249 | -0.028435 | 0.078549 | 0.413333 |
| Static GCN | 200 | 2025 | static_gcn_500u_top200_h32_seed2025 | 3ce48b66c11e | 0.106109919 | -0.040145 | 0.133709 | 0.000000 |
| Static GCN | 500 | 42 | static_gcn_top500_final_snapshot_same_split_h32_seed42 | 0f963679cd28 | 0.086305324 | -0.003832 | 0.000000 | 0.986667 |
| Static GCN | 500 | 123 | static_gcn_500u_top500_final_snapshot_h32_seed123_minmax | 853549f16ef8 | 0.097979526 | -0.001732 | 0.042413 | 0.400000 |
| Static GCN | 500 | 2025 | static_gcn_500u_top500_final_snapshot_h32_seed2025_minmax | 3ce48b66c11e | 0.103832561 | -0.011820 | 0.148118 | 0.000000 |

## Artifact verification

Every row has a parseable config.json, metrics.json, training CSV,
predictions/test_predictions.csv, and checkpoints/best_model.pt. Checkpoints
were checked only for existence; they were never loaded. Prediction files have
75 finite, unique, nonmissing ordered universe IDs.

## Split verification

Every split has 350 training, 75 validation, and 75 test IDs; partitions are
disjoint and cover LH_0 through LH_499 exactly. For each seed, the exact
ordered split signature agrees across Top100/Top200/Top500 and across both
models.

## Metric recomputation

MAE, RMSE, MSE, R², Pearson under the established variance policy, target and
prediction moments, dispersion ratio, exact and approximate repetition,
ranges, and residual diagnostics were recomputed. Saved MAE/RMSE/MSE agree at
absolute tolerance 1e-6; maximum absolute discrepancy is 1.3877787807814457e-17.
Negative-R², poor, compressed, and repeated-prediction rows are retained.
Undefined Pearson values, if present, remain explicit and are never zero-filled.

## Implementation and normalization

Source verification confirms raw-Mvir descending Top-N selection before
log10 feature construction and normalization; raw XYZ periodic kNN; padding
after normalization; real-node masks; padded-node exclusion; zero-real-node
rejection; and topology rebuilding at each N. Top-N sets are expected prefixes
but are not byte-proven, and equal-mass ties have no stable secondary key.

Minmax statistics are sample-local per universe, snapshot, and feature.
Changing N therefore changes local minima/maxima as well as graph size. This
is part of the intervention, not target or train/test leakage.

## Exclusions

U750 Top1000, universe-scaling/debug runs, alternative k/width/pooling/head/
normalization, hybrids, target-normalized runs, GraphSAGE, legacy Static
anchors, and duplicate reproductions are excluded from the canonical mapping.
No canonical row is duplicated.

## Final analysis registration

- Family identifier: `controlled_static_vs_evolvegcn_topn_halo_scaling_500u`.
- Specification: `configs/analysis_reports/controlled_static_vs_evolvegcn_topn_halo_scaling_500u.json`.
- Output: `reports/analysis/controlled_static_vs_evolvegcn_topn_halo_scaling_500u/`.
- Mapping: 18 artifact-complete experiments and 18 unique prediction files.
- Validator result: PASS.
- Training decision: no training or graph rebuilding is required.
- U750 Top1000: excluded because universe count and normalization differ.

## Paired Top-N findings

| model | comparison | mean_delta_mae | sd_delta_mae | larger_topn_wins |
|---|---|---|---|---|
| EvolveGCN-H | Top200-Top100 | -0.004788185 | 0.005264344 | 3/3 |
| EvolveGCN-H | Top500-Top200 | -0.006007669 | 0.004694297 | 3/3 |
| EvolveGCN-H | Top500-Top100 | -0.010795854 | 0.005375550 | 3/3 |
| Static GCN | Top200-Top100 | +0.000722149 | 0.001511132 | 1/3 |
| Static GCN | Top500-Top200 | -0.000945473 | 0.001153590 | 3/3 |
| Static GCN | Top500-Top100 | -0.000223324 | 0.000733298 | 2/3 |

## Prediction-compression findings

| model | Top-N | mean_prediction_sd_ratio | mean_exact_repeat_fraction |
|---|---|---|---|
| EvolveGCN-H | 100 | 0.062896 | 0.000000 |
| EvolveGCN-H | 200 | 0.394002 | 0.000000 |
| EvolveGCN-H | 500 | 0.494102 | 0.000000 |
| Static GCN | 100 | 0.043305 | 0.448889 |
| Static GCN | 200 | 0.070753 | 0.466667 |
| Static GCN | 500 | 0.063510 | 0.462222 |

## Computational-scaling evidence

At k=8, Top100/Top200/Top500 deterministically produce 800/1,600/4,000
directed neighbor selections and dense adjacency capacities of
10,000/40,000/250,000 entries per snapshot. Wall time, GPU/CPU peak memory,
and inference time were not recorded and are not estimated.
