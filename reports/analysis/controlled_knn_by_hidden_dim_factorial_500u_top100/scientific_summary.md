# Scientific Summary

All 48 factorial rows are artifact-complete and verifier-complete. All primary metrics are finite and saved-versus-recomputed checks pass.

## k and hidden-dimension findings

Descriptive best k under the tested protocol: EvolveGCN-H h32: k=12; EvolveGCN-H h64: k=8; Static GCN h32: k=6; Static GCN h64: k=8.

- EvolveGCN-H: h64 had lower MAE in 5/12 matched rows; mean h64-minus-h32 MAE was 0.000604.
- Static GCN: h64 had lower MAE in 1/12 matched rows; mean h64-minus-h32 MAE was 0.001288.

The h64-minus-h32 differences change with k and sometimes change sign, which is descriptive evidence of a k × hidden-dimension interaction. With three seeds, this is not an inferential claim.

Differences between k-cell mean MAEs are generally small relative to between-seed variability; changing k from 4 to 12 does not produce a consistent improvement at either width.

## Descriptive cross-model comparison

EvolveGCN-H had lower MAE in 10/24 matched rows and Static GCN in 14/24. This compares complete protocols and is not a pure causal test of temporal input.

## Prediction compression and repetition

24 rows met the displayed compression/repetition diagnostic (prediction-SD ratio < 0.05 or any exact repeat). The Static GCN h32/k4/seed42 row is exactly constant, so Pearson is undefined due to zero prediction variance; it remains in every applicable table and plot. Denser connectivity does not consistently resolve prediction compression.

Negative R² values and collapsed rows are retained. No p-values are reported.
