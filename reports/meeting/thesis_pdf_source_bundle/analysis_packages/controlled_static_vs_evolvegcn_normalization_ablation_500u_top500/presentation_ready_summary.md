# Presentation-ready summary

## Recommended supervisor-meeting sequence

1. Normalization protocol table.
2. Six-cell main result table.
3. Test MAE versus normalization.
4. Paired minmax-minus-none and zscore-minus-none MAE differences.
5. Prediction-SD ratio.
6. Exact repeated-prediction fraction.
7. Median-MAE representative true-versus-predicted panels.

## Concise conclusion

All 30 artifacts and metrics verify. Under the controlled U500 Top500 k=8
protocol, none has lower test MAE than minmax and zscore in every matched seed
for both EvolveGCN-H and Static GCN. The sample-local normalized variants also
show reduced prediction dispersion, with severe repetition in several Static
GCN runs. This supports the implementation-specific interpretation that
per-universe, per-snapshot scaling may remove useful absolute-scale information;
it does not establish that normalization is universally harmful.

Artifact completion, metric verification, normalization effects, prediction
compression, and the implementation-specific limitation should be presented as
five distinct claims.
