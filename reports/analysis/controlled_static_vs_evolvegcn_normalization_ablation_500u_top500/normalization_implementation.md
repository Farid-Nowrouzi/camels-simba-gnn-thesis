# Normalization implementation

All runs use the seven node features `log10(Mvir), X, Y, Z, VX, VY, VZ`.
Top500 halos are selected by raw Mvir before feature construction and scaling.

## none

Mvir remains transformed to log10(Mvir), but no subsequent per-feature scaling
is applied.

## minmax

For every universe, snapshot, and feature:

`x_scaled = (x - x_min) / (x_max - x_min)`

The implementation replaces a denominator whose absolute value is below
`1e-8` with 1.0.

## zscore

For every universe, snapshot, and feature:

`x_scaled = (x - mean) / standard_deviation`

The implementation replaces a standard deviation whose absolute value is below
`1e-8` with 1.0.

Normalization occurs after raw-Mvir Top500 selection. Periodic kNN edges use a
separate copy of raw physical XYZ, with k=8 and box size 25, so normalization
does not intentionally alter topology. Omega_m is unnormalized and summary
features are disabled.

No cross-universe scaler is fitted. Each validation or test graph uses only its
own observed node features, so no target or cross-split leakage occurs. The
scientifically important limitation is that minmax and zscore remove absolute
between-universe feature-scale differences.
