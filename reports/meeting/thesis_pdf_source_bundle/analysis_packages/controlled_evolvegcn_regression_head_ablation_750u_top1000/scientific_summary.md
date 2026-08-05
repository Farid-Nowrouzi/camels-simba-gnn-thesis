# Scientific summary

## Objective

Test whether a direct linear head or the repository's shallow MLP better maps
the fixed EvolveGCN-H pooled representation to Omega_m.

## Experimental design

Ten U750 Top1000 runs form five exact matched-seed pairs. Dataset, encoder,
graph mean pooling, temporal mean pooling, target scale, optimization, and
training settings are fixed; only the regression head differs.

## Verification

All artifacts and 201-row prediction files verify. Ordered train/validation/
test splits match exactly within every pair. Saved and recomputed primary
metrics agree within 1e-6.

## Quantitative results

Linear achieves MAE 0.055843 ±
0.008748, RMSE
0.071136, and R²
0.627697. MLP achieves MAE
0.061404 ±
0.005142, RMSE
0.076820, and R²
0.571180. Mean paired MAE(linear−MLP) is
-0.005561 ±
0.005265; linear wins four of five seeds.

## Prediction-compression diagnostics

Mean prediction-SD ratios are 0.862457
for linear and 0.779290 for MLP.
Linear has no repeated predictions. MLP has repeated predictions in two seeds,
with mean repeated fraction
0.061692.

## Interpretation

The tested linear head offers a modest and mostly consistent improvement. This
is compatible with useful Omega_m information already being linearly
accessible in the pooled representation, but does not prove that explanation.
The head affects compression in some runs but is not a complete explanation
for remaining prediction compression.

## Limitations

The effect is comparable to between-seed variability and reverses at seed 777.
Only one shallow MLP was tested. Similar train-to-validation MAE gaps
(0.017388 linear;
0.016990 MLP) do not support
a specific MLP-overfitting claim. Deeper, residual, probabilistic, and
uncertainty-aware heads remain untested.

## Conclusion

Under the tested U750 Top1000 protocol, the simpler linear head is
descriptively preferable to the shallow MLP, while the magnitude and one
reversed seed require a protocol-specific, non-universal conclusion.
