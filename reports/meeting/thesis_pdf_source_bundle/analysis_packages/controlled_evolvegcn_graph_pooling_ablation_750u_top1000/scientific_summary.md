# Scientific summary

## Objective

This analysis tests whether concatenating maximum node embeddings with the
mean-pooled representation improves EvolveGCN-H Omega_m regression.

## Experimental design

Ten completed runs form five exact seed-matched pairs. All use the same U750
Top1000 unnormalized temporal dataset, periodic kNN with k=8, hidden dimension
32, two Evolve layers, batch size 4, temporal mean pooling, and a linear head.
Only graph pooling and its required 32-versus-64-dimensional linear-head input
differ.

## Verification

Every config, metric file, training log, prediction file, and checkpoint path
was present. Checkpoints and graph datasets were not loaded. All prediction
files contain 201 finite, uniquely identified rows in exact declared test
order. Ordered splits match within every seed. Saved MAE, RMSE, and MSE agree
with independent recomputation within 1e-6.

## Quantitative results

Mean pooling achieved MAE 0.055843 ±
0.008748, RMSE
0.071136 ±
0.009807, and mean R²
0.627697. Mean_max achieved MAE
0.080973 ±
0.010152, RMSE
0.106170 ±
0.015578, and mean R²
0.165548.

The mean paired MAE difference, defined as mean_max minus mean, was
+0.025130 ± 0.011204. All five MAE and RMSE differences were
positive, and all five R² differences were negative. The paired MAE effect
exceeded the ordinary seed SD within either pooling cell.

## Prediction-dispersion diagnostics

Mean prediction-SD ratios were
0.862457 for mean and
0.863478 for mean_max.
The average values were nearly unchanged, while seed-level changes were
inconsistent. Neither method produced exact or approximate repeated
predictions. The error degradation is therefore not primarily explained by
stronger prediction collapse.

## Interpretation

Directly appending maximum node embeddings did not recover a more informative
readout under this protocol. The maxima may add noisy or unstable extreme
responses that the linear head does not use effectively.

## Limitations

This is a controlled comparison of only mean and concatenated mean-and-maximum
pooling for EvolveGCN-H at U750 Top1000. It does not establish that mean is
universally optimal, that maximum-related information is irrelevant, or that
learned attention would fail. Attention pooling remains unimplemented and
untested.

## Conclusion

Across five matched seeds, mean_max increased MAE and RMSE and reduced R² in
every pair without materially changing aggregate prediction dispersion.
Simple maximum concatenation therefore does not improve the tested graph
readout; adaptive pooling would require a separate controlled experiment.
