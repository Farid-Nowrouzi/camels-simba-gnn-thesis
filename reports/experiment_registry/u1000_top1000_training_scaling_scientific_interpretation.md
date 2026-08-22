# U1000 Top1000 Training Scaling: Presentation and Thesis Interpretation

## Headline findings

- **Static GCN is better overall in this completed matrix.** It has lower seed-matched MAE in 18/18 comparisons and wins the mean MAE, RMSE, and R² comparison at every training-universe count.
- At Train700, Static GCN achieves MAE 0.0397 ± 0.0031 and R² 0.815 ± 0.016; EvolveGCN-H achieves MAE 0.0580 ± 0.0033 and R² 0.595 ± 0.105.
- More training universes help both tested architectures overall, but the curves are not perfectly monotonic and gains diminish at the upper end.
- The conclusion is architecture- and setup-specific: it does not show that temporal information is inherently useless.

## Validated evidence base

This interpretation uses only the 36 completed/PASS matrix runs: two models, six training-universe counts (20, 50, 100, 200, 450, and 700), and seeds 42, 123, and 2025. Exactly one earlier invalid run is preserved in the registry but is matrix-excluded and is not analyzed. Every included run has 201 test predictions whose IDs, order, and true Ωm values were checked against its split manifest and authoritative target table. MAE, MSE, RMSE, and R² were independently recomputed and match the stored metrics.

“Training universes” is the number of universes assigned to the training split. Within each seed, validation (99 universes) and test (201 universes) remain fixed across training counts and models according to the reviewed manifests; the additional eligible universes are unused. The split is seed-specific, so different seeds have different 201-universe test sets.

## Reading the true-vs-predicted figures

The x-axis is the actual true Ωm value for each held-out test universe; moving left is not “better.” Accuracy is represented by proximity to the diagonal y=x line. Unequal point coverage toward the upper range can reflect the realized test-target distribution, while predictions that occupy a narrower vertical range than the true targets indicate model compression or regression toward the mean.

## Learning curves

Static GCN improves from mean MAE 0.0687 ± 0.0056 at Train20 to 0.0397 ± 0.0031 at Train700. EvolveGCN-H is less monotonic and remains worse at Train700. MSE, RMSE, and R² give the same overall ranking. From Train450 to Train700, paired MAE changes are -0.0025 ± 0.0010 for Static GCN and -0.0058 ± 0.0075 for EvolveGCN-H (negative means improvement). These final increments are smaller than the full Train20-to-Train700 change, supporting diminishing returns, but only three paired seeds quantify each increment.

## Static versus Evolve performance and seed stability

Static GCN has lower paired MAE in 18/18 model/count/seed comparisons and is generally more stable at larger training counts. EvolveGCN-H is especially unstable at Train20: MAE 0.5427 ± 0.4938, R² -76.120 ± 115.452, and prediction-SD ratio 6.914 ± 6.462. Sample SD reflects only three seeds, so stability claims should remain descriptive.

## Prediction spread and residual structure

At Train700, the prediction-SD ratios are 0.855 ± 0.025 for Static GCN and 0.888 ± 0.075 for EvolveGCN-H. Ratios below one indicate compressed prediction spread; ratios above one indicate excessive spread relative to the true targets. The residual definition is prediction minus true target. For the lowest / middle 60% / highest target ranges, the Train700 mean biases are respectively 0.0302 ± 0.0053, 0.0036 ± 0.0125, and -0.0502 ± 0.0091 for Static GCN, and 0.0478 ± 0.0109, 0.0183 ± 0.0143, and -0.0501 ± 0.0164 for EvolveGCN-H. Thus the models tend to overpredict low targets when low-quintile bias is positive and underpredict high targets when high-quintile bias is negative, a regression-to-the-mean pattern visible in the sorted-target and residual plots.

## Train450 versus Train700

The paired comparison preserves model, seed, validation set, and test set. It is the cleanest assessment of the final training-size increase, but the three seed pairs do not support fine-grained uncertainty claims.

## Seed-specific prediction-table constraint

The three seeds do not evaluate the same 201 universe IDs (only 12 test IDs occur in all three). Therefore a 201-row table containing three seed predictions and their per-universe mean is not identifiable from the authoritative test predictions. Creating it would require fabricating missing predictions or mixing training/validation predictions into test analysis. The package instead provides 18 exact, 201-row tables—one per training count and seed—with EvolveGCN-H and Static GCN paired on the same held-out universe IDs. Cross-seed means are used only for aggregate metrics, where that operation is scientifically valid.

## Scope of the temporal conclusion

The comparison does not show that temporal information is inherently unhelpful. The architectures differ in how they process the representation: the tested EvolveGCN-H uses all five temporal snapshots, whereas the tested Static GCN uses the final snapshot, alongside architecture-specific parameterization and optimization behavior. The results therefore support a statement about these tested implementations under this controlled protocol, not a general causal claim about temporal information. Better temporal architectures, tuning, regularization, or alternative temporal summaries could change the outcome.

## Thesis-ready conclusion

Across the completed U1000 Top1000 scaling matrix, Static GCN delivers lower error, stronger R², and generally greater seed stability than EvolveGCN-H, especially once moderate-to-large training sets are available. EvolveGCN-H is notably unstable in the smallest-data regime and its prediction spread and target-dependent residuals reveal calibration limitations. Increasing training data from 450 to 700 universes should be interpreted from the paired estimates above rather than assumed to help uniformly. Overall, the tested Static GCN architecture uses the available representation more effectively than the tested EvolveGCN-H architecture under the controlled protocol, while the experiment does not establish that temporal information itself lacks predictive value.
