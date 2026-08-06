# U1000 Top1000 Training Scaling Scientific Interpretation

## Validated evidence base

This interpretation uses all 36 completed, validated runs: two models, six training-set sizes, and three seeds per cell. Every run has 201 ordered test predictions whose IDs and targets were checked against its stored split manifest and the authoritative target table. MAE, MSE, RMSE, and R² were independently recomputed from those predictions and matched the reported metrics.

## Learning curves

Static GCN improves substantially as the training set grows: its mean MAE changes from 0.0687 ± 0.0056 at Train20 to 0.0397 ± 0.0031 at Train700. EvolveGCN-H is much less monotonic and remains worse at Train700, where its mean MAE is 0.0580 ± 0.0033. The MSE, RMSE, and R² curves give the same broad ranking while exposing particularly large errors in unstable EvolveGCN-H runs.

## Static versus Evolve performance and seed stability

Static GCN has lower paired MAE in 18 of 18 model/count/seed comparisons. At Train700, Static GCN MAE is 0.0397 ± 0.0031, compared with 0.0580 ± 0.0033 for EvolveGCN-H. Static GCN is also more stable across seeds at the larger training sizes. The tested Static GCN architecture uses the available representation more effectively than the tested EvolveGCN-H architecture under the controlled protocol.

## Small-training-set Evolve instability

At Train20, EvolveGCN-H has MAE 0.5427 ± 0.4938, R² -76.120 ± 115.452, and prediction-SD ratio 6.914 ± 6.462. The large across-seed dispersion and poor R² indicate unstable generalization in this low-data regime; they do not establish a universal property of evolving graph models.

## Prediction spread and residual structure

At Train700, the prediction-SD ratios are 0.855 ± 0.025 for Static GCN and 0.888 ± 0.075 for EvolveGCN-H. Ratios below one indicate compressed prediction spread; ratios above one indicate excessive spread relative to the true targets. The residual definition is prediction minus true target. For the lowest / middle 60% / highest target ranges, the Train700 mean biases are respectively 0.0302 ± 0.0053, 0.0036 ± 0.0125, and -0.0502 ± 0.0091 for Static GCN, and 0.0478 ± 0.0109, 0.0183 ± 0.0143, and -0.0501 ± 0.0164 for EvolveGCN-H. Thus the models tend to overpredict low targets when low-quintile bias is positive and underpredict high targets when high-quintile bias is negative, a regression-to-the-mean pattern visible in the sorted-target and residual plots.

## Train450 versus Train700

The paired Train700-minus-Train450 MAE change is -0.0025 ± 0.0010 for Static GCN and -0.0058 ± 0.0075 for EvolveGCN-H (negative favors Train700). Because each comparison preserves model and seed, this is the cleanest assessment of the final increase in training size, but only three seed pairs support each estimate.

## Scope of the temporal conclusion

The comparison does not show that temporal information is inherently unhelpful. The architectures differ in how they process the representation: the tested EvolveGCN-H uses all five temporal snapshots, whereas the tested Static GCN uses the final snapshot, alongside architecture-specific parameterization and optimization behavior. The results therefore support a statement about these tested implementations under this controlled protocol, not a general causal claim about temporal information. Better temporal architectures, tuning, regularization, or alternative temporal summaries could change the outcome.

## Thesis-ready conclusion

Across the completed U1000 Top1000 scaling matrix, Static GCN delivers lower error, stronger R², and generally greater seed stability than EvolveGCN-H, especially once moderate-to-large training sets are available. EvolveGCN-H is notably unstable in the smallest-data regime and its prediction spread and target-dependent residuals reveal calibration limitations. Increasing training data from 450 to 700 universes should be interpreted from the paired estimates above rather than assumed to help uniformly. Overall, the tested Static GCN architecture uses the available representation more effectively than the tested EvolveGCN-H architecture under the controlled protocol, while the experiment does not establish that temporal information itself lacks predictive value.
