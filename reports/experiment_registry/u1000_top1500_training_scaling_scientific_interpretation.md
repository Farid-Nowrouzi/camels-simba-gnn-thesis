# U1000 Top1500 training-scaling scientific interpretation

## Scope and integrity

This report uses all 36/36 completed and artifact-validated Top1500 runs: EvolveGCN-H and Static GCN at training sizes 20, 50, 100, 200, 450, and 700 with seeds 42, 123, and 2025. For every seed and training size, the ordered train, validation, test, and unused IDs match exactly between Top1000 and Top1500.

The test populations differ across seeds. Each seed has 201 unique test universes; the pairwise overlaps are 42/123=53, 42/2025=51, and 123/2025=52. The three-way overlap is 12, and the union contains 459 universes. Cross-seed prediction rows are therefore analyzed as distinct seed-specific test evaluations, not repeated predictions for one common test population.

## Static GCN versus EvolveGCN-H

Across the 18 training-size/seed matched comparisons, Static GCN has lower MAE in 18, EvolveGCN-H in 0, with 0 ties.

| Training universes | Evolve MAE mean ± SD | Static MAE mean ± SD | Static wins | Evolve wins | Ties |
| --- | --- | --- | --- | --- | --- |
| 20 | 1.368058 ± 0.955295 | 0.064821 ± 0.002954 | 3 | 0 | 0 |
| 50 | 0.158660 ± 0.096365 | 0.057781 ± 0.003615 | 3 | 0 | 0 |
| 100 | 0.096144 ± 0.018335 | 0.050661 ± 0.002555 | 3 | 0 | 0 |
| 200 | 0.069747 ± 0.003792 | 0.045980 ± 0.002934 | 3 | 0 | 0 |
| 450 | 0.054271 ± 0.004384 | 0.041090 ± 0.002082 | 3 | 0 | 0 |
| 700 | 0.055392 ± 0.004653 | 0.037887 ± 0.000942 | 3 | 0 | 0 |

This is a matched model-protocol comparison, not an architecture-only comparison: EvolveGCN-H uses all five snapshots, two graph layers, temporal mean pooling, a linear head, and batch size 4, whereas Static GCN uses the exact final snapshot, three graph layers, an MLP head, and batch size 8.

## Train700 metrics

| Model | MAE mean ± SD | RMSE mean ± SD | R² mean ± SD | Prediction SD / target SD mean ± SD |
| --- | --- | --- | --- | --- |
| EvolveGCN-H | 0.055392 ± 0.004653 | 0.084741 ± 0.027509 | 0.462177 ± 0.332427 | 0.932660 ± 0.183853 |
| Static GCN | 0.037887 ± 0.000942 | 0.049348 ± 0.000624 | 0.826323 ± 0.010566 | 0.870902 ± 0.044981 |

## Top1000 versus Top1500

The paired difference is defined as **Top1500 MAE − Top1000 MAE**. Negative values favor Top1500; positive values favor Top1000.

| Model | Train | Seed 42 | Seed 123 | Seed 2025 | Mean ± SD | Top1500 wins | Top1000 wins | Ties |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EvolveGCN-H | 20 | 2.000370 | 0.402070 | 0.073509 | 0.825316 ± 1.030801 | 0 | 3 | 0 |
| EvolveGCN-H | 50 | -0.029198 | -0.025994 | 0.001062 | -0.018043 ± 0.016623 | 2 | 1 | 0 |
| EvolveGCN-H | 100 | 0.005421 | -0.202370 | -0.007350 | -0.068100 ± 0.116457 | 2 | 1 | 0 |
| EvolveGCN-H | 200 | 0.001175 | -0.004546 | 0.001269 | -0.000701 ± 0.003330 | 1 | 2 | 0 |
| EvolveGCN-H | 450 | -0.016478 | -0.004089 | -0.007955 | -0.009507 ± 0.006339 | 3 | 0 | 0 |
| EvolveGCN-H | 700 | 0.004816 | -0.010782 | -0.001785 | -0.002584 ± 0.007830 | 2 | 1 | 0 |
| Static GCN | 20 | -0.008355 | 0.004043 | -0.007430 | -0.003914 ± 0.006907 | 2 | 1 | 0 |
| Static GCN | 50 | 0.002446 | -0.004593 | 0.000818 | -0.000443 ± 0.003685 | 1 | 2 | 0 |
| Static GCN | 100 | 0.002703 | -0.003382 | 0.000798 | 0.000040 ± 0.003113 | 1 | 2 | 0 |
| Static GCN | 200 | 0.000295 | -0.002346 | 0.001751 | -0.000100 ± 0.002077 | 1 | 2 | 0 |
| Static GCN | 450 | -0.002872 | 0.001524 | -0.001786 | -0.001045 ± 0.002290 | 2 | 1 | 0 |
| Static GCN | 700 | -0.004986 | -0.000792 | 0.000416 | -0.001787 ± 0.002835 | 2 | 1 | 0 |

- EvolveGCN-H: Top1500 has lower MAE in 10/18 paired cells, Top1000 in 8/18, with 0 ties.
- Static GCN: Top1500 has lower MAE in 9/18 paired cells, Top1000 in 9/18, with 0 ties.

## Training-size behaviour

- EvolveGCN-H: mean MAE changes from 1.368058 at Train20 to 0.055392 at Train700; the lowest mean MAE is at Train450. The mean learning curve is not monotonic, and the largest seed-to-seed MAE SD occurs at Train20 (0.955295).
- Static GCN: mean MAE changes from 0.064821 at Train20 to 0.037887 at Train700; the lowest mean MAE is at Train700. The mean learning curve is monotonic, and the largest seed-to-seed MAE SD occurs at Train50 (0.003615).

## Prediction dispersion

Prediction-SD / target-SD near one indicates similar marginal dispersion; values below one indicate prediction compression, and values above one indicate overdispersion. This ratio is a dispersion diagnostic and does not establish calibration by itself.

- EvolveGCN-H: Train700 prediction-SD / target-SD is 0.932660 ± 0.183853; on average the predictions are compressed relative to the target distribution.
- Static GCN: Train700 prediction-SD / target-SD is 0.870902 ± 0.044981; on average the predictions are compressed relative to the target distribution.

## Conclusions and limitations

The analysis is descriptive and uses only three seeds per cell, so the reported means and sample standard deviations should not be treated as inferential uncertainty. Test populations differ across seeds, while Top1000/Top1500 comparisons are exactly paired within each seed. Static GCN and EvolveGCN-H also use different input and model protocols. No significance, causal, or architecture-only claim is made.
