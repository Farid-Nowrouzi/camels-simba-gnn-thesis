# Presentation-ready Summary

## Experiment question

How does periodic-kNN connectivity affect Omega_m regression under the completed 500U Top100 h64 protocols?

## One-sentence protocol

Compare Static GCN and EvolveGCN-H at k = 4, 6, 8, and 12 over seeds 42, 123, and 2025 with matched population, preprocessing, graph construction, hidden dimension, and exact splits.

## Four numerical findings

- Both models have their smallest descriptive mean MAE at k=8: EvolveGCN-H 0.096562; Static GCN 0.096295.
- The mean-MAE range across k is only 0.001211 for EvolveGCN-H and 0.001138 for Static GCN.
- Static GCN wins 4/12 paired MAE comparisons; EvolveGCN-H wins 8/12.
- Prediction-SD ratios span 0.003–0.144; the largest exact-repeat fraction is 0.973.

## Recommended outputs

- **Main table:** `tables/main_results_table.md`, with mean ± sample SD by model and k.
- **Main figure:** `figures/test_mae_vs_k.png`, showing seed points and mean ± sample SD.
- **Diagnostic figure:** `figures/repeated_prediction_fraction_vs_k.png`, paired with `figures/prediction_std_ratio_vs_k.png`.

## Presentation-safe conclusions

- Connectivity effects are small relative to seed variability under this h64 protocol.
- k=8 is the descriptive minimum-mean-MAE setting for both models, but the result is not robust enough to claim a universal optimum.

## Important caveats

- Cross-model comparisons are mostly controlled, not architecture-only, because temporal input and several model-specific settings differ.
- Prediction compression and exact repeats are empirical diagnostics; their cause is not established.

## Suggested captions

- **Main figure:** “Test MAE across periodic-kNN connectivity at 500 universes, Top100 halos, and h64. Points are seeds; curves show mean ± sample SD.”
- **Diagnostic figure:** “Exact repeated-prediction fraction across k. Severe Static GCN repetition is seed-dependent and is not resolved monotonically by denser connectivity.”
