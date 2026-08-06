# U1000 Top1000 Training Scaling Thesis Summary

The final matrix contains 36 completed and validated runs: EvolveGCN-H and Static GCN at 20, 50, 100, 200, 450, and 700 training universes, with seeds 42, 123, and 2025 and 201 test universes per run. Prediction IDs, ordering, targets, finiteness, and reported metrics were independently verified from the stored artifacts.

At Train700, Static GCN reaches MAE 0.0397 ± 0.0031 and R² 0.815 ± 0.016, whereas EvolveGCN-H reaches MAE 0.0580 ± 0.0033 and R² 0.595 ± 0.105. Static GCN has lower MAE in 18/18 seed-matched comparisons across the full learning curve. The EvolveGCN-H results are especially variable at Train20, and both models show target-dependent residual structure consistent with some regression toward the mean.

The thesis-level conclusion is deliberately architecture-specific: **the tested Static GCN architecture uses the available representation more effectively than the tested EvolveGCN-H architecture under the controlled protocol.** This result does not imply that temporal information is inherently unhelpful, because the comparison also reflects model architecture, capacity, optimization, and the particular temporal aggregation strategy.

Supporting evidence is provided in the per-run and mean±sample-SD tables, paired model and Train450/Train700 comparisons, six ID-preserving sorted prediction tables, and ten figures in `presentation_assets/u1000_top1000_training_scaling/`.
