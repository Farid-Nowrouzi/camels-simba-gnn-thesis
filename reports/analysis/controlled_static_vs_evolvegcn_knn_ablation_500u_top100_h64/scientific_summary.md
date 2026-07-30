# Scientific Summary: Controlled kNN Connectivity Ablation at h64

## Scope and question

How does changing periodic-kNN connectivity across k = 4, 6, 8, and 12 affect Omega_m regression for Static GCN and EvolveGCN-H at 500 universes, Top100 halos, minmax normalization, and hidden dimension 64?

The comparison is **mostly controlled**: population, graph protocol, k values, hidden dimension, seeds, and exact split IDs match. Temporal versus final-snapshot input, architecture, layer count, batch size, regression head, and temporal pooling intentionally differ. It is not a pure architecture-only ablation.

## Answers to the scientific questions

1. **Does mean Test MAE change materially across k?** The full mean-MAE range is 0.001211 for EvolveGCN-H and 0.001138 for Static GCN. These shifts are much smaller than the between-seed sample SDs (0.009886–0.011783 and 0.009144–0.010580, respectively), so the evidence does not support a material connectivity effect at this resolution.
2. **Is there a consistent best k across seeds?** No. Seed-level rankings change, and the mean curves are non-monotonic.
3. **Do both models favour the same k descriptively?** Yes: k=8 has the smallest mean MAE for both (0.096562 and 0.096295), but its advantage is tiny relative to seed variability and is not evidence of a universal optimum.
4. **Which model wins paired comparisons?** Static GCN has lower MAE in 4/12 matched k/seed pairs and EvolveGCN-H in 8/12. This descriptive split does not identify a causal architecture or temporal-input effect.
5. **How large is seed variability?** It dominates the differences among k means for both models.
6. **Does increasing k reduce prediction compression?** Not consistently. Prediction-SD ratios remain far below 1 and do not improve monotonically with k.
7. **Does increasing k reduce repeated predictions?** Not consistently. EvolveGCN-H has no exact repeats in these files, whereas Static GCN shows seed- and k-dependent repetition, including severe cases at k=4 and k=8 for seed 42.
8. **Does denser connectivity solve regression toward the mean?** No. Across the canonical rows, prediction-SD/target-SD spans 0.003–0.144, remaining substantially below 1.
9. **Are apparent best-k results robust?** No; the differences are small and seed-dependent.

## Representative runs

Representative plots use the median-MAE seed within every model/k group, never the best seed automatically.

## Caveats

- Selecting the smallest mean MAE from four tested k values and three seeds does not prove a universally optimal k.
- Poor dispersion and repeated predictions identify behaviour, not its mechanism.
- Temporal processing cannot be claimed as the cause of cross-model differences because several model-specific protocol fields differ intentionally.
- This experiment tests kNN connectivity at hidden dimension 64 and does not automatically generalize to h32.
