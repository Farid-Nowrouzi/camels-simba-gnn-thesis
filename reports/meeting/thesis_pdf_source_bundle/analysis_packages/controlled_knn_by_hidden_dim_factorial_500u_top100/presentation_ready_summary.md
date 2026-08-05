# Presentation-Ready Summary

## Recommended main items

1. `protocol_table.md`
2. `tables/main_results_table.md`
3. `figures/test_mae_vs_k_by_hidden_dim.png`
4. `figures/paired_width_mae_difference_vs_k.png`
5. `figures/prediction_std_ratio_vs_k_and_width.png`
6. `figures/repeated_prediction_fraction_vs_k_and_width.png`
7. `figures/representative_true_vs_predicted.png`

## Central conclusion

Training and artifact completion are 48/48, and both family verifiers pass. Under the tested protocol, changing k or increasing width does not yield a consistent performance improvement relative to between-seed variability. Prediction compression is widespread and one Static h32/k4/seed42 run is exactly constant, making Pearson undefined rather than zero. Results use three seeds, and Static-versus-Evolve differences are descriptive protocol comparisons rather than causal temporal ablations.
