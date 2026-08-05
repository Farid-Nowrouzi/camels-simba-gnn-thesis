# Presentation-ready summary

## Recommended material

1. Protocol table.
2. Two-row main head table.
3. Test MAE by regression head.
4. Paired MAE difference by seed.
5. Prediction-SD-ratio comparison.
6. Representative true-versus-predicted figure.
7. Controlled progression table.

## Supervisor-meeting narrative

**Question:** Does a nonlinear MLP improve the final Omega_m mapping?

**Method:** Five matched seeds with the encoder, pooling, data, splits, and
training protocol fixed.

**Result:** Linear reduced paired MAE by 0.005561
on average and won four of five seeds.

**Diagnostic:** Linear predictions were unrepeated and somewhat better
dispersed on average, but the dispersion effect varied by seed.

**Conclusion:** The tested MLP adds parameters without improving aggregate
generalization; more expressive head families require separate controls.
