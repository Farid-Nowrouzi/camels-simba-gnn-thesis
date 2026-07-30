# Presentation-ready summary

## Recommended material

1. Protocol table.
2. Two-row main results table.
3. Test MAE by graph pooling.
4. Paired MAE difference by seed.
5. Prediction-SD ratio comparison.
6. Representative true-versus-predicted figure.
7. Representative residual figure.

## Supervisor-meeting conclusion

In a five-seed controlled EvolveGCN-H comparison, mean_max graph pooling
worsened MAE, RMSE, and R² in every matched seed. Prediction dispersion
remained similar and neither method repeated predictions, so the degradation
was not mainly an additional-collapse effect. Simple maximum concatenation
does not improve this U750 Top1000 readout; adaptive pooling remains separate
future work.
