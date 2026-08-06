# U1000 Train700 Seed42 Full Pilot Audit

Run finalized UTC: 2026-08-06T00:10:01.878357+00:00

## Final decision

**GO FOR REMAINING MATRIX**

Both authorized pilots passed infrastructure and artifact validation. Weak scientific performance does not alter infrastructure PASS.

## Evolve

- Experiment: `experiments/evolvegcn_h_u1000_top1000_sparse_train700_seed42_none_h32_l2_mean_temporal_mean_linear`
- Best epoch / epochs run: 30 / 70
- Test MAE / RMSE / R2: 0.05544274356 / 0.06747741308 / 0.6893879244
- Prediction SD ratio: 0.8304538617
- Repeated-prediction fraction: 0
- Runtime seconds: 532
- Peak GPU memory MiB: 800.0
- Exact ordered test IDs: PASS
- Metric recomputation: PASS

## Static

- Experiment: `experiments/static_gcn_u1000_top1000_sparse_train700_seed42_none_h32_l3_mean_mlp_final`
- Best epoch / epochs run: 73 / 113
- Test MAE / RMSE / R2: 0.04318159033 / 0.05444578993 / 0.7977772122
- Prediction SD ratio: 0.8550144853
- Repeated-prediction fraction: 0
- Runtime seconds: 129
- Peak GPU memory MiB: 610.0
- Exact ordered test IDs: PASS
- Metric recomputation: PASS
