# U1000 Top1000 Training Scaling: Slide Summary

- **Evidence:** 36 completed/PASS runs; 6 training-universe counts × 3 seeds × 2 models; 201 test universes per run. One preserved invalid run is excluded.
- **Meaning of TrainN:** N universes are used for training; within each seed, validation (99) and test (201) stay fixed across model and training-count comparisons.
- **Overall winner:** Static GCN has lower MAE in 18/18 seed-matched comparisons and wins mean MAE, RMSE, and R² at every training count.
- **Train700:** Static GCN MAE 0.0397 ± 0.0031, R² 0.815 ± 0.016; EvolveGCN-H MAE 0.0580 ± 0.0033, R² 0.595 ± 0.105.
- **Scaling:** More training universes help overall; Train450→Train700 mean paired MAE changes are -0.0025 ± 0.0010 (Static) and -0.0058 ± 0.0075 (Evolve), indicating smaller late-stage gains.
- **Stability:** Static GCN is generally steadier at larger training counts; EvolveGCN-H is highly seed-sensitive at Train20.
- **Calibration:** At Train700, prediction-SD ratios are 0.855 ± 0.025 (Static) and 0.888 ± 0.075 (Evolve); both show some target-dependent regression toward the mean.
- **How to read plots:** x is true Ωm—not a score where lower is better. Distance from y=x measures prediction error; restricted predicted range indicates compression.
- **Scientific conclusion:** In this implementation and controlled setup, Static GCN performs better. This does **not** establish that temporal information is inherently useless.
- **Table note:** Test universes differ across seeds, so per-universe cross-seed means would be invalid; the 18 sorted tables pair both models within seed.
