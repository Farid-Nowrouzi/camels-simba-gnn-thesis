# Controlled normalization ablation package

This package contains the complete model-stratified U500 Top500 k=8
node-feature normalization analysis:

- 30 verified seed-level rows;
- 6 aggregate model × normalization rows;
- 30 within-model paired-normalization rows;
- 15 descriptive cross-model paired rows;
- all requested tables;
- 13 figures, each with PNG, PDF, and plot-data CSV.

The source experiments are read-only. Graph datasets and checkpoints are never
loaded. Run the generic builder first, then:

```bash
env MPLCONFIGDIR=/tmp/codex-matplotlib-cache   envs/camels-gnn/bin/python   reports/analysis/controlled_static_vs_evolvegcn_normalization_ablation_500u_top500/rebuild_normalization_outputs.py   --repo-root /home/ml/thesis-camels   --spec configs/analysis_reports/controlled_static_vs_evolvegcn_normalization_ablation_500u_top500.json
```

The primary scientific comparisons are within model. Cross-model differences
are descriptive protocol comparisons.
