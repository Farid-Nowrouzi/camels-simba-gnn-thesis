# U1000 Top1000 Notebook 11 Creation Audit

## Artifact identity

- Notebook path: `notebooks/visualization/11_u1000_top1000_training_universe_scaling.ipynb`
- Notebook 10 template path: `notebooks/visualization/10_final_experiment_report.ipynb`
- Notebook title: **U1000 Top1000 Training-Universe Scaling: EvolveGCN-H versus Static GCN for Omega_m Regression**
- Current source/analysis commit displayed by the executed notebook: `3cd330e8b27cccd11912be12c802598fcb1a3ee7`
- Main section count: 20, including the title/executive-summary section
- Appendix section count: 5
- Total top-level section count: 25
- Markdown-cell count: 36
- Code-cell count: 26
- Executed code-cell count: 26

## Figures used

The notebook embeds 15 unique figures:

1. `presentation_assets/u1000_top1000_training_scaling/08_mae_learning_curve.png`
2. `presentation_assets/u1000_top1000_training_scaling/09_rmse_learning_curve.png`
3. `presentation_assets/u1000_top1000_training_scaling/10_r2_learning_curve.png`
4. `presentation_assets/u1000_top1000_training_scaling/11_prediction_sd_ratio_curve.png`
5. `presentation_assets/u1000_top1000_training_scaling/07_train700_sorted_low_to_high_omega_m.png`
6. `presentation_assets/u1000_top1000_training_scaling/08_train700_residual_vs_true.png`
7. `presentation_assets/u1000_top1000_training_scaling/09_static_minus_evolve_mae.png`
8. `presentation_assets/u1000_top1000_training_scaling/10_train450_vs_train700.png`
9. `presentation_assets/u1000_top1000_training_scaling/14_main_metrics_table.png`
10. `notebooks/visualization/outputs/11_u1000_top1000_training_universe_scaling/train20_true_vs_predicted_target_zoom.png`
11. `notebooks/visualization/outputs/11_u1000_top1000_training_universe_scaling/train50_true_vs_predicted_target_zoom.png`
12. `notebooks/visualization/outputs/11_u1000_top1000_training_universe_scaling/train100_true_vs_predicted_target_zoom.png`
13. `notebooks/visualization/outputs/11_u1000_top1000_training_universe_scaling/train200_true_vs_predicted_target_zoom.png`
14. `notebooks/visualization/outputs/11_u1000_top1000_training_universe_scaling/train450_true_vs_predicted_target_zoom.png`
15. `notebooks/visualization/outputs/11_u1000_top1000_training_universe_scaling/train700_true_vs_predicted_target_zoom.png`

The six notebook-specific true-versus-predicted figures use a common target-range presentation window. Every test point is retained: predictions outside the visible range are represented by boundary triangles and counted in the corresponding panel.

## Tables used

The notebook renders 27 tables or per-model/per-seed table views:

- population accounting;
- exact ordered-ID manifest summary;
- controlled variables;
- model architecture comparison;
- authoritative matrix validation;
- compact 12-row main metrics;
- six Train700 universe-example tables;
- Train700 target-range bias/compression summary;
- 18-row paired Static-versus-Evolve comparison;
- six-row Train450-versus-Train700 comparison;
- two-row late-scaling summary;
- scientific diagnosis;
- complete 36-run metric table;
- six complete Train700 prediction tables;
- runtime/best-epoch table;
- final notebook audit table.

The notebook also embeds 21 collapsible split-provenance blocks: three detailed Train700 blocks and one block for each of the 18 immutable manifests.

## Validation result

**PASS**

- Valid nbformat JSON: PASS
- Top-to-bottom execution: PASS
- Code cells executed: 26/26
- Failed cells or traceback outputs: 0
- Required authoritative inputs resolved: PASS
- Completed/PASS run count: 36
- Per-model run count: 18 EvolveGCN-H and 18 Static GCN
- Training levels: Train20, Train50, Train100, Train200, Train450, Train700
- Seeds: 42, 123, 2025
- Immutable manifest count: 18
- Exact train/validation/test/unused counts: PASS
- Pairwise split disjointness: PASS
- Ordered training-subset nesting: PASS
- Fixed validation/test identities within seed: PASS
- Train700 accounts for all `LH_0` through `LH_999`: PASS
- Static/Evolve manifest identity within matched comparisons: PASS
- Preserved invalid experiment excluded: PASS
- All figure and table inputs resolved: PASS
- Main scientific conclusion matches the authoritative interpretation: PASS
- Halo-count scaling transition is present: PASS
- `git diff --check`: PASS

## Execution and integrity statement

Notebook execution completed successfully in a clean Jupyter kernel using
`/home/ml/thesis-camels/envs/camels-gnn/bin/python`. The execution performed
read-only analysis of validated CSV, JSON, manifest, prediction, and image
artifacts. No training or epoch loop occurred. No CUDA job was launched. No
graph dataset, checkpoint, split manifest, original prediction, or original
metric artifact was created or modified. The only generated scientific assets
are the six notebook-specific presentation figures listed above.
