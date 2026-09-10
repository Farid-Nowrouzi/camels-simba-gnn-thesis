# Deprecated result artifacts

Status date: 2026-09-10 UTC. These files are retained as historical evidence and must not be used in the thesis.

| Path | Why deprecated | Authoritative replacement | Thesis use | Status / evidence |
|---|---|---|---|---|
| `outputs/evolvegcn_500top500_vs_750top1000_5seed_table.csv` | Its 30 stored test MAE/RMSE/R² values disagree with metrics recomputed from the named runs' raw test predictions. | `reports/reproducibility/evolvegcn_500u_top500_vs_750u_top1000_authoritative.csv` | NO | DEPRECATED; registry rebuild detected 30 mismatches on 2026-09-10. |
| `outputs/evolvegcn_500u_vs_750u_results/seed_level_results_clean.csv` | Its 20 stored test MAE/RMSE values inherit the same stale values. | `reports/reproducibility/evolvegcn_500u_top500_vs_750u_top1000_authoritative.csv` | NO | DEPRECATED; registry rebuild detected 20 mismatches on 2026-09-10. |

The replacement contains the same two configurations and five seeds but recomputes test MAE, RMSE, and R² directly from each named experiment's `predictions/test_predictions.csv`. Notebook 07 now reads the replacement and reconstructs group summaries from it. No historical file was deleted or overwritten.

The registry's full mismatch evidence is in `reports/experiment_registry/registry_validation_report.md`. The Phase 3 scan found no other derived artifact with a detected disagreement against raw predictions.
