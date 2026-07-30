# Top-N halo-count scaling reporting plan

## Recommended analysis

Identifier: `controlled_static_vs_evolvegcn_topn_halo_scaling_500u`.

Use 18 rows: EvolveGCN-H and Static GCN, Top100/Top200/Top500, and seeds
42/123/2025. Analyze Top-N effects within each model. Cross-model differences
are descriptive because input mode, layer count, batch size, and readout differ.

## Tables

- Full fixed-protocol table.
- Dataset Top-N table.
- Seed-level and aggregate model × Top-N results.
- Paired Top200−Top100, Top500−Top200, and Top500−Top100 differences.
- Prediction-compression diagnostics.
- Computational-cost table with explicit missing timing/memory fields.
- Descriptive best-Top-N summary.

## Figures

- Test MAE, RMSE, and R² versus Top-N.
- Paired MAE differences.
- Per-seed MAE trajectories.
- Prediction-SD ratio and repeated-prediction fraction versus Top-N.
- Median-MAE representative true-versus-predicted, residual, and distribution
  plots.
- Nodes/edges versus Top-N.
- Training time versus Top-N only if later evidence exists.
- Predictive improvement versus computational cost with unavailable cost axes
  clearly omitted rather than fabricated.

Representative runs must use median test MAE within every model × Top-N cell.
Top1000 belongs in a separately labelled descriptive U750 context, not the
controlled trend.

## Interpretation policy

Report that Evolve improves monotonically across the three matched seeds if
the final rebuild reproduces this audit. Report Static as essentially flat and
mixed. State that minmax statistics change with Top-N and that graph topology
is rebuilt. Do not claim node count universally improves GNNs.

No new training or graph rebuilding is required for the primary analysis.
