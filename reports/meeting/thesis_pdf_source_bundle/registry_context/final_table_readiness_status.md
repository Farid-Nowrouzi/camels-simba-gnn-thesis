# Final Table Readiness Status

Inspection date: 2026-07-29 UTC. This audit uses raw experiment configs,
metrics, prediction CSVs, train logs, lightweight dataset metadata, and artifact
presence. No dataset or checkpoint was loaded.

| # | Intended table | Classification | Evidence available | Missing evidence | Additional runs required | Show professor now? | Required caveat |
|---:|---|---|---|---|---|---|---|
| 1 | Historical 20U–750U inventory | **ready with caveat** | Raw historical configs/metrics and the registry/blueprints cover the major 20U, 50U, 100U, 200U, 500U, and 750U stages. | The generated registry predates the new 20U canonical pilot and two known output summaries are stale. | None. | Yes, as a research-progression table. | It is a timeline of protocol changes, not a controlled scaling curve; 750U also changes Top-N, normalization, splits, and head. Use raw artifacts where summaries disagree. |
| 2 | Controlled EvolveGCN-H 20U–500U scaling | **waiting for experiments** | Ten compatible complete rows: 20U seeds 42/123, 50U seed 123, 100U seed 123, and all 200U/500U seeds. | Five canonical replacements. | 5: 20U/2025; 50U/42 and /2025; 100U/42 and /2025. | No final three-seed curve. | Current state is 10/15; legacy low-U seed-42/2025 rows used 200 rather than 300 maximum epochs and must remain excluded. |
| 3 | EvolveGCN-H Top100–Top200–Top500 scaling | **ready with caveat** | Three seed-matched 500U minmax families at Top100, Top200, and Top500, with h32/L2/B4/300 and k8. | No run evidence is missing. | None. | Yes. | The legacy Top500 seed-42 config omits defaults that are behaviorally compatible; document their resolved values. Do not add 750U Top1000 to this line. |
| 4 | kNN ablation | **ready now** | Three seeds for both EvolveGCN-H and Static GCN at k=4,6,8,12; U500, Top100, minmax, periodic, h64 fixed within each model. | None. | None. | Yes. | State h64 explicitly and avoid a pure absolute cross-model claim because model depth, batch behavior, and temporal access differ. |
| 5 | Normalization ablation | **ready with caveat** | Five behaviorally compatible seeds for none/minmax/zscore at 500U Top500 for both models. | No run evidence is missing. | None. | Yes. | Resolve missing legacy defaults in the annotation; compare within model and do not mix debug/target-normalization pilots. |
| 6 | Regression-head ablation | **ready now** | Five matched 750U Top1000 seeds for MLP versus linear heads. | None. | None. | Yes. | Linear wins in mean, but the advantage is seed-dependent; exclude single-seed 500U head pilots. |
| 7 | Graph-pooling ablation | **ready now** | Five matched 750U Top1000 linear-head temporal-mean seeds for mean versus mean_max. | None. | None. | Yes. | The completed five-seed 750U family is authoritative; exclude the single-seed 500U mean_max pilot. |
| 8 | Temporal-pooling ablation | **ready now** | Five matched 750U Top1000 linear-head graph-mean seeds for temporal mean versus last. | None. | None. | Yes. | Results are nearly tied; do not claim a robust temporal-last advantage. |
| 9 | Static GCN versus EvolveGCN-H | **ready with caveat** | Seed/split-matched 500U Top100 and close-match 500U Top500-none tiers exist. | The proposed 750U Top1000 Static tier is absent. | 3 only if the proposed 750U tier is required: Static seeds 42, 123, 2025. | Yes, restricted to the two existing 500U tiers. | This compares established model pipelines and information pathways, not temporal history alone; disclose depth/head/batch differences. Do not show a 750U tier yet. |
| 10 | Best GNN versus summary baselines | **waiting for validation** | Matched seed-42 500U Top500-none GNN results plus Ridge, Random Forest, and Gradient Boosting aggregate metrics. Code/split audit found no obvious split leakage. | Independent confirmation of raw catalog column semantics and target-generation provenance; the baseline has aggregate metrics but no saved prediction CSV. | None. | No final claim yet. | If provenance is confirmed, label it a one-seed matched comparison and explain that raw summary statistics retain absolute-scale information. |

## Ready now

- kNN ablation.
- Regression-head ablation.
- Graph-pooling ablation.
- Temporal-pooling ablation.

## Ready with caveat

- Historical research progression.
- EvolveGCN-H Top-N scaling.
- Normalization ablation.
- Static GCN versus EvolveGCN-H, restricted to existing 500U tiers.

## Still waiting

- Controlled EvolveGCN-H universe scaling: five runs.
- Best GNN versus summary baselines: provenance validation, not another run.

The classifications above supersede the older blueprint only where current raw
evidence changed: the 20U seed-42 canonical pilot is now complete, reducing the
controlled EvolveGCN-H gap from six runs to five.
