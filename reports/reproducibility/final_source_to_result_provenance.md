# Final source-to-result provenance

This is the canonical Phase 3 source-to-result map. Per-seed paths and hashes are in `final_thesis_artifact_manifest.csv`; aggregate values and thesis roles are in `../experiment_registry/thesis_final_selection.csv`. Unknown history is stated rather than inferred.

| Family | Source | Trainer / runner | Config and split | Result | Final report / notebook | Selection basis |
|---|---|---|---|---|---|---|
| Gradient Boosting | `src/evaluation/run_modern_summary_baselines.py` | same module; exact invocation in config where available | per-run config; declared split manifest | Top1500 three-seed directories in `experiments/` | Notebook 13 | frozen configuration; training commit stored in configs where known |
| Static GCN | `src/models/static_gcn.py` | `src/training/train_static_gcn.py` | Top1500 raw7 kNN(k=8), seeds 42/123/2025 | canonical three-seed run directories | Notebooks 12, 13, 15, 16 | k and representation validation-ranked; post-freeze test evaluation |
| Static PNA | `src/models/static_pna.py` | `src/training/train_static_pna.py`; production runner under `scripts/production/` | frozen Static-GCN-matched Top1500 splits | canonicalized three-seed run directories | Notebook 15 | frozen architecture comparison; post-freeze evaluation; PNA was not selected over Static GCN |
| EvolveGCN-H | `src/models/evolvegcn_h.py` | `src/training/train_evolvegcn_h.py` | Top1500 raw7 kNN(k=8), five snapshots | canonical three-seed Train700 directories | Notebooks 12, 14, 16 | frozen configuration; validation/test separation recorded in temporal reports |
| EvolveGCN-O | `src/models/evolvegcn_o.py` | `src/training/train_evolvegcn_o.py` | same frozen temporal dataset and splits | canonicalized three-seed directories | Notebook 16 | validation-ranked comparison; seed 123 recovered by inference-only finalization with unchanged checkpoint |
| GCN-GRU | `src/models/gcn_gru.py` | `src/training/train_gcn_gru.py` | same frozen temporal dataset and splits | canonicalized three-seed directories | Notebook 16 | validation-ranked; final five-model validation winner |
| GCN Temporal Transformer | `src/models/gcn_temporal_transformer.py` | `src/training/train_gcn_temporal_transformer.py` | same frozen temporal dataset and splits | canonicalized three completed runs; interrupted seed-123 archive retained separately | Notebook 16 | validation-ranked comparison; interrupted archive excluded from aggregates |
| DeepSets | `src/models/deepsets.py` | `src/evaluation/run_modern_deepsets.py`; exact historical command UNKNOWN where absent | raw7 final-snapshot halo set; Top1000 headline | canonical three-seed run directories | Notebook 13 | frozen configuration; Top1000 headline documented by Notebook 13; earlier selection provenance is incomplete |

All headline rows identify config, metrics, raw test predictions, model/checkpoint, split manifest, source commit when actually recorded, and SHA-256 values. Missing historical commits are `UNKNOWN`; they are not reconstructed from later repository history.

Historical freeze and closure reports may retain absolute Notebook15/16 paths because those paths describe where the work originally occurred. They are provenance statements, not active loading dependencies. Canonical notebooks and validation code load final artifacts from this repository.
