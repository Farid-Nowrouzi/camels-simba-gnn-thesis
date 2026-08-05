# U1000 Top1000 Static final-snapshot reuse audit

## Decision

**A — direct reuse supported in the current repository.** The Static CLI accepts `--dataset_format temporal_final_snapshot` and its in-memory converter exposes the final elements of `A_list`, `Nodes_list`, and `mask_list` plus the unchanged target. It does not copy tensor contents, select halos, normalize, or run kNN. `[-1]` is unambiguously a=1.00000 because the temporal builder sorts the exact five scale factors numerically.

## Schema mapping

| temporal field | Static field | expected shape | equivalence |
|---|---|---|---|
| `A_list[-1]` | `A` | current dense `[1000,1000]` | same tensor reference |
| `Nodes_list[-1]` | `X` | `[1000,7]` | same tensor reference |
| `mask_list[-1]` | `mask` | `[1000,1]` | same tensor reference |
| `target` | `target` | scalar → batch `[B,1]` | unchanged |
| dictionary key | universe ID | string | retained by Dataset wrapper |
| `snapshots[-1]` | final identity | a=1.00000 | currently not copied into view |

The audit did not deserialize the U750 graph file, so tensor equivalence was established from code and existing validation metadata, not re-executed.

## Exact implementation recommendation

Retain `convert_temporal_final_snapshot_to_static` and `load_dataset` in `src/training/train_static_gcn.py`; after the mandatory sparse refactor, update the view to expose `edge_index_list[-1]` (and optional edge weights) by reference and retain `snapshots[-1]` metadata. Update `CamelsStaticGraphDataset`, collate, and `StaticGCNRegressor` for sparse batches. Keep backward-compatible `dataset_format=static` and legacy dense loading only for historical datasets; require explicit schema metadata and never silently densify. No derived static file should be created.

## Canonical Static protocol

Hidden 32; three GCN layers; batch 8; graph mean pooling; fixed MLP readout `32→32→16→1`; dropout 0.2; AdamW lr 0.001, weight decay 1e-5; MSE; 300 epochs; patience 40; clip 1.0; layer norm/residual/model self-loops as canonical trainer defaults; raw target; same seed-specific split manifest. No exact Top1000 historical Static run exists.
