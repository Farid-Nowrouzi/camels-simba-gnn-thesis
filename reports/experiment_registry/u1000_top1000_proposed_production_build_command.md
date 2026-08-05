# Proposed U1000 Top1000 production build command

**Inert review artifact — do not execute while the final decision is NO-GO.** The command below is the exact current CLI expression of the scientific graph protocol. It deliberately omits `--overwrite`, `--allow_partial`, `--force_unsafe_dense`, and `--dummy_target`.

```bash
envs/camels-gnn/bin/python -m src.data.build_temporal_sequences \
  --raw_dir data/raw/CAMELS_SIMBA_1000U \
  --output_path data/processed/temporal_1000u_none_top1000_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1000_periodic_knn_sparse.pt \
  --num_universes 1000 \
  --num_snapshots 5 \
  --num_nodes 1000 \
  --normalization none \
  --graph_mode knn \
  --k 8 \
  --periodic_boundary \
  --box_size 25.0 \
  --graph_storage sparse_edge_index \
  --targets_csv outputs/target_inspection_1000u.csv \
  --device cpu
```

The raw-file chooser sorts the five discovered files numerically, yielding `0.20000, 0.25000, 0.51209, 0.75065, 1.00000`. `num_universes=1000` means exactly `LH_0..LH_999`. The output directory is also where UUID-named temporary data/metadata files and the exclusive `.pt.lock` reside, so publication is same-filesystem. Default collision behavior is refusal. Successful publication creates `.pt`, `.metadata.json`, and `.complete`; the checksum is recorded in the latter two.

Logical dataset ID: `camels_simba_u1000_top1000_temporal5_none_periodic_knn_k8_box25_sparse_v1`. Immutable split `dataset_identity` must be the post-build SHA-256 from the completion record, conventionally `sha256:<64-hex>`, because serialized bytes and their checksum do not exist before the build.

## Why this command is not yet authorized

`outputs/target_inspection_1000u.csv` intentionally does not exist yet. More importantly, the current CLI has no source-manifest-policy or precomputed-manifest option: it always records the weak path/size/mtime catalogue aggregate and does not hash the target CSV. After the required code remediation, the reviewed command must gain an explicit full-content source-manifest input/policy and verify all 5,000 catalogue SHA-256 values plus the target SHA-256 before processing. The command above is therefore exact for scientific parameters but not yet provenance-complete.

Future authorized operation should run the re-reviewed command in a monitored `tmux` session, with at least 5 GB free, logging wall time and peak RSS. That recommendation is deferred until a subsequent gate returns GO.
