# U1000 Top1000 build and validation plan

This is an inert plan; no command below was executed.

## Gate before build

1. Implement deterministic sparse periodic kNN (`edge_index`, no `[N,N]` or `[N,N,3]` allocation), stable halo and neighbor tie keys, schema/version metadata, and graph/selection hashes.
2. Update Evolve/Static models, loaders and collates for sparse data; add validated external split manifests to Evolve; preserve Static final-snapshot view.
3. Add fail-if-output-exists, same-filesystem temporary output, lock, sharding/resume or checkpointed manifest, atomic rename, checksum, and completion marker.
4. Rerun this audit. Only a GO authorizes the build.

## Future names

- Master: `data/processed/temporal_1000u_none_top1000_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1000_periodic_knn_sparse.pt`
- Metadata: same stem `.metadata.json`; checksum `.sha256`; marker `.complete`.
- Target input: `outputs/target_inspection_1000u.csv`.
- Build manifest: `configs/datasets/camels_simba_u1000_top1000_temporal5_k8_periodic_none_sparse.json`.
- Split manifests: `data/splits/u1000_top1000_fixed_seed{seed}_train{level}.json` plus one summary manifest.

## Safe build sequence

Preflight and source-manifest hash → acquire lock → write deterministic shards/temp path → validate every shard → assemble temp master → full validation → fsync → atomic rename → write metadata/checksum/completion marker → read-only smoke test. An interrupted file without all three matching sidecars is partial and must never be consumed. Resume only verified immutable shards; do not overwrite a complete master.

## Required post-build validation

- Exactly 1,000 unique ordered IDs and five exact snapshots each; all finite targets/features; X `[<=1000,7]` or explicitly padded `[1000,7]`; correct masks; no zero-real-node graph.
- Recompute valid/padding statistics against this audit; Top1000 max real nodes; k=8 degree/edge sanity; periodic boundary fixtures; no self-edges in preprocessing; no edges to padding; edge indices in bounds; no duplicate edge entries after canonicalization.
- Verify feature header mapping and no scaling beyond log10 mass; box 25; target absent from features; positions alone drive topology.
- Verify source/config/ordered-ID/selected-halo/graph hashes, builder commit, schema, atomic completion/checksum.
- Validate all 18 nested manifests: exact counts, ordering, hashes, disjoint fixed val/test, nesting, full coverage at Train700.
- Static view equality for x/edge_index/mask/target/universe/snapshot without copies or rebuild.
- Pilot Evolve Train700 seed42 then Static: no OOM/nonfinite loss; exact test order; metrics recompute; nonconstant predictions; runtime/peak RSS/GPU memory recorded.
