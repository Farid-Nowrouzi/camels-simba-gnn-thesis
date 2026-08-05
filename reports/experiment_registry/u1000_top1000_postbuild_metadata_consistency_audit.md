# U1000 Top1000 post-build metadata consistency audit

Audit date: 2026-08-05 UTC  
Branch: `thesis-sparse-integrity-hardening`  
Mode: read-only artifact inspection; no build, kNN, split creation, or training

## Outcome

**Classification: C — serialized-record provenance defect that does not alter tensors.**

The historical string `v2_logmass_minmax_top100_periodic_knn` is not merely terminal output. It occurs once in the metadata sidecar, at the top level of all 1,000 serialized universe records, and in all 5,000 serialized snapshot records. The serialized root is only the ordered universe dictionary (`LH_0` through `LH_999`); it has no separate dataset-level metadata object.

The string is false as a compact description of this build because the effective fields are normalization `none`, Top1000, k=8, periodic boundaries, box size 25, and `sparse_edge_index`. It is provenance text only. Tensor and target content is governed by separate fields and was independently validated.

**Decision: GO FOR SPLIT CREATION.** The stale string is not consumed by either trainer and does not control normalization, Top-N selection, feature construction, graph representation, batching, or model behavior. Split creation must bind to dataset checksum `6a794c38216175212ec969283fdbd214bcdc64caecde78d24b81545e681a332a`; no split was created during this audit.

## Exact active U1000 occurrences

- `data/processed/temporal_1000u_none_top1000_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1000_periodic_knn_sparse.pt`: 1,000 universe-level values plus 5,000 snapshot-level values. This binary was inspected through `torch.load`; it was not edited.
- `data/processed/temporal_1000u_none_top1000_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1000_periodic_knn_sparse.metadata.json`: one sidecar value.
- `logs/dataset_builds/u1000_top1000_sparse_build_20260805T204219Z.log`: 1,002 emitted values (builder header, one per universe, and final summary).
- `outputs/target_inspection_1000u.csv`: 1,000 `preprocessing_version_context` values. This is historical target-inspection context, not a field used to derive `omega_m`; it is part of the immutable source manifest and was not changed.

Repository search also found the literal in historical datasets, validation outputs, experiment-family declarations, reports, diagnostic output, and compatibility documentation. Those references describe earlier protocols or inherit the same historical label and are outside this artifact correction scope. The behavior-bearing source audit found the legacy constant in `src/data/camels_graph_utils.py` and comparison/display uses in legacy static and diagnostic validators. Trainer occurrences are docstrings only.

## Assignment trace and prospective correction

Before this audit, `src/data/camels_graph_utils.py` defined one hard-coded `PREPROCESSING_VERSION`. `process_snapshot` assigned it to individual snapshot results; `build_universe_sequence` copied it to each snapshot record and assigned it to each universe record. `src/data/build_temporal_sequences.py` reused the constant for validation, terminal output, and the sidecar.

The production artifact was not changed. Future temporal builds now call `preprocessing_version_for_config(...)`, which incorporates normalization, Top-N, periodicity, graph mode and k/radius, box size, and graph storage. The production configuration would emit:

`v3_logmass_none_top1000_periodic_knn_k8_box25_sparse_edge_index`

The generated value is used consistently for temporal terminal output, sidecar metadata, universe records, snapshot records, and in-build consistency checks. Regression coverage exercises contrasting configurations and a complete tiny sparse build, asserting equality across all four output locations. The historical constant remains for legacy static/single-snapshot compatibility and is explicitly marked as such.

## Downstream-consumer audit

- EvolveGCN-H loader/trainer: loads the universe dictionary and dispatches graph representation from `edge_index_list`; it never reads `preprocessing_version`.
- Static loader/trainer: converts the final temporal snapshot by direct object reference and dispatches sparse representation from `edge_index`; it never reads `preprocessing_version`.
- Sparse batch utilities: use `graph_storage`/edge keys, tensor shapes, masks, and edge indices; they do not read the label.
- Split utility: records checksum, source-manifest hash, target hash, storage, schema, config hash, and builder commit from the sidecar. It does not read the label.
- Production validator: requires the field to exist but derives and checks protocol identity from authoritative fields. It does not parse the label.
- Registry builder: reads `normalization`, `num_nodes`, `k`, periodicity, box size, feature names, dataset type, and other structured fields. It does not parse `preprocessing_version` to infer behavior.
- Analysis reporting: may compare explicitly declared preprocessing-version fields as provenance strings, but no inspected analysis code parses this string into normalization, Top-N, feature semantics, storage, or model behavior.
- Legacy graph validators/diagnostics compare the value to the historical constant as a provenance equality check. They do not use it to transform tensors or configure a model.

## Independent effective-protocol verification

The sidecar, full serialized scan, target table, source manifest, completion marker, and dedicated validator jointly establish:

- 1,000 ordered universes (`LH_0..LH_999`), 1,000 successful, zero failed, partial mode false.
- Five ordered snapshots: 0.2, 0.25, 0.51209, 0.75065, 1.0.
- All 5,000 node tensors have shape `[1000,7]`; feature order is `log10_Mvir,X,Y,Z,VX,VY,VZ`.
- All 1,000 universe records say normalization `none`, Top-N/num_nodes 1000, graph mode kNN, periodic true, and box size 25.
- All 5,000 snapshot records say normalization `none`, k=8, periodic kNN true, and box size 25.
- All universe records use schema `camels_temporal_sparse_v1` and storage `sparse_edge_index`; no dense adjacency key/tensor is present.
- Target table: `outputs/target_inspection_1000u.csv`, SHA-256 `9692a97760ee0e3a97cf3293f1b73911ee0a1af028617f03fe85f88f431703c2`, 1,000 target rows, exact float32 equality for every serialized target.
- Source manifest: `camels_source_manifest_v1`, policy `full_sha256`, 5,001 entries (5,000 catalogues plus one target table), manifest SHA-256 `ba22c3611a70763566ffb38a20f9b5a36fb6c1a27c3ad8030c4a7e189ce87618`, verification result `verified_full_sha256`.
- Builder config hash: `807922adb36064e572ea9e3e9be9b5f9456a0c788180a60d13e68d7518a34a7b`.

## Checksums and tensor hashes

No production artifact was modified, so pre- and post-audit checksums are identical:

| Artifact | Before | After |
|---|---|---|
| dataset `.pt` | `6a794c38216175212ec969283fdbd214bcdc64caecde78d24b81545e681a332a` | same |
| metadata sidecar | `d4ea0ba0c3a1abc6f49d6856be86c7fc1226090daac8924eb6b72262d22753b9` | same |
| completion marker | `4eea1a4bbbfc57d0c3420a115ae436240e0dcb1588cf47588ab2ee5809edd85a` | same |

Deterministic tensor-content hashes from the ordered 1,000-universe scan (dtype, shape, and contiguous bytes included per tensor):

- `X`: `5a5c5ecce8459f4a9549e1c99dde0a46d64817247d377d25fffaf1f4cbdc23e9`
- `edge_index`: `0ae0e05571b3dbff04a0c626ee1212deb53ed8e9f631be7427c6f2df56311301`
- masks: `53c7896b514dabb9a0ae81a36410bde58c78180250183f2271d713d43252812e`
- targets: `418daeee3b73dfef896bd832d02c5fe36888e040d00a3f404c6d35a13cc01c87`
- ordered combined scientific tensors: `c2263ea187154c5695e769c70f3fc01dc947d21eedc1235e1d31aca19106df8d`

## Validation

The dedicated read-only validator returned `PASS` before the source correction and `PASS` again afterward. The focused sparse pipeline suite passed 14/14 tests. No build, neighbor search, split, checkpoint, prediction, or training process was run.

## Migration plan (not executed)

Migration is unnecessary for model behavior and was not authorized. If uniform provenance is later required, it must be an explicit atomic migration, not an in-place edit:

1. Require separate approval, at least 2 GiB free space, and no active builder/trainer.
2. Preserve the original `.pt`, sidecar, and completion marker under immutable audit names (or verified same-filesystem hard links) and record their SHA-256 values.
3. Load the original on CPU and recompute the five tensor hashes above.
4. Assert the stale value occurs exactly 1,000 times at universe level and 5,000 times at snapshot level; reject every other difference.
5. Change only those 6,000 values to the generated label, write a new temporary `.pt`, fsync it, and reload it.
6. Recompute tensor hashes and require exact equality for `X`, `edge_index`, masks, targets, and the combined hash.
7. Copy the sidecar, change only `preprocessing_version`, update its dataset checksum, and preserve the original sidecar as the requested audit backup. Do not change the source manifest or target CSV.
8. Atomically publish the migrated dataset/sidecar/completion marker, then run the dedicated validator and a record-count audit. Any mismatch aborts publication and retains the original artifact.

## Exact next action

Create reviewed split manifests against dataset checksum `6a794c38216175212ec969283fdbd214bcdc64caecde78d24b81545e681a332a`. Do not migrate or rebuild the dataset for training. Treat structured protocol fields—not the legacy string—as authoritative, and retain this audit with the split authorization record.
