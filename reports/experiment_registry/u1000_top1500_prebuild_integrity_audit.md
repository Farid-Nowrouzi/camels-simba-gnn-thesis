# Comprehensive pre-build integrity audit: U1000 Top1500 sparse temporal graph dataset

Audit date: 2026-08-06 UTC
Repository: `/home/ml/thesis-camels`
Mode: inspection and bounded tests only

## Final decision

**READY AFTER SPECIFIC FIXES.** Raw identity, targets, sequences, features, Top1500 selection/padding, pooling masks, periodic sparse kNN, builder command parity, atomic publication, resources, partitions, configurations, and current registry state pass. The build must not be authorized yet because:

1. `scripts/validation/manage_u1000_top1500_training_scaling_matrix.py` treats any non-`PENDING_POST_BUILD` binding as acceptable. Matrix preflight does not recompute and compare the current `.pt`, metadata, completion-marker, source-manifest, config, registry, and all 18 manifest hashes.
2. The same preflight accepts a CUDA pilot JSON solely when `status == "PASS"`; it does not require that the pilot's dataset and seed42/Train700 manifest hashes equal the current artifacts. A stale or fabricated PASS file can therefore cross the intended pilot gate.
3. Builder metadata records the builder Git commit and derived preprocessing version but not an explicit builder module/path. The requested builder-path provenance field is absent from `src/data/build_temporal_sequences.py` and is not required by the validator.

These are repairable integrity defects. They do not indicate a scientific or resource problem in the intended Top1500 dataset.

## 1. Repository and process safety

| Check | Result |
|---|---|
| Branch | `thesis-sparse-integrity-hardening` (PASS) |
| HEAD | `461430ba5bd1abbad3b218ce4d828c2e77f55397` (exact expected commit; PASS) |
| Staged changes | none |
| Tracked worktree modifications | 18 pre-existing files; no Top1500 builder/model/validator source is modified |
| Scientific-path difference from preparation commit | `scripts/analysis_reporting/analyze_u1000_top1000_training_scaling.py` and the Top1000 matrix registry config are modified as unrelated Notebook 11/Top1000 analysis work; builder/trainer/model/Top1500 preparation paths match HEAD |
| Untracked work | pre-existing completed Top1000 experiment/report artifacts; preserved |
| Notebook 11 work | present and preserved, including notebook, analysis script/reports, presentation artifact, registries, and experiment outputs |
| Active builders | 0 |
| Active EvolveGCN-H trainers | 0 |
| Active Static GCN trainers | 0 |
| Active matrix launchers | 0 |
| Active CUDA jobs | 0; `nvidia-smi` cannot communicate with the driver |

Active tmux sessions observed: `canonical-knn-hidden-factorial-r3` (dead pane), `canonical-scaling`, `u1000_scaling`, `u1000_top1500_scaling`, and three dead pilot panes (`u1000_train700_seed42`, `_r2`, `_r3`). The Top1500 session is an idle bash shell, not a launcher or trainer. No process was killed or signalled.

The accidental command transcript is exact:

```text
bash scripts/production/run_u1000_top1500_training_scaling_matrix.sh --resume
TOP1500 BUILD REQUIRED BEFORE CUDA: missing data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt
```

The repeated safe test returned exit code 3 at the same gate. It occurs before log-directory creation, state-file creation, registry mutation, experiment-directory creation, or trainer launch. There are no Top1500 experiment files, checkpoints, predictions, build logs, processed outputs, or partial directories. Registry state remains planned=36, running=0, completed=0, failed=0.

## 2. Recovered Top1000 production history

The authoritative production anchor is:

- dataset: `data/processed/temporal_1000u_none_top1000_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1000_periodic_knn_sparse.pt`;
- dataset SHA-256: `6a794c38216175212ec969283fdbd214bcdc64caecde78d24b81545e681a332a`;
- metadata SHA-256: `d4ea0ba0c3a1abc6f49d6856be86c7fc1226090daac8924eb6b72262d22753b9`;
- completion-marker SHA-256: `4eea1a4bbbfc57d0c3420a115ae436240e0dcb1588cf47588ab2ee5809edd85a`;
- launcher: `scripts/production/run_u1000_top1000_sparse_build.sh`;
- retained build log: `logs/dataset_builds/u1000_top1000_sparse_build_20260805T204219Z.log`;
- monitor: `scripts/production/monitor_u1000_top1000_build.sh`;
- validator: `scripts/validation/validate_u1000_top1000_sparse_dataset.py`;
- metadata and marker: the two sidecars adjacent to the anchor dataset.

Safeguard evidence also includes:

- raw/identity: `u1000_top1000_raw_data_completeness.csv`, `u1000_top1000_halo_count_feasibility.csv`, `u1000_top1000_population_and_role_plan.csv`;
- targets: `u1000_target_table_creation_audit.md`, `u1000_target_table_production_readiness.md`, `u1000_target_table_snapshot_consistency.csv`, `u1000_target_table_statistics.csv`;
- sparse/graph/periodic: `sparse_u1000_integrity_hardening_audit.md`, `sparse_u1000_integrity_regression_report.md`, `sparse_u1000_equivalence_test_report.md`, `sparse_u1000_schema_specification.md`, `u1000_top1000_sparse_realdata_benchmark.csv`;
- source provenance: `u1000_top1000_source_manifest_policy_review.md`, the embedded full source manifest, and `u1000_top1000_postbuild_metadata_consistency_audit.md`;
- post-build/stale label: `u1000_top1000_post_sparse_build_gate_audit.md`, `u1000_top1000_postbuild_metadata_field_matrix.csv`, `u1000_top1000_postbuild_metadata_consistency_audit.md`;
- CUDA: `u1000_top1000_cuda_pilot_audit.md`, `u1000_top1000_cuda_pilot_metrics.csv`, `u1000_top1000_cuda_pilot_production_readiness.md`;
- partitions: `u1000_top1000_nested_split_manifest_creation_audit.md`, `u1000_top1000_nested_split_manifest_validation.csv`, `u1000_top1000_split_manifest_inventory.json`, `u1000_top1000_split_manifest_production_readiness.md`;
- matrix/registry: `u1000_top1000_training_scaling_matrix_preparation.md`, `u1000_top1000_training_scaling_matrix_preflight.md`, and `u1000_top1000_training_scaling_matrix_registry_status.csv`.

The parity CSV defines 20 safeguards: 18 pass for the Top1500 preparation and two fail at the shared binding/pilot preflight defect.

## 3. Universe identity and five-snapshot sequence

- Exact canonical population: `LH_0` through `LH_999`.
- Unique IDs: 1,000; missing: 0; duplicate: 0; malformed: 0; outside-range: 0.
- Raw catalogue names matching the exact identity/snapshot pattern: 5,000/5,000.
- Storage order is numerical, not lexical: `LH_0, LH_1, ..., LH_999`.
- Raw layout is one flat root with filenames `LH_<integer>_hlist_<snapshot>.list`; there is no alternative directory identity.
- Target table uses canonical `LH_<integer>` strings plus the matching integer `universe_index`.
- Top1000 output keys are exactly the same ordered IDs. Top1500 builder loops `range(1000)` and creates the same keys.
- Existing manifests use canonical strings, not bare integers.

Required ordered snapshots are exactly:

```text
0.20000, 0.25000, 0.51209, 0.75065, 1.00000
```

Every universe has exactly those five files. Complete universes: 1,000; incomplete: 0; duplicate sequences: 0; order anomalies: 0; substituted or mismatched paths: 0. The builder sorts parsed scale factors. Static conversion takes index `[-1]` by exact object reference, so it consumes the same `1.00000` final graph used by EvolveGCN-H. No universe is silently dropped because production does not pass `--allow_partial` and the builder checks 1,000 successes.

The complete 1,000-row inventory is `u1000_top1500_universe_identity_inventory.csv`; every row has anomaly flag `NONE` and includes all required paths/counts plus Train700 roles for seeds 42, 123, and 2025.

## 4. Target integrity

Authoritative source: `outputs/target_inspection_1000u.csv`.

| Property | Result |
|---|---:|
| Rows / unique IDs | 1,000 / 1,000 |
| Missing / duplicate / nonfinite | 0 / 0 / 0 |
| Snapshot-constancy failures | 0 |
| Minimum / maximum | 0.1002 / 0.4998 |
| Mean / median | 0.3000 / 0.3000 |
| Population / sample standard deviation | 0.1154699961 / 0.1155277744 |
| SHA-256 | `9692a97760ee0e3a97cf3293f1b73911ee0a1af028617f03fe85f88f431703c2` |

All 1,000 serialized Top1000 float32 targets agree within 1e-6. All 4,824 truth rows found across 24 completed Top1000 prediction/report CSVs also agree within 1e-6. The immutable split manifests contain the same canonical universe IDs and target summaries derived from this table.

The historical `v2_logmass_minmax_top100_periodic_knn` string in the target table and old Top1000 serialization is stale provenance context only. The dedicated post-build audit proves it is not consumed by loaders/models and separately validates the structured fields. Current code derives a `v3_logmass_none_top1500_periodic_knn_k8_box25_sparse_edge_index` label for the future build.

## 5. Raw catalogues, features, and selection

The embedded Top1000 full source manifest was independently verified against current bytes by the permitted launcher preflight:

- entries: 5,001 (5,000 halo catalogues plus one target table);
- raw read failures: 0;
- full source-manifest SHA-256: `ba22c3611a70763566ffb38a20f9b5a36fb6c1a27c3ad8030c4a7e189ce87618`.

The raw header and implementation agree:

| Quantity | Column | Units / handling |
|---|---|---|
| stable halo ID | `col_1` | Rockstar halo ID; ascending tie key |
| Mvir | `col_10` | Msun/h; must be finite and >0; raw value ranks nodes |
| X/Y/Z | `col_17..19` | Mpc/h comoving; sampled range lies in [0,25] |
| VX/VY/VZ | `col_20..22` | km/s physical peculiar |
| fallback order | `_original_row_index` | deterministic input-row index |

Cleaning replaces nonfinite required fields with NaN, drops them, and drops nonpositive Mvir before selection. Selection is stable mergesort by raw Mvir descending and stable halo ID ascending; mergesort preserves original row order as the final fallback for identical mass and ID. A tie fixture returned IDs `[3,5,8,1]` for masses `[10,10,10,9]`. No random operation appears in selection.

Only after selection is Mvir transformed with `log10`; the final float32 feature order is exactly `[log10_Mvir, X, Y, Z, VX, VY, VZ]`. `normalization=none` returns these float32 values unchanged. Four deterministic raw samples (including the worst padded and the 9,795-halo high-count catalogue) had 85 columns, finite features, positive mass, coordinates inside the periodic box, plausible velocities, and exact `[1500,7]` output.

## 6. Top1500 padding and mask gate

The existing full raw-count audit was recomputed from its 5,000 rows:

| Property | Result |
|---|---:|
| Snapshots with at least 1,500 real halos | 4,889 |
| Snapshots needing padding | 111 |
| Affected universes | 89 |
| Total padded slots / total slots | 66,313 / 7,500,000 |
| Padded fraction | 0.8841733333% |
| Worst snapshot | LH_418, 0.20000: 157 real, 1,343 padded |
| Largest universe totals | LH_418: 1,974; LH_522: 1,837; LH_802: 1,769 |

Padding is explicit zero-filled float32 data with mask shape `[1500,1]`; real rows form a contiguous prefix and `num_real_nodes` plus `selected_num_halos_before_padding` are stored in snapshot metadata. Sparse edge construction uses only positive-mask indices and bounds every endpoint below the real-node count.

Mask use is effective, not merely serialized. `collate_sparse_static` removes every padded row, remaps edges, rejects padded endpoints, and constructs batch vectors only for real nodes. Temporal collation applies this to all five snapshots. Consequently `sparse_graph_pool` divides by the real-node batch count, and both Static and Evolve sparse paths pool only real nodes. Temporal aggregation operates on five correctly pooled graph embeddings. **Pooling-mask gate: PASS.**

## 7. Periodic kNN and sparse schema

The exact Top1000/Top1500 implementation uses box size 25.0 on all three XYZ axes. For each source it computes displacement to all real positions, applies `min(abs(diff), 25-abs(diff))`, squares/sums, excludes self with infinity, chooses `min(8,R-1)`, adds both directions, deduplicates in a set, and lexicographically sorts final `(source,target)` pairs. Selected rank provides deterministic distance-tie ordering.

Bounded tests passed:

- boundary crossing (`0.1` and `24.9`) selects the wrapped neighbor;
- ordinary non-boundary nearest neighbors are correct;
- three nodes with k=8 yield all six directed non-self pairs;
- padded nodes never appear;
- repeated calls are byte-identical;
- edges are symmetric, unique, self-loop-free, and bounded by real count.

Sparse implementation retains `[R,3]` and `[R]` per-source intermediates only. Dense distance/adjacency functions remain for the legacy dense branch but are unreachable when `graph_storage=sparse_edge_index`; neither model converts sparse graphs back to dense.

Schema parity with Top1000 is exact except N=1500: ordered universe dictionary; five `Nodes_list`, `edge_index_list`, and `mask_list` entries; x float32 `[1500,7]`; edge_index int64 `[2,E]`; mask float32 `[1500,1]`; target scalar float32; snapshot metadata; no dense adjacency; no internal split. Evolve consumes all five snapshots. Static exposes the exact final x/edge/mask/target/metadata objects.

## 8. Builder command parity

The validated Top1000 log and the Top1500 `--preflight-only` output show identical builder options except the intended dataset identity changes:

| Field | Top1000 | Top1500 | Result |
|---|---|---|---|
| raw root / target source | `CAMELS_SIMBA_1000U` / authoritative CSV | same | MATCH |
| universe/snapshot counts | 1000 / 5 | same | MATCH |
| Top-N | 1000 | 1500 | EXPECTED |
| output path/name | Top1000 sparse path/name | Top1500 sparse path/name | EXPECTED |
| normalization/features | none / seven fixed features | same | MATCH |
| graph | kNN k=8 periodic box=25 | same | MATCH |
| storage/provenance/device | sparse_edge_index / full_sha256 / cpu | same | MATCH |
| ordering/partial/overwrite | deterministic defaults / false / false | same | MATCH |

Exact Python command difference is limited to `--num_nodes 1000 -> 1500` and the output path. Metadata, marker, and log names are derived from that output and use Top1500-specific names. The launcher itself lowers the disk gate from 5 GiB to 4 GiB and adds direct verification against the Top1000 source manifest; neither changes scientific output. The 4 GiB floor is sufficient, though 5 GiB parity is preferable.

## 9. Atomicity, overwrite, and recovery

Expected final paths:

```text
data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt
data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.metadata.json
data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.complete
```

Expected temporary/lock patterns use an unpredictable UUID token:

```text
.../.camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt.<uuid>.tmp
.../.camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.metadata.json.<uuid>.tmp
.../camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt.lock
```

The writer takes an exclusive lock, serializes and fsyncs the temp dataset, validates the in-memory dataset, computes SHA-256, writes/fsyncs temp metadata, atomically replaces data and metadata, fsyncs the directory, and writes the completion marker last. The launcher preserves Python exit status through `tee`, retains failure logs, refuses any complete/partial/lock/temp artifact, and never passes `--overwrite`. Top1000 is in a distinct directory and cannot be targeted by the Top1500 command. Python exceptions and normal signals clean unique temps/lock; SIGKILL can leave them, which preflight preserves and reports. Rerun policy is manual inspection/removal or explicit recovery—not silent overwrite.

The marker is published after the builder's full internal schema validation; the stronger Top1500 raw-count validator runs immediately afterward and again in matrix launch. A failed external validation therefore leaves a marker but cannot pass the dataset validator. The absence of a post-marker directory fsync is a low power-loss durability caveat inherited from Top1000.

## 10. Resources

Current available disk is 105,979,060,224 bytes (approximately 99 GiB). RAM is 62.5 GiB total and 58.7 GiB available; swap is zero; CPU count is 16. The CPU-only builder sees no usable NVIDIA driver.

Top1000 size is 1,047,033,642 bytes and measured production elapsed time is about 49m47s. Top1500 estimates are final `.pt` 1.50–1.65 GB (central 1.57 GB), incremental temporary peak about 1.65 GB, peak host RAM about 4–7 GiB, and duration about 60–100 minutes. Full-snapshot distance work is at most 2.25× Top1000, but catalogue parsing and source hashing do not scale quadratically. Resource classification: **LOW**. See the dedicated resource report.

## 11. Provenance and output identity

The build will record dataset checksum, full raw/target manifest, target checksum, Git commit, derived effective protocol, 1,000 ordered IDs/hash, five snapshot IDs, feature order, Top-N, k/periodicity/box, padding/edge/target statistics, timestamp, Python/PyTorch versions, and builder config hash. Marker identity binds dataset name, metadata name, status, and dataset SHA; the post-build binding tool also computes metadata and marker SHA-256.

The Top1500 validator requires metadata Top-N=1500 and independently scans every x tensor for shape `[1500,7]`, so a Top1000 tensor cannot masquerade as Top1500. It also binds padding counts to the 5,000-row raw audit.

**Provenance defect:** metadata has no explicit `builder_path` or builder-module field. Git commit plus preprocessing version identify code state but do not satisfy the requested exact builder path. Add and validate an immutable value such as `src/data/build_temporal_sequences.py`.

## 12. Split manifests, configurations, and registry

All 18 Top1500 manifests pass:

- exact ordered train/val/test/unused IDs copied from Top1000 for six levels and three seeds;
- zero reshuffles, missing IDs, duplicates, or overlaps;
- fixed validation=99 and test=201;
- Train20 ⊂ Train50 ⊂ Train100 ⊂ Train200 ⊂ Train450 ⊂ Train700 as ordered prefixes;
- Train700 + validation99 + test201 = 1,000 and unused=0;
- Evolve and Static reference the same manifest per seed/level;
- only dataset-specific hashes remain `PENDING_POST_BUILD`.

Partition-only SHA-256 checksums, in seed/level order, are recorded in the readiness JSON. Each Top1000/Top1500 pair produced the same checksum.

Prepared configurations: 36 unique cells = 18 EvolveGCN-H + 18 Static GCN. Every seed/level/model combination exists; dataset path is Top1500; four dataset/source hash fields are pending; node normalization and target normalization are none; architecture/hyperparameters match the Top1000 templates; experiment names/directories and canonical IDs are unique. No Top1000/Top500 path occurs in a Top1500 config.

Registry: planned=36, running=0, completed=0, failed=0, duplicate IDs/cells=0, and no runtime/checkpoint/prediction fields populated.

Pending values correctly fail current matrix preflight. However, replacing pending fields with arbitrary non-pending strings can pass `validate_preparation`, and matrix preflight does not bind them back to actual files. This violates the requirement that no fake/placeholder hash can pass.

## 13. Post-build gate enforcement

The intended order exists: build → marker → dataset validator → bind → CUDA pilot → matrix preflight → sequential training. `--resume` cannot bypass the initial missing-dataset check, dataset validator, manager preflight, or dirty-scientific-tree check.

Passes:

- missing dataset blocks before mutation;
- missing/invalid marker or failed dataset validator blocks;
- literal pending dataset hashes block;
- missing pilot file or non-PASS status blocks;
- trainer-side manifest loading enforces seed, canonical manifest hash, ordered partitions, exact population coverage, and the supplied dataset identity.

Fails:

- manager preflight does not compare bound dataset/metadata/marker/source hashes with actual artifacts;
- it does not compare pilot dataset/manifest hashes with actual current artifacts;
- trainer-side identity compares manifest with the caller-supplied config identity, not independently recomputed bytes.

Therefore `--resume` cannot skip named stages accidentally, but crafted/stale binding artifacts can satisfy them. This is a pre-build integrity blocker under the requested decision rules.

## 14. Lightweight test record

Passed:

- four Top1500 Python source compilation checks;
- `bash -n` for four Top1500 production/monitor scripts;
- builder, monitor, dataset validator, CUDA pilot, manager, and matrix launcher help;
- build launcher `--preflight-only`, including byte-exact full source verification;
- matrix launcher `--status`;
- matrix launcher `--preflight-only` at expected missing-dataset exit 3;
- exact 1,000-ID/5,000-catalogue assertions;
- exact five-snapshot sequence assertions;
- target statistics/hash/Top1000/prediction truth assertions;
- all 18 manifest partition and nestedness assertions;
- all 36 configuration/cell and registry uniqueness assertions;
- boundary, ordinary, fewer-than-k, padded-node, and deterministic periodic-kNN fixtures;
- four real-catalogue Top1500 selection/feature/mask/edge samples;
- 14/14 sparse graph pipeline unit tests;
- manager `--validate-configs`;
- `git diff --check`.

Failed integrity assertions: two conceptual tamper cases identified by direct code inspection—fake non-pending binding and stale/fabricated PASS pilot are not rejected. No permitted command unexpectedly failed.

## 15. Exact minimal fixes

Do not build until these are reviewed:

1. In `scripts/validation/manage_u1000_top1500_training_scaling_matrix.py`, make `preflight()` call the existing validator, recompute `dataset_binding()`, and require exact equality for dataset, metadata, marker, source-manifest, target-table, logical ID, schema, and storage across every registry entry, config, and split manifest. Validate 64-lowercase-hex syntax and reject all placeholders.
2. In the same `preflight()`, recompute the seed42/Train700 manifest SHA and require the CUDA pilot JSON's `dataset_sha256` and `manifest_sha256` to equal current artifacts; require both model subrecords and safety counters to pass.
3. In `src/data/build_temporal_sequences.py`, add explicit immutable builder module/path metadata. In `scripts/validation/validate_u1000_top1000_sparse_dataset.py` (inherited by the Top1500 wrapper), require its exact value.
4. In `tests/test_u1000_top1500_preparation.py`, add negative preflight tests for a fake non-pending dataset hash, stale metadata/marker/source hash, stale pilot dataset hash, stale pilot manifest hash, and missing/incorrect builder path.

Optional parity improvement: raise `MIN_FREE_KIB` in `run_u1000_top1500_sparse_build.sh` from 4 GiB to the Top1000 launcher's 5 GiB. This is not required by measured resources.

## Audit safety confirmation

No production graph build, tmux creation, CUDA pilot, trainer, epoch loop, optimizer step, checkpoint, prediction, split regeneration, binding, source modification, processed-dataset modification, raw/target modification, staging, commit, push, or dependency installation occurred. The only persistent writes are the five requested reports.
