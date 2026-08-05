# U1000 Top1000 pre-build control audit

Audit date: 2026-08-05. Repository commit `329ab5c3abe3ad911331da465bc015e9f120d75e`; branch `thesis-controlled-scaling`. Inspection only: no graph dataset was built, no production split was created, no model/checkpoint was loaded, and no training was run.

## 1. Executive summary

**Decision: NO-GO under the requested criteria.** Raw data, targets, population size, snapshot protocol, aggregate Top1000 padding, fixed seed-specific validation/test reuse, and Static final-snapshot reuse are feasible. The current canonical pipeline is nevertheless dense end-to-end: preprocessing allocates `[N,N,3]` displacement and `[N,N]` distance/adjacency arrays; serialization stores five dense `[1000,1000]` float adjacencies per universe; both trainers collate dense matrices; both models normalize and multiply dense adjacency. The U750 anchor is 15,125,161,534 bytes and projects to about 20.17 GB for U1000 before Python/RAM overhead. This violates the explicit no-critical-O(N²) GO criterion. EvolveGCN-H also lacks the Static trainer's external split-config interface, and the builder writes its final output non-atomically with no resume or completion marker.

Exactly one next action: **refactor and validate the graph pipeline to a deterministic sparse `edge_index` schema—including builder, both trainers/models, split-config support for Evolve, and atomic output—then rerun this pre-build audit before creating U1000.**

## 2. Scientific question

How does Train20/50/100/200/450/700 affect Omega_m regression when every universe uses the identical CAMELS-SIMBA Top1000, five-snapshot, no-node-normalization, periodic-kNN k=8 protocol, with fixed validation/test universes; secondarily, how do EvolveGCN-H and Static GCN scale when Static consumes the same temporal sequence's final graph? Cross-model results are descriptive.

## 3. Current U750 Top1000 anchor

- Dataset: `data/processed/temporal_750u_none_top1000_periodic_knn/camels_750u_temporal_logmass_none_top1000_periodic_knn.pt`; SHA-256 `97acf26f2b3e767450c17c1925fb7c6c48cf371a061bb473286de53c4ac5480b`; size 15,125,161,534 bytes.
- Sidecar: same stem `.metadata.json`; SHA-256 `4533d80ba18a8676ccc74795635a3fbbc47c284eb6f6c48847d4b9f44fc3c91f`.
- 750 universes `LH_0..LH_749`; five snapshots; Top1000; none; periodic kNN k=8; box 25; seven features; raw Omega_m.
- Five linear-head seeds exist (42, 123, 777, 999, 2025); planned initial seeds are 42, 123, 2025. Configured 450/99/201 splits.
- Checkpoints exist (about 40.9 MB each) but were not opened. Metrics, logs and predictions exist.

## 4–6. Raw availability, five snapshots, and targets

Canonical raw root: `data/raw/CAMELS_SIMBA_1000U`. Exactly 5,000 well-formed unique catalogue names cover `LH_0..LH_999` and `0.20000, 0.25000, 0.51209, 0.75065, 1.00000`. Missing universes/files/snapshots, duplicate names within this root, malformed names, and suspicious empty names: zero. `CAMELS_SIMBA_750U` contains 3,750 hard links to the same device/inodes as the corresponding U1000 files; it is not a second source population.

All five headers for every universe contain one finite, consistent `Omega_M`; 1,000/1,000 targets complete. No separate U1000 target CSV exists yet; the future builder needs a lightweight `outputs/target_inspection_1000u.csv` derived from these headers.

## 7–8. Top1000 and diagnostic Top1500/Top2000 feasibility

Across 5,000 snapshots, valid rows equal total, finite-Mvir, and positive-Mvir rows in every file. Overall valid count: min 157, max 9795, mean 6097.398, median 6415.0, population SD 1722.202; P1/P5/P25/P50/P75/P95/P99 = 898.76/2415.70/5350.00/6415.00/7259.25/8346.10/8943.09.

| snapshot a | minimum | mean | median | below 1000 | mean Top1000 padding |
|---|---:|---:|---:|---:|---:|
| 0.20000 | 157 | 4703.0 | 4737.0 | 57 | 22.203 |
| 0.25000 | 869 | 5919.4 | 6202.5 | 4 | 0.356 |
| 0.51209 | 3005 | 6963.2 | 7237.0 | 0 | 0.000 |
| 0.75065 | 3715 | 6430.8 | 6449.5 | 0 | 0.000 |
| 1.00000 | 4256 | 6470.5 | 6460.5 | 0 | 0.000 |

Top1000: 61/5,000 snapshots (1.22%) and 57/1,000 universes (5.7%) need padding; 22,559/5,000,000 node slots are padding (0.45118%); max 843 and mean 4.5118 padded nodes/snapshot. The severe tail is confined mainly to a=0.2 and must be reported, but masks make aggregate padding acceptable.

Top1500 diagnostic: 4,889/5,000 (97.78%) meet the count; mean padding 13.2626, max 1,343, 66,313 padded slots (0.88417%). Top2000: 4,820/5,000 (96.40%) meet it; mean padding 27.6908, max 1,843, 138,454 slots (1.38454%). No build is proposed for either.

## 9–11. Top-N, feature provenance, and normalization

`camels_graph_utils.py` maps header-confirmed Rockstar columns: Mvir=`col_10`; X/Y/Z=`col_17..19` (Mpc/h comoving); VX/VY/VZ=`col_20..22` (km/s physical peculiar). Cleaning drops nonfinite values in all seven required columns and Mvir<=0. Selection sorts raw Mvir descending and takes a prefix; log10 is applied only after selection. Raw XYZ is copied separately before feature normalization. `normalization='none'` returns float32 features unchanged after log10 mass. Padding happens after feature construction; mask 1=real, 0=padded; kNN sees only real indices.

Risk: `DataFrame.sort_values` does not request a stable algorithm or secondary halo-ID key. Equal Mvir boundary membership is therefore not cross-version deterministic. Smallest safe fix: stable lexicographic `(Mvir desc, halo ID col_1 asc)` and persisted selected-ID/rank hashes. kNN distance ties likewise need an explicit secondary rank/ID key.

## 12–13. Periodic kNN and dense-allocation risk

The code computes minimum-image component separation `min(abs(dx), 25-abs(dx))`, Euclidean distance, sets the diagonal infinite, selects `min(k, real_nodes-1)`, and symmetrizes every neighbor choice. Preprocessing has no self-loops; model self-loops are added later. Duplicate edges collapse in binary adjacency; padded nodes are disconnected; no edge attributes exist. Thus k=8 means about 8,000 directed selections for 1,000 real nodes, but the stored symmetric nonzeros are typically about 10,000 (U750 validation mean 10,015.94; union bounds approximately 8,000–16,000). Graph order is chronological by numeric scale factor and final `[-1]` is a=1.0.

Complexity: catalogue/filter/padding O(N); sort O(N log N); neighbor selection currently O(N² log N) because each full distance row uses `np.argsort`; displacement O(N²) memory; dense adjacency O(N²) memory; dense training normalization/matmul O(N²) storage/work per layer. **Critical dense risk confirmed.** Sparse topology is not used anywhere in build, load, collate, or either model.

## 14. Existing U750 dataset identity

The sidecar and previous validation establish schema/protocol and the U750 file hash above. The old raw files are hard links to the first 750 files in the U1000 root, but exact selected halo IDs and graph hashes were never stored and this audit did not deserialize the `.pt`. Consequently per-graph identity is likely, not proven. A sparse refactor changes storage representation and invalidates literal byte equivalence even if mathematical edges/features match.

## 15–16. U1000 population and fixed validation/test reuse

The exact population is `LH_0..LH_999`. For each seed, preserve its ordered historical 99 validation and 201 test IDs; these partitions are seed-specific, not identical across seeds. All are complete, target-valid, and present in U1000. The 450 historical train IDs plus `LH_750..LH_999` form a disjoint 700 maximum pool and cover all 1,000 with fixed val/test.

| seed | Train450 | val | test | val hash prefix | test hash prefix |
|---:|---:|---:|---:|---|---|
| 42 | 450 | 99 | 201 | b3504d7eda16 | bb0ebc3a94dc |
| 123 | 450 | 99 | 201 | 2e73c457617b | 1c9a2a141e32 |
| 2025 | 450 | 99 | 201 | 0b147c27abbc | ff6a0b1b9c04 |

## 17. Nested training-subset design

No repository utility supports fixed validation/test plus nested Evolve subsets. The planning CSV defines each seed's historical ordered Train450 prefix levels 20/50/100/200, then appends numeric `LH_750..LH_999` for Train700. This guarantees T20⊂T50⊂T100⊂T200⊂T450⊂T700, exact Train450 order, fixed ordered val/test, no overlaps, deterministic newline-ID SHA-256 hashes, target summaries/quantiles, and explicit introduced IDs. The later minimal utility should accept `(anchor_config, extension_ids, levels, target_map, output_dir)`, validate completeness/disjointness/nesting/coverage, and emit one immutable JSON manifest per seed/level plus summary/hash manifest. It must not optimize using test outcomes.

## 18. Existing run reuse

The 21 U750 Top1000 candidates are classified in the reuse CSV. The exact linear mean/mean rows for seeds 42/123/2025 are `likely_reusable_pending_dataset_identity`, never unconditionally reusable: their ordered Train450/val/test and optimizer/model protocol match, but graph identity was not proven and the required sparse redesign changes representation. Seeds 777/999 are optional/excluded from the initial matrix. MLP, mean-max, temporal-last and the duplicate reproduction are incompatible. There is no historical Static Top1000 row.

## 19. Evolve trainer compatibility

Current shapes are dense A `[B,T,N,N]`, X `[B,T,N,7]`, mask `[B,T,N,1]`, target `[B,1]`; the full dataset is loaded eagerly. It handles masks, five snapshots, graph mean, temporal mean, linear head, batch 4 and unnormalized targets. It cannot accept external split IDs: it always shuffles the entire dataset from seed/ratios. Required changes: sparse temporal sample/collate/model path; validated `--split_config_path` identical to Static; collision-safe outputs; config records split source/hash; deterministic worker policy (currently DataLoader default `num_workers=0`).

## 20–21. Static compatibility and final-snapshot decision

**Classification A — direct reuse supported in the current dense schema.** `train_static_gcn.py` already exposes `--dataset_format temporal_final_snapshot`; `convert_temporal_final_snapshot_to_static` maps `A_list[-1]`, `Nodes_list[-1]`, `mask_list[-1]`, and the same target by reference, without reselecting, renormalizing, or rebuilding kNN. It also supports `--split_config_path`. Expected current shapes: A `[1000,1000]`, X `[1000,7]`, mask `[1000,1]`, scalar target; final metadata is a=1.0. The wrapper currently omits snapshot metadata in the returned view, so preserve/validate universe ID and `snapshots[-1]` in the sparse revision. Recommended fixed Static anchor: h32, 3 layers, batch 8, mean pooling, GCN, fixed `32→32→16→1` MLP readout, dropout .2, AdamW lr .001/weight decay 1e-5, MSE, 300 epochs, patience 40, clip 1.0, model self-loops/layer norm/residual defaults. No independent static file is needed.

## 22–24. Storage, runtime, and memory

Filesystem: 197 GiB capacity, 88 GiB used, 101 GiB free (47%). Repository 58 GiB; raw 21 GiB; processed 23 GiB; experiments 14 GiB; reports 64 MiB; outputs 97 MiB. The dense U750 anchor is 15.125 GB decimal. Linear ratio predicts 20.167 GB; conservative dense bounds 18.8–22.5 GB. Atomic build plus validation/checksum should reserve 45–50 GB temporary/safety space. Current disk is sufficient; archival is not required for space, though old artifacts may be archived by policy.

Historical file timestamps provide only approximate wall evidence: seed42 13m10s for 111 logged epochs, seed123 21m32s/189, seed2025 9m54s/92; config-to-log implies about 6.4–7.0 s/epoch, followed by roughly 4.5 minutes of final three-split prediction. No explicit runtime telemetry was logged. Train700 epoch work is approximately 700/450=1.556x Train450. The six Evolve levels total 1,520 training universes per seed, 3.378 Train450 epoch-equivalents; actual early stopping and fixed evaluation dominate small levels. Static timing is not recorded, so no absolute Static/full-matrix time is claimed.

Historical resource record names an NVIDIA L40 with 46,068 MiB and 31 GiB host RAM (14 GiB then available). Current environment reports PyTorch 2.6.0+cu124 but CUDA unavailable and nvidia-smi inaccessible; PyG is not installed and is not used. Peak GPU/CPU memory and dataset-load time were not logged. Dense U1000 serialization alone is ~20.2 GB before Python objects/copies; eager load/collation is unsafe on the recorded 31 GiB host. Sparse refactoring is mandatory.

## 25. Failure recovery

Current builder accumulates all universes in memory, calls `torch.save(dataset, output_path)` directly, then writes metadata. It has no existing-output guard, resume, shard/temp path, atomic rename, lock, checksum, or completion marker. Failure before save leaves no reusable progress; interruption during save can leave a corrupt final-named file; rerun overwrites it.

Future workflow: preflight → exclusive lock → deterministic shards or temp output on same filesystem → build/validate → fsync → atomic rename → metadata plus `.complete` marker → SHA-256 → read-only smoke test. Never treat a file without matching metadata/checksum/completion marker as complete.

## 26–27. Required future code and files

Code changes before build: sparse periodic kNN and serialization in `src/data/camels_graph_utils.py` and `build_temporal_sequences.py`; sparse inputs/message passing in both model files; sparse collates and external split config in Evolve trainer; metadata/hashes/atomic build; output collision guards. Static's final-snapshot view should be retained and metadata-aware.

Future artifacts (not created now): `data/processed/temporal_1000u_none_top1000_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1000_periodic_knn_sparse.pt` and sidecar/completion/checksum; `outputs/target_inspection_1000u.csv`; one reviewed dataset-build manifest; 18 immutable seed/level split JSONs plus a split summary; 36 experiment configs/directories. Metadata must include dataset/source IDs, ordered universe/snapshot hashes, feature/selection/normalization/k/periodic/box protocol, node/padding/edge/target stats, builder commit/config hash/timestamp, source manifest hash, graph/selected-halo hashes, schema version, and split-design reference.

## 28–29. Pilot and matrix

After blockers are fixed, run exactly: Train700 Top1000 none seed42 EvolveGCN-H, then Static GCN on the same temporal final graph. Success: no OOM; finite losses; exact fixed test IDs/order; metrics recompute; nonconstant predictions; graph/split hashes match; runtime and peak host/GPU memory recorded. Failure: OOM, nonfinite values, graph/hash mismatch, wrong split/order, constant output, incomplete artifacts, or resource margin breach.

Maximum matrix is 36 cells. Thirty-three are definitely new; three Evolve Train450 rows are conditional reuse candidates. If post-refactor mathematical graph equivalence and protocol identity cannot be established, all 36 are new. Optional seeds 777/999 are outside this matrix.

## 30. Risks and mitigations

See `u1000_top1000_risk_register.csv`. Critical blockers are dense O(N²), eager RAM load, absent Evolve split injection, and non-atomic build. Controlled scientific cautions are early padding tail and unstable selection/tie rules. Leakage checks pass by source inspection: target is only the sample label; none of the seven node columns is a simulation parameter/target; topology uses XYZ only; no transform is fit; summaries/hybrids are disabled in the anchor; and splits are disjoint. Historical validation/test prediction CSVs exactly follow their configured ordered IDs. Train prediction membership is exact and unique, but its rows are shuffled because the same `shuffle=True` training loader is reused for final prediction; future exports should use a separate non-shuffled train-evaluation loader when ordered train predictions are required.

## 31–32. Decision and highest-value next action

**NO-GO.** Raw completeness is not the blocker. The current dense graph implementation and eager loading are explicitly disallowed by the GO criteria and unsafe at U1000; split and build-recovery gaps are also unresolved.

**Exact next action:** refactor and validate the graph pipeline to a deterministic sparse `edge_index` schema—including builder, both trainers/models, Evolve split-config support, and atomic output—then rerun this audit before building.

## 33–34. Git and non-execution confirmation

Pre-existing tracked modifications: `notebooks/visualization/10_final_experiment_report.ipynb` and `reports/meeting/assets/10_final_experiment_report/prediction_collapse/prediction_collapse_protocol_comparison.pdf`. Numerous pre-existing untracked experiment/report artifacts were present. This task creates only the twelve named U1000 audit reports. `tmux ls` failed with socket permission denied; this is environmental, not scientific. No active matching trainer was observed. No dataset build, production split, graph construction, training, notebook execution, checkpoint load, graph `.pt` deserialization, commit, or push occurred.
