# Sparse U1000 final build reauthorization

## Decision: GO

The two final production-integrity blockers are remediated:

1. New sparse builds require a deterministic, verified `camels_source_manifest_v1` containing full SHA-256 for every selected halo catalogue and the authoritative target CSV. The manifest is verified before preprocessing and again before atomic completion. Stat-only provenance cannot be selected for a new sparse build; historical files remain readable and explicitly legacy/unverified.
2. EvolveGCN-H and Static GCN require exact equality between trainer invocation seed and external split-manifest seed before dataset loading or experiment-output creation. Matching seeds and historical no-manifest splitting remain compatible.

All 18 baseline and 17 new tests pass: 35 passed, 0 failed, 0 skipped. Compilation and all three relevant CLI help commands pass. The bounded real-file hashing benchmark is deterministic and projects approximately 45 seconds for build-plus-verification over the 22.15 GB raw catalogue corpus. No U1000 production dataset, production split, experiment config, training run, checkpoint, or scientific prediction was created.

GO authorizes the separately controlled production build only after the authoritative `outputs/target_inspection_1000u.csv` is prepared and reviewed. The builder will hash that file as a first-class target source and will fail if it is absent or changes. The highest-value next action is to prepare/review that target CSV, rerun the inert production command preflight, and execute the reviewed build in monitored `tmux`; production split manifests must still wait for the published dataset checksum.

Source-provenance commit: `71d94e6` (`Add full source and target SHA-256 provenance`). The seed-binding/report commit is this report's containing commit and is reported exactly in the final task response because a commit cannot embed its own final hash.
