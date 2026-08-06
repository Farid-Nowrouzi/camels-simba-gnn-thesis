# U1000 Top1500 integrity fixes audit

Date: 2026-08-06

Starting commit: `461430ba5bd1abbad3b218ce4d828c2e77f55397`

Decision: **READY FOR MANUAL TOP1500 BUILD**

The original pre-build integrity audit is preserved. This follow-up records the minimal backward-compatible fixes for its three blocking findings. No Top1500 dataset build, CUDA pilot, trainer, epoch loop, checkpoint creation, or prediction generation was started.

## Fix A: matrix artifact identity

Every Top1500 matrix preflight now uses bounded streaming SHA-256 reads to recompute the current dataset, metadata, completion-marker, and target-table identities. It canonically recomputes the embedded full raw-source-manifest identity. It validates metadata and completion-marker references to the current dataset, then compares the recomputed identities with all authoritative locations used by the matrix:

- 18 dataset-specific split bindings, including `dataset_identity`;
- 36 production model configurations;
- 36 scoped registry rows;
- metadata dataset/target/raw-manifest identities;
- completion-marker dataset identity and artifact names.

The same pass verifies split-manifest file hashes, canonical split-binding hashes, and ordered-partition identities. Missing values, malformed values, `PENDING_POST_BUILD`, valid-looking wrong hashes, and cross-location inconsistencies fail with the artifact and binding location in the error. This runs on every manager preflight, including the preflight used by `--resume`, before any trainer can be dispatched.

## Fix B: CUDA pilot binding

The successful pilot schema is `u1000_top1500_cuda_pilot_v2`. It records paths and SHA-256 identities for the dataset, metadata, and completion marker; raw- and target-source identities; Top-N, universe/snapshot/feature dimensions, normalization, periodicity, k, and box size; tested model names and production batch sizes; seed42 Train700 manifest path/hash and ordered-partition identity; source Git commit; UTC execution timestamp; CUDA device identity; forward/backward, finite-loss, and finite-gradient results; per-model peak allocated/reserved memory; and bounded safety counters.

Matrix preflight recomputes the current artifact identities and the authoritative seed42 Train700 manifest and partition identities, then compares every required pilot binding. It rejects a missing field, non-PASS result, malformed identity, stale artifact, wrong protocol, wrong seed/manifest/partition, different commit, missing device identity, or incomplete model evidence. `--resume` uses this same gate.

## Fix C: builder provenance

The production launcher explicitly passes the actual module and its own canonical repository-relative path. The builder derives its module and source path from the executing Python file, verifies the launcher declaration for Top1500, and records:

- `builder_provenance_schema_version`;
- `builder_entrypoint`;
- `builder_module`;
- `builder_source_path`;
- `builder_source_sha256`;
- `build_launcher_path`;
- `build_launcher_sha256`;
- `source_git_commit`.

Top1500 validation requires these fields, canonical repository-relative production paths, valid SHA-256 values, matching current source hashes, the expected builder module and launcher, and a full Git commit agreeing with the legacy `git_commit` field. Historical Top1000 metadata remains accepted when the new schema is absent; its full existing validator passed.

## Safe validation results

- Python compilation: PASS.
- Shell syntax for both production launchers: PASS.
- Required launcher/validator/manager help commands: PASS.
- Build launcher preflight: PASS; 5,000/5,000 raw catalogues, anchored raw-source identity `ba22c3611a70763566ffb38a20f9b5a36fb6c1a27c3ad8030c4a7e189ce87618`, and target SHA-256 `9692a97760ee0e3a97cf3293f1b73911ee0a1af028617f03fe85f88f431703c2`.
- Matrix manager and launcher preflight: expected exit 3 at the missing Top1500 dataset gate.
- Full lightweight unittest discovery: 46 passed, 0 failed.
- Historical completed Top1000 dataset validator: PASS.
- Matrix configuration validation: 36 unique planned cells; Evolve 18, Static 18; six levels and three seeds; 18/18 ordered partition bindings unchanged.
- Registry: planned 36, running 0, completed 0, failed 0, duplicate cells 0.
- Top1500 hashes: still `PENDING_POST_BUILD` because no dataset was built.
- Top1500 experiment directories/checkpoints/predictions: 0/0/0.

The build remains a deliberate manual action. The safe command is recorded only in the final handoff.
