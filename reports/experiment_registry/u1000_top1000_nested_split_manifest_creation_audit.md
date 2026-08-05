# U1000 Top1000 nested split-manifest creation audit

## Decision

**GO FOR CUDA PILOTS.** Exactly 18 immutable manifests were created for seeds 42, 123, and 2025 at Train20/50/100/200/450/700. This authorizes only separately reviewed pilots; no training occurred in this task.

## Immutable inputs and design

All four requested file SHA-256 identities matched before generation, and the read-only production dataset validator returned PASS. Manifests bind the sparse schema, full-SHA256 source policy, source-manifest identity, seven ordered features, no normalization, periodic kNN k=8, box size 25, five ordered snapshots, and Omega_m target. The stale serialized provenance label is recorded as non-operative and was not used.

For each seed, the exact historical U750 ordered Train450, validation 99, and test 201 lists match the canonical linear-head EvolveGCN-H configuration. Prefixes supply Train20/50/100/200. Train700 appends the role-plan-approved ordered LH_750..LH_999 extension. Unused IDs are the numeric master-dataset order complement.

## Validation outcome

All counts, ordered hashes, prefix transitions, parent canonical identities, disjointness, uniqueness, known-ID checks, full population accounting, target summaries, trainer ordering, seed binding, Static final-snapshot behavior, and deterministic regeneration passed. Train20 target coverage is reported in the target-distribution CSV and was not optimized against test results.

The canonical manifest identity excludes only `creation_metadata` and the identity field itself. Parent identities therefore remain deterministic even when creation timestamps differ.

No graph was rebuilt, no dataset/metadata/completion marker/target was modified, and no training, optimizer step, checkpoint, prediction, experiment directory, or 36-cell configuration matrix was created.
