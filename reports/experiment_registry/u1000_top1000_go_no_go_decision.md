# U1000 Top1000 go/no-go decision

## Decision: NO-GO

Passed: 1,000 complete universes; exact five snapshots; 1,000 finite consistent targets; aggregate Top1000 padding 0.45118%; verified feature provenance and none normalization; correct minimum-image periodic geometry; 700-universe pools with fixed seed-specific 99/201 validation/test; deterministic nested design specified; Static final-snapshot reuse class A; 101 GiB free.

Failed blockers: dense O(N²) preprocessing, serialization and both models; eager projected ~20.2 GB dataset loading against a historically 31 GiB host; no external split manifest support in Evolve; non-atomic/non-resumable builder; selection and kNN tie determinism incomplete; historical graph identity not provable without a later permitted equivalence check.

The padding tail (minimum 157 valid halos at a=0.2) is a material caution but not the decisive blocker because only 1.22% of snapshots pad and masks are correct.

## Exactly one next action

**Refactor and validate the graph pipeline to a deterministic sparse `edge_index` schema—including builder, both trainers/models, Evolve split-config support, and atomic output—then rerun this pre-build audit before building.**

No dataset, production split, config, checkpoint, or experiment was created or run during this audit.
