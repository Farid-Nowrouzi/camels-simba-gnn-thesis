# Sparse U1000 tiny smoke-build report

Decision: PASS. This was infrastructure validation under `/tmp`, not a scientific dataset or training run.

- Fixture: two explicitly generated tiny CAMELS-like universes, the exact five snapshot names, Top16, none normalization, periodic kNN k=3, box 25, seven canonical features, dummy target 0.3.
- Build: 0.0715 seconds; output 30,778 bytes; process peak RSS 446,748 KiB (includes Python/PyTorch/pandas imports and is not incremental build memory).
- Graphs: ten snapshots; 16 real nodes each; 54 directed symmetric edges each. Batched temporal snapshots each had 32 nodes and 108 edges. Static final batch had 32 nodes and 108 edges.
- Output: schema `camels_temporal_sparse_v1`; SHA-256 `0007c06bf81e13cb146a930982bf73053a31f38a72afb548062e87af82b18d54`; matching metadata and completion marker.
- Models: Evolve and Static outputs both `[2,1]`; one loss/backward per model; losses 42.7523155 and 0.17245695 respectively; all gradients finite. No optimizer or epoch loop ran.
- Recovery: no `.tmp` or `.lock` leaked. Atomic collision refusal also passed in unit tests.
- Cleanup: `/tmp/sparse_u1000_smoke_jn028f9h` no longer existed after the run.

The checksum is specific to this one ephemeral smoke serialization and is recorded only as execution evidence.
