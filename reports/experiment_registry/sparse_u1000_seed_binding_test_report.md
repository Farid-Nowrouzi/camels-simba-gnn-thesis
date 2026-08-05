# Sparse U1000 split-manifest seed-binding test report

Decision: PASS. Date: 2026-08-05.

`src/training/split_manifest.py` now validates that the manifest contains an exact Python integer seed and, when a manifest is supplied by a trainer, requires equality with the invocation seed. The mismatch exception includes trainer seed, manifest seed, manifest path, and the instruction to use the matching seed or correct manifest.

Both `train_evolvegcn_h.create_loaders` and `train_static_gcn.create_loaders` pass their invocation seed to `load_split_manifest`. Both top-level training functions additionally call `validate_split_manifest_seed` before resolving dataset provenance, creating output directories, loading the dataset, constructing a DataLoader/model/optimizer, or writing any artifact. No silent seed substitution or override exists.

Six dedicated tests passed:

- seed 42 manifest with trainer seed 42 passes for both loaders and preserves exact stored order;
- Evolve seed 42 versus manifest seed 123 fails and reports both values;
- Static seed 42 versus manifest seed 123 fails and reports both values;
- missing seed and string `"42"` are rejected;
- no-manifest seeded internal splitting remains deterministic and available;
- both top-level trainer calls reject mismatch without creating the requested experiment root, checkpoint, prediction, metrics, or config file.

The historical sparse integration fixture was corrected from its former intentional mismatch to matching seed 42. Existing structural validation for dataset identity, ordered hashes, counts, duplicates, overlap, unknown IDs, and parent prefixes remains unchanged.

New experiment config metadata records `trainer_invocation_seed`, `split_manifest_seed`, `split_manifest_sha256`, `ordered_split_hashes`, caller dataset identity, published dataset/source/target identities from the sidecar when available, and the training-code Git commit. Existing graph/model protocol fields remain in their established config locations.
