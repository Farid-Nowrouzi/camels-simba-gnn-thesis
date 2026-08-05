# U1000 Top1000 CUDA Pilot Production Readiness

## Final decision

**GO FOR TRAIN700 PILOTS** for the validated U1000 Top1000 sparse dataset and immutable seed42 Train700 manifest.

All required batch-size-1 gates passed on an NVIDIA L40: true CUDA placement, real sparse inputs, finite scalar outputs and MSE losses, complete finite gradients, correct temporal/final-snapshot semantics, and no dense adjacency. The intended EvolveGCN-H batch size 4 and Static GCN batch size 8 also passed bounded forward-only tests with substantial memory headroom.

## Approved production inputs

- Dataset identity: `6a794c38216175212ec969283fdbd214bcdc64caecde78d24b81545e681a332a`
- Split manifest: `configs/splits/u1000_top1000_none_k8_sparse/seed42_train700.json`
- Manifest seed: 42
- Training universes: 700
- EvolveGCN-H intended batch size: 4 (forward-only pilot PASS)
- Static GCN intended batch size: 8 (forward-only pilot PASS)

## Operational conditions

This GO applies when the same production environment exposes the NVIDIA GPU, the immutable dataset/metadata/completion and manifest identities remain unchanged, and no competing large GPU workload is active. Training must continue to use the external manifest rather than automatic splitting and must preserve sparse `edge_index` batching.

The pilot does not authorize or substitute for scientific training, checkpoint review, final metrics, or experiment-family configuration. It only establishes that the two canonical model/data paths can execute safely at the intended batch sizes.

## Highest-value next action

Prepare and independently review the exact Train700 experiment configurations, then launch the first monitored production training only after confirming the same identities, GPU availability, sparse path, seed binding, and intended batch size at runtime.
