# U1000 Top1500 matrix preflight

**Decision: TOP1500 BUILD REQUIRED BEFORE CUDA.** This is the expected preparation-stage gate.

- Intended matrix cells: 36/36.
- EvolveGCN-H: 18; Static GCN: 18.
- Lifecycle state: planned 36, running 0, completed 0, failed 0.
- Duplicate canonical IDs/cells: 0.
- Config protocol validation: PASS.
- Immutable partition reuse: PASS (18/18; ordered train/validation/test/unused IDs identical to Top1000).
- Dataset checksum/source binding: explicitly pending until post-build validation.
- Dataset completion marker: missing, as expected before manual build.
- CUDA pilot: not run; correctly gated.
- Matrix preflight exit: 3 at the missing-dataset gate.
- Analysis gate: refuses output until 36/36 runs are completed and validated.

After the build, run the dataset validator, then `manage_u1000_top1500_training_scaling_matrix.py --bind-dataset`, then the exact production-batch CUDA pilot. Only a PASS pilot permits matrix readiness.
