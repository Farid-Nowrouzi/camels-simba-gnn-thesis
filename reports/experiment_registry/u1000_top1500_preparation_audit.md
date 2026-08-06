# U1000 Top1500 preparation audit

## Decision

**READY FOR MANUAL TOP1500 BUILD.** The raw-data decision is GO, but CUDA and matrix execution remain correctly gated on a completed, validated dataset and post-build binding.

## Controlled protocol

- Anchor dataset validation: PASS; SHA-256 `6a794c38216175212ec969283fdbd214bcdc64caecde78d24b81545e681a332a`.
- Anchor Train700 MAE: Static GCN `0.039673901872 ± 0.003052405006`; EvolveGCN-H `0.057975908294 ± 0.003345054932` (sample standard deviations).
- Controlled change: Top-N 1000 to 1500 only.
- U1000, LH_0..LH_999, five snapshots, seven features, no node/target normalisation, periodic kNN k=8, box 25, sparse edge indices, and Omega_m target are preserved.
- Production builder ordering remains raw `Mvir` descending, halo ID ascending tie-break, stable original-row fallback.

## Evidence and preparation

- Raw catalogues: 5,000/5,000 inspected successfully; 111 require padding; total padding 66,313 slots (0.884173%).
- Existing Top1500 reuse: BUILD REQUIRED.
- Split partitions: exact ordered IDs preserved for all 18 seed/training-count manifests; no random split generation.
- Dataset bindings: 18 explicit `PENDING_POST_BUILD` bindings. `--bind-dataset` replaces them only after validator PASS.
- Configurations: 36 (18 EvolveGCN-H, 18 Static GCN), copied from the completed Top1000 hyperparameter templates.
- Registry: 36 planned, zero running/completed/failed, zero duplicate cells.
- Current source commit during preparation: `651003751bbd975ed5b2aa9314f7513d76550d19`.
- Existing Notebook 11, Top1000 analysis, experiment outputs, and reports were preserved as unrelated worktree changes.

No dataset build, CUDA pilot, epoch loop, checkpoint, prediction export, model training, or tmux session was started.
