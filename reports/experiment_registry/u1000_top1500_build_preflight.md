# U1000 Top1500 build preflight

**Decision: READY FOR TOP1500 BUILD.**

- Required branch: PASS (`thesis-sparse-integrity-hardening`).
- Target table: PASS, SHA-256 `9692a97760ee0e3a97cf3293f1b73911ee0a1af028617f03fe85f88f431703c2`.
- Required raw catalogues: PASS (5,000/5,000 readable; exact expected filenames).
- Raw halo-count audit: GO.
- Conflicting builder/trainer: none observed during preparation.
- Existing/partial Top1500 output: absent.
- Available disk during preflight: approximately 99 GiB.
- Minimum launcher disk gate: 4 GiB.
- Estimated dataset size: approximately 1.5 GiB (linear sparse-storage projection from the 1.047 GB Top1000 dataset; actual serialization may vary).
- Estimated build cost: roughly 1.5–2.25 times Top1000. The 50-minute Top1000 build suggests approximately 75–115 minutes; periodic kNN distance work is the upper, quadratic component.
- `--preflight-only`: PASS; exact builder command printed; Python construction not invoked.

The launcher refuses valid completed outputs, preserves partial/invalid artifacts, uses the builder's atomic paths, logs through `tee` with preserved exit status, and runs the production validator after a successful build.
