# k × Hidden-Dimension Factorial Audit

## Executive decision

The clean target is a 48-row factorial:

`2 models × 2 hidden dimensions × 4 k values × 3 seeds`.

The repository currently provides 27 reusable rows and requires 21 new training rows:

- EvolveGCN-H: 15 reusable, 9 missing.
- Static GCN under the recommended three-layer protocol: 12 reusable, 12 missing.

All eight required k-specific graph datasets already exist. No graph construction is required.

The recommended analysis identifier is:

`controlled_knn_by_hidden_dim_factorial_500u_top100`

No training or dataset/checkpoint loading was performed during this audit.

## Fixed scientific protocol

Both model families use:

- 500 CAMELS-SIMBA universes;
- Top100 halos selected by raw Mvir descending;
- node features `log10(Mvir), X, Y, Z, VX, VY, VZ`;
- minmax feature normalization;
- periodic kNN construction in a 25 h^-1 Mpc box;
- k values 4, 6, 8, and 12;
- hidden dimensions 32 and 64;
- seeds 42, 123, and 2025;
- 350/75/75 ordered train/validation/test splits.

Only k, hidden dimension, and seed may vary within a model.

## EvolveGCN-H audit

The established clean Evolve protocol is:

- `num_layers=2`;
- batch size 4;
- 300 epochs;
- patience 40;
- learning rate 0.001;
- weight decay 0.00001;
- dropout 0.2;
- ReLU;
- mean graph pooling;
- mean temporal pooling;
- MLP head;
- self loops enabled;
- gradient clipping 1.0;
- no summary features;
- no target normalization.

Observed coverage:

| Width | k coverage | Reusable rows | Missing rows |
|---|---|---:|---:|
| h64/L2 | 4, 6, 8, 12 | 12 | 0 |
| h32/L2 | 8 | 3 | 9 at k=4,6,12 |

All 15 existing rows are 5/5 artifact-complete and use the expected dataset and split identity. The clean Evolve factorial therefore requires exactly nine new h32/L2 runs.

## Static GCN protocol comparison

### Protocol A — three layers at both widths

- Reuse the complete h64/L3 kNN family.
- Fixed batch size: 8.
- Fixed depth: 3.
- Existing reusable rows: 12 h64/L3 rows.
- Existing h32/L3 rows: 0.
- Missing rows: all 12 h32/L3 combinations.
- Incompatible rows: the three canonical h32/L2 k=8 rows, plus the less suitable historical h32/L2 batch-size-4 alternatives.
- New runs: **12**.

Advantages:

- lowest new-run count;
- preserves all 12 completed h64 kNN results;
- changes only hidden dimension within the repaired Static design;
- all new runs use h32 and are therefore the less expensive width relative to h64;
- retains the established h64 kNN architecture used in the completed connectivity analysis.

Disadvantages:

- the new Static h32 protocol differs from the two-layer h32 universe-scaling protocol;
- existing h32 k=8 results cannot be reused in the clean factorial.

Compute-cost classification: **moderate**, comprising 12 h32 Static runs.

### Protocol B — two layers at both widths

- Reuse the three canonical h32/L2 k=8 rows.
- Fixed batch size: 8.
- Fixed depth: 2.
- Reusable rows: 3.
- Missing h32/L2 rows: 9 at k=4,6,12.
- Missing h64/L2 rows: all 12 cells.
- Incompatible rows: all 12 existing h64/L3 kNN rows.
- New runs: **21**.

Advantages:

- matches the canonical h32 Static universe-scaling depth;
- matches EvolveGCN-H's numerical layer count, although the architectures remain different.

Disadvantages:

- requires nine more Static runs than Protocol A;
- replaces rather than reuses the complete h64/L3 kNN family;
- includes 12 comparatively expensive h64 runs;
- disconnects the factorial from the completed h64 connectivity analysis.

Compute-cost classification: **high**, comprising nine h32 and twelve h64 Static runs.

### Static recommendation

Choose **Protocol A: Static GCN L3 at both widths**. It is clean within Static and requires 12 new runs rather than 21.

Do not mix the existing h32/L2 rows with h64/L3 rows. At k=8, those three h32 rows are recorded as incompatible candidates and must be replaced by h32/L3 runs for this factorial.

## Graph reuse

Filesystem and metadata checks confirm all eight required graph datasets:

| Model input | k=4 | k=6 | k=8 | k=12 |
|---|---|---|---|---|
| Evolve temporal | present | present | present | present |
| Static final snapshot | present | present | present | present |

Every sidecar reports 500 successful universes, Top100, minmax, periodic kNN, the correct k, box size 25, the seven required node features, and raw-Mvir descending node selection.

Changing `hidden_dim` or `num_layers` changes the model, not the graph. Missing graph datasets: **0**.

## Split design

Every proposed row uses one canonical source per seed:

- seed 42: `experiments/evolvegcn_h_500u_seed42_final32/config.json`
- seed 123: `experiments/evolvegcn_h_500u_seed123_final32/config.json`
- seed 2025: `experiments/evolvegcn_h_500u_seed2025_final32/config.json`

Signatures:

- seed 42: `0f963679cd284fca861fc2c59d88bdae8e8f1f21e2cbe1bb73bd593b49056748`
- seed 123: `853549f16ef8eb3d7f18ae850c94b13c0c8bf0e770bb99cfbffff48b03530266`
- seed 2025: `3ce48b66c11e30bec459c52ba7f4a900809dd2b45be0995b8b56aeaefc747951`

Each source contains 350/75/75 disjoint ordered IDs whose union is exactly `LH_0` through `LH_499`. All 27 reusable rows already match these signatures across k, width, and model.

Static proposed runs should explicitly use the corresponding Evolve config as `split_config_path`. Evolve proposed runs use the established deterministic seed-and-ratio procedure and must be verified post-run against the same ordered signature.

## Row-count decision

| Component | Reusable | New | Final |
|---|---:|---:|---:|
| EvolveGCN-H L2 factorial | 15 | 9 | 24 |
| Static GCN L3 factorial | 12 | 12 | 24 |
| Total | 27 | **21** | 48 |

Eighteen new runs are not sufficient. That count would omit the three Static h32/L3 k=8 replacements and incorrectly reuse the existing h32/L2 rows, violating the fixed-depth requirement.

The alternative total of 30 corresponds to nine Evolve runs plus 21 Static Protocol-B runs. It is scientifically clean but unnecessarily expensive. The recommended total is **21**.

## Scientific readiness

The proposed completed design is fully controlled within each model:

- EvolveGCN-H holds L2, batch 4, temporal input, head, pooling, and optimization fixed.
- Static GCN holds L3, batch 8, final-snapshot input, head, pooling, and optimization fixed.
- k, hidden dimension, and seed are the only within-model varying fields.

The two models intentionally retain different architectures, temporal inputs, depths, heads, and batch sizes. Cross-model results compare complete fixed model protocols; they are not a pure architecture-only causal test.

## Highest-value next action

Create reuse-aware family specifications for the 9 missing Evolve h32/L2 rows and 12 missing Static h32/L3 rows, then dry-run and independently verify the exact 21-job selection. Do not build datasets and do not include the incompatible Static h32/L2 rows.
