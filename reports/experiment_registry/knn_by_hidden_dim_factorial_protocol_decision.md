# Protocol Decision: Controlled k × Hidden-Dimension Factorial

## Chosen design

Analysis name:

`controlled_knn_by_hidden_dim_factorial_500u_top100`

| Model | Fixed depth | Fixed batch size | Reusable rows | New rows | Final rows |
|---|---:|---:|---:|---:|---:|
| EvolveGCN-H | 2 | 4 | 15 | 9 | 24 |
| Static GCN | 3 | 8 | 12 | 12 | 24 |
| Total | — | — | 27 | 21 | 48 |

Only k, hidden dimension, and seed vary within either model.

## Why Static L3 is selected

Static Protocol A reuses the complete 12-row h64/L3 kNN family and requires 12 h32/L3 runs. Static Protocol B would reuse only three h32/L2 k=8 rows and require 21 new Static runs, including 12 h64 runs.

Protocol A therefore:

- saves nine training runs;
- avoids replacing a completed controlled family;
- puts all new Static work at h32 rather than h64;
- preserves direct continuity with the completed h64 connectivity analysis.

The drawback is that the Static factorial uses L3 rather than the L2 depth of the canonical h32 universe-scaling family. That is acceptable because depth is fixed inside this factorial and the scientific question concerns k and hidden dimension, not cross-study equality with the universe-scaling protocol.

## Why 18 runs are insufficient

The obvious missing cells are:

- nine Evolve h32 cells at k=4,6,12;
- nine Static h32 cells at k=4,6,12.

That gives 18, but leaves Static k=8 as h32/L2 versus h64/L3. The three Static h32/L3 k=8 replacements are also required. Therefore:

`9 + 9 + 3 = 21`.

## Why 30 runs are unnecessary

Choosing Static L2 would require:

- nine missing h32/L2 cells;
- twelve missing h64/L2 cells;
- nine missing Evolve cells.

Total: `9 + 21 = 30`.

This is clean but has no scientific-control advantage over fixing Static at L3, because either depth is acceptable when held constant within Static.

## Graph and split policy

All temporal and static graph datasets for k=4,6,8,12 already exist. No graph rebuild is needed.

Every new run must use the exact ordered seed split represented by:

- `experiments/evolvegcn_h_500u_seed42_final32/config.json`
- `experiments/evolvegcn_h_500u_seed123_final32/config.json`
- `experiments/evolvegcn_h_500u_seed2025_final32/config.json`

Static should consume these configs explicitly. Evolve must reproduce their deterministic ordered IDs and be checked after completion.

## Future specification requirements

The future family specifications must:

- contain the complete 24-row product for each model;
- mark all 27 existing compatible rows as `reuse`;
- create exactly 21 `run_if_missing` rows;
- reject partial directories;
- exclude Static h32/L2 rows from the L3 factorial;
- keep all graph paths repository-relative;
- launch sequentially only;
- verify 48/48 complete rows before analysis.

Do not create those training specifications as part of this design audit.
