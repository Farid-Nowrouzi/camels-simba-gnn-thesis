# Graph-pooling implementation

EvolveGCN-H receives node embeddings with shape `[B, T, N, H]` and a
`[B, T, N, 1]` real-node mask.

- `mean` computes the masked node sum divided by the masked node count and
  returns `[B, T, H]`.
- `mean_max` concatenates that mean with a masked maximum and returns
  `[B, T, 2H]`.

With `H=32`, the graph representation is 32-dimensional for mean and
64-dimensional for mean_max. Temporal pooling is mean in both methods and
preserves that final feature dimension. The fixed linear head therefore gains
exactly 32 input weights for mean_max. This is the necessary mathematical
consequence of concatenation, not a different head design.

Both reductions are permutation invariant and operate independently for every
graph and snapshot. Masks exclude padded nodes. Dataset validation disallows
zero-real-node snapshots; if that guarantee were bypassed, the masked maximum
would expose a latent all-masked edge case.

No canonical attention, sum, max-only, TopK, SAGPool, or Set2Set experiment was
trained. Attention pooling is unimplemented and untested.
