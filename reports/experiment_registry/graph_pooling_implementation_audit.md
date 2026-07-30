# Graph-pooling implementation audit

## Scope

This is a source-only audit of `src/models/`, `src/training/`, preprocessing
utilities, experiment configs, and Git history. No checkpoint or graph dataset
was loaded. Graph pooling is kept distinct from temporal pooling,
message-passing aggregation, hierarchical pooling, regression heads, and
handcrafted summary-feature concatenation.

## Implemented methods

| Model | Accepted `graph_pooling` values | Default | Implementation |
|---|---|---|---|
| EvolveGCN-H | `mean`, `sum`, `mean_max` | `mean` | `EvolveGCNHRegressor.masked_graph_pool` |
| Static GCN/GraphSAGE | `mean`, `max`, `mean_max` | `mean` | `StaticGCNRegressor.pool_graph`, `masked_mean_pool`, `masked_max_pool` |

The command-line parsers and model constructors accept only these exact
canonical spellings. `meanmax`, `mean+max`, `add`, `global_mean_pool`, and
similar strings are not aliases. Some directory names use `meanmax`, but their
parsed configs correctly record `mean_max`. Historical configs without
`graph_pooling` resolve to the constructor and trainer default, `mean`.

No attention, global-attention, `AttentionalAggregation`, TopK, SAGPool,
Set2Set, or other trainable/hierarchical pooling module is implemented in the
model or trainer source. Consequently there are no attention-pooling
parameters and no trained attention-pooling experiment.

## EvolveGCN-H

After message passing, the input to graph pooling has shape `[B, T, N, H]`,
with an optional mask `[B, T, N, 1]`.

- `mean`: multiply by the mask, sum over `N`, and divide by the real-node
  count clamped to at least 1. Output: `[B, T, H]`.
- `sum`: multiply by the mask and sum over `N`. Output: `[B, T, H]`.
- `mean_max`: compute the masked mean and a masked maximum, then concatenate
  them along the feature dimension. Masked entries are replaced by the
  smallest representable value before the maximum. Output: `[B, T, 2H]`.

Without a mask, the same reductions operate directly over `N`.
Graph pooling is performed independently for each graph and each snapshot.
Temporal pooling then reduces `T` by either a mean or final-snapshot selection
without changing the last dimension.

The constructor sets:

```text
graph_embedding_dim = H       for mean or sum
graph_embedding_dim = 2H      for mean_max
regressor_input_dim = graph_embedding_dim + summary_feature_dim
```

Thus a linear head changes from `Linear(H, 1)` to `Linear(2H, 1)`. At `H=32`,
the verified U750 linear mean and mean_max runs differ by exactly 32 trainable
parameters, the mathematically required extra input weights. The head pattern
is otherwise identical.

For an MLP head, the first layer changes from `Linear(H, H)` to
`Linear(2H, H)`. The U500 seed-42 mean_max pilot therefore has 1,024 more
parameters than its mean anchor, again exactly the required `32 × 32`
difference.

## Static GCN and Static GraphSAGE

The input to graph pooling has shape `[B, N, H]`, with optional mask
`[B, N, 1]`.

- `mean`: masked sum over `N` divided by the real-node count. Output:
  `[B, H]`.
- `max`: replace masked nodes by the smallest representable value and take
  the maximum over `N`. Output: `[B, H]`.
- `mean_max`: concatenate the mean and maximum. Output: `[B, 2H]`.

The regression head is the same MLP pattern for every method, with its first
input dimension set to `H` or `2H` as required. Static pooling occurs once on
the selected final-snapshot graph. There is no temporal pooling.

## Invariance, batching, and masks

All implemented reductions are permutation invariant with respect to node
ordering. Mean, sum, and maximum are applied independently along the node
dimension for each batch element. Evolve additionally preserves the snapshot
dimension until temporal pooling.

Processed datasets have fixed padded shapes (`N=500` or `N=1000` in the
candidate protocols), while masks distinguish real from padded nodes.
Preprocessing and dataset validation reject snapshots with zero real nodes.
This makes the canonical candidate datasets safe for masked mean and maximum.

If an all-masked graph were passed directly despite those validation
guarantees, mean pooling would return a zero vector and max pooling would
return the smallest finite value. This is a latent defensive-programming
limitation, not an observed experiment bug. Static `masked_max_pool` also
relies on the model's earlier batch validator for mask shape rather than
checking mask dimensionality itself.

## Optimizer coverage

Both trainers construct AdamW with `model.parameters()`. All registered
pooling-dependent head weights are therefore optimized. Mean, sum, and maximum
themselves are parameter-free. No attention module exists whose parameters
could be omitted.

## Scientific implementation risks, not code defects

- Maximum pooling can be dominated by a single noisy or outlying node
  embedding.
- Sum pooling depends on the number and scale of real nodes; fixed padded
  tensor size does not guarantee identical real-node count.
- `mean_max` doubles readout width and changes the number of downstream input
  weights. This is a necessary consequence of concatenation, not an
  uncontrolled head redesign.
- Mean and maximum preserve no information about intermediate distributional
  structure beyond their respective statistics.

No implementation error was found that invalidates the controlled U750
mean-versus-mean_max comparison.

## History

Git history shows graph-pooling support in the initial Evolve pipeline and
subsequent reproducible Evolve workflow, and in the original Static GCN model
and training pipeline. Current accepted values remain backward compatible with
historical configs through the default `mean` behavior.
