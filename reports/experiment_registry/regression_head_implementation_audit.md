# Regression-head implementation audit

## EvolveGCN-H

`EvolveGCNHRegressor` accepts only `mlp` and `linear`; `mlp` is the default in
both the constructor and CLI. Historical missing values therefore resolve to
MLP. No aliases are accepted.

Graph pooling produces `[B,T,H]` for mean/sum and `[B,T,2H]` for mean_max.
Temporal mean or last produces `[B,D]`. Optional summary features are then
concatenated, after which the head maps to `[B,1]`.

- Linear: `nn.Linear(D,1)`, bias enabled, identity output.
- MLP: `nn.Linear(D,H)`, ReLU, `nn.Dropout(dropout)`,
  `nn.Linear(H,1)`, biases enabled, identity output.

For canonical graph mean, temporal mean, `D=H=32`: linear has 33 parameters;
MLP has 1,089. Both receive the exact same 32-dimensional representation.
The comparison changes only the final mapping. The optimizer is constructed
from `model.parameters()`, so both heads are optimized.

Evolve head Linear modules use PyTorch default initialization. Head type
changes checkpoint tensor shapes; checkpoints are not interchangeable across
head types. No output bounding is used, and target normalization is handled by
the training/evaluation path rather than changing head structure.

## Static GCN

Static GCN has no `head_type` option. Its fixed readout is
`Linear(D,H) → ReLU → Dropout → Linear(H,H/2) → ReLU → Dropout →
Linear(H/2,1)`, with explicit Xavier-uniform weights and zero biases.
At `D=H=32` this has 1,601 trainable parameters. It is a different
implementation and is not a canonical head comparator.

## Potential issues

No implementation bug invalidates the controlled U750 comparison. The MLP
dropout is active only during training, as intended. The exact architecture
offers no configurable depth, residual connection, normalization, or
uncertainty output. Historical configs without `head_type` depend on the MLP
default, which remains backward compatible but should be reported explicitly.
