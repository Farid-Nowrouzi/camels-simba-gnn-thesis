# Regression-head implementation

Both heads receive the same 32-dimensional representation after masked mean
graph pooling and temporal mean pooling.

- `linear`: `Linear(32,1)`, 33 trainable parameters.
- `mlp`: `Linear(32,32) → ReLU → Dropout(0.2) → Linear(32,1)`, 1,089
  trainable parameters.

All Linear layers use bias and identity output activation. The Evolve head
modules use PyTorch default initialization. AdamW receives `model.parameters`,
so head parameters are optimized. Target Omega_m is unnormalized. Head type
changes checkpoint structure; checkpoints were checked only for existence.
