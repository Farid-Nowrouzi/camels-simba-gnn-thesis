"""Permutation-invariant DeepSets regressor for raw CAMELS halo sets."""

from __future__ import annotations

import torch
from torch import nn


FEATURE_NAMES = ["log10_Mvir", "X", "Y", "Z", "VX", "VY", "VZ"]


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def validate_set_batch(x: torch.Tensor, mask: torch.Tensor, input_dim: int = 7) -> None:
    if x.ndim != 3 or x.shape[-1] != input_dim:
        raise ValueError(f"Expected x [B, N, {input_dim}], got {tuple(x.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"Expected mask [B, N], got {tuple(mask.shape)}")
    if tuple(mask.shape) != tuple(x.shape[:2]):
        raise ValueError(f"Mask {tuple(mask.shape)} is incompatible with x {tuple(x.shape)}")
    if mask.dtype is not torch.bool:
        raise TypeError(f"mask must have dtype torch.bool, got {mask.dtype}")
    if not torch.all(mask.any(dim=1)):
        raise ValueError("Every halo set must contain at least one valid node; all-masked input rejected.")


def masked_mean_pool(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if encoded.ndim != 3:
        raise ValueError(f"Expected encoded [B, N, H], got {tuple(encoded.shape)}")
    if mask.ndim != 2 or tuple(mask.shape) != tuple(encoded.shape[:2]):
        raise ValueError(f"Mask {tuple(mask.shape)} is incompatible with encoded {tuple(encoded.shape)}")
    if mask.dtype is not torch.bool:
        raise TypeError(f"mask must have dtype torch.bool, got {mask.dtype}")
    counts = mask.sum(dim=1, keepdim=True)
    if not torch.all(counts > 0):
        raise ValueError("Every halo set must contain at least one valid node; all-masked input rejected.")
    valid = mask.unsqueeze(-1).to(dtype=encoded.dtype)
    return (encoded * valid).sum(dim=1) / counts.to(dtype=encoded.dtype)


class IndependentHaloLayer(nn.Module):
    """A shared transformation applied independently to every halo."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.use_residual = input_dim == output_dim

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.layer_norm(self.dropout(self.activation(self.linear(values))))
        return values + residual if self.use_residual else values


class DeepSetsRegressor(nn.Module):
    """Raw-set comparator: independent halo encoding, masked mean, Static-GCN head."""

    def __init__(self, input_dim: int = 7, hidden_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.phi = nn.ModuleList([
            IndependentHaloLayer(input_dim, hidden_dim, dropout),
            IndependentHaloLayer(hidden_dim, hidden_dim, dropout),
            IndependentHaloLayer(hidden_dim, hidden_dim, dropout),
        ])
        half_hidden = hidden_dim // 2 if hidden_dim >= 2 else hidden_dim
        # Deliberately identical to StaticGCNRegressor.regressor for mean pooling.
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, half_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(half_hidden, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        validate_set_batch(x, mask, self.input_dim)
        values = x.float()
        for layer in self.phi:
            values = layer(values)
            values = values * mask.unsqueeze(-1).to(values.dtype)
        return values

    def pool(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_mean_pool(encoded, mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(x, mask)
        pooled = self.pool(encoded, mask)
        return self.regressor(pooled)
