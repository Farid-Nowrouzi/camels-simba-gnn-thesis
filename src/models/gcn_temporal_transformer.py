"""GCN temporal-Transformer Omega_m regressor."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from src.models.temporal_gcn_encoder import SharedGCNSnapshotEncoder


DEFAULT_SCALE_FACTORS = (0.2, 0.25, 0.51209, 0.75065, 1.0)


def sinusoidal_scale_factor_encoding(
    scale_factors: Sequence[float], d_model: int,
) -> torch.Tensor:
    """Fixed encoding from actual cosmic scale factors.

    For ``T`` coordinates, ``q_t=(a_t-a_0)(T-1)/(a_{T-1}-a_0)`` and
    ``PE[t,2i]=sin(q_t/10000^(2i/d))``,
    ``PE[t,2i+1]=cos(q_t/10000^(2i/d))``.
    """
    coordinates = torch.as_tensor(scale_factors, dtype=torch.float32)
    if coordinates.ndim != 1 or coordinates.numel() < 1:
        raise ValueError("scale_factors must be a non-empty one-dimensional sequence")
    if not torch.isfinite(coordinates).all() or (
        coordinates.numel() > 1 and not torch.all(coordinates[1:] > coordinates[:-1])
    ):
        raise ValueError("scale_factors must be finite and strictly increasing")
    if d_model <= 0:
        raise ValueError("d_model must be positive")
    if coordinates.numel() == 1:
        positions = torch.zeros_like(coordinates)
    else:
        positions = (coordinates - coordinates[0]) * (coordinates.numel() - 1)
        positions = positions / (coordinates[-1] - coordinates[0])
    even = torch.arange(0, d_model, 2, dtype=torch.float32)
    divisor = torch.exp(even * (-math.log(10000.0) / d_model))
    encoding = torch.zeros((coordinates.numel(), d_model), dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions[:, None] * divisor[None, :])
    if d_model > 1:
        encoding[:, 1::2] = torch.cos(
            positions[:, None] * divisor[None, :encoding[:, 1::2].shape[1]]
        )
    return encoding.unsqueeze(0)


class GCNTemporalTransformerRegressor(nn.Module):
    """Shared snapshot GCN -> one Transformer encoder -> final token -> MLP."""

    def __init__(self, input_dim: int = 7, hidden_dim: int = 32,
                 num_gcn_layers: int = 3, dropout: float = 0.2,
                 scale_factors: Sequence[float] = DEFAULT_SCALE_FACTORS,
                 nhead: int = 4, num_encoder_layers: int = 1,
                 dim_feedforward: int = 64) -> None:
        super().__init__()
        if hidden_dim % nhead:
            raise ValueError("hidden_dim must be divisible by nhead")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.dropout_rate = dropout
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.snapshot_encoder = SharedGCNSnapshotEncoder(
            input_dim, hidden_dim, num_gcn_layers, dropout,
            use_layer_norm=True, residual=True, add_self_loops=True,
        )
        coordinates = torch.as_tensor(scale_factors, dtype=torch.float32)
        self.register_buffer("scale_factors", coordinates, persistent=True)
        self.register_buffer(
            "temporal_encoding",
            sinusoidal_scale_factor_encoding(scale_factors, hidden_dim),
            persistent=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.temporal_module = nn.TransformerEncoder(layer, num_layers=num_encoder_layers)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._reset_head()

    def _reset_head(self) -> None:
        for module in self.regressor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_snapshots(self, A_seq, X_seq=None, mask_seq=None) -> torch.Tensor:
        return self.snapshot_encoder.encode_sequence(A_seq, X_seq, mask_seq)

    def temporal_representation(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != self.hidden_dim:
            raise ValueError(f"Expected sequence [B,T,{self.hidden_dim}], got {tuple(sequence.shape)}")
        if sequence.shape[1] != self.temporal_encoding.shape[1]:
            raise ValueError(
                f"Expected {self.temporal_encoding.shape[1]} snapshots, got {sequence.shape[1]}"
            )
        encoded = self.temporal_module(
            sequence + self.temporal_encoding.to(dtype=sequence.dtype)
        )
        return encoded[:, -1, :]

    def forward(self, A_seq: torch.Tensor | Dict[str, Any],
                X_seq: Optional[torch.Tensor] = None,
                mask_seq: Optional[torch.Tensor] = None,
                summary_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if summary_features is not None:
            raise ValueError("GCN-Transformer baseline does not use summary features")
        sequence = self.encode_snapshots(A_seq, X_seq, mask_seq)
        return self.regressor(self.temporal_representation(sequence))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
