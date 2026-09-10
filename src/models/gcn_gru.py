"""GCN-GRU temporal Omega_m regressor."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.temporal_gcn_encoder import SharedGCNSnapshotEncoder


class GCNGRURegressor(nn.Module):
    """Shared snapshot GCN -> unidirectional GRU final state -> MLP."""

    def __init__(self, input_dim: int = 7, hidden_dim: int = 32,
                 num_gcn_layers: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.dropout_rate = dropout
        self.snapshot_encoder = SharedGCNSnapshotEncoder(
            input_dim, hidden_dim, num_gcn_layers, dropout,
            use_layer_norm=True, residual=True, add_self_loops=True,
        )
        self.temporal_module = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1,
            batch_first=True, bidirectional=False, dropout=0.0,
        )
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
        _, hidden = self.temporal_module(sequence)
        return hidden[-1]

    def forward(self, A_seq: torch.Tensor | Dict[str, Any],
                X_seq: Optional[torch.Tensor] = None,
                mask_seq: Optional[torch.Tensor] = None,
                summary_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if summary_features is not None:
            raise ValueError("GCN-GRU baseline does not use summary features")
        sequence = self.encode_snapshots(A_seq, X_seq, mask_seq)
        return self.regressor(self.temporal_representation(sequence))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
