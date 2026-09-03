"""Shared Static-GCN snapshot encoder for temporal graph regressors."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.static_gcn import (
    DenseGCNLayer,
    masked_mean_pool,
    normalize_adjacency,
    normalize_sparse_edges,
    sparse_graph_pool,
    sparse_message,
    validate_static_batch,
)


class SharedGCNSnapshotEncoder(nn.Module):
    """Encode any snapshot with one shared three-layer GCN and mean pooling.

    The module has no temporal state and never assumes that node indices identify
    the same halo in different snapshots.  A sparse snapshot is a disconnected
    graph batch with ``x``, ``edge_index``, ``batch`` and ``num_graphs`` fields.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 32,
        num_layers: int = 3,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
        residual: bool = True,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("input_dim, hidden_dim, and num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.use_layer_norm = use_layer_norm
        self.residual = residual
        self.add_self_loops = add_self_loops
        self.graph_pooling = "mean"

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_activation = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.layers = nn.ModuleList([
            DenseGCNLayer(
                hidden_dim, hidden_dim, dropout=dropout, activation=nn.ReLU(),
                use_layer_norm=use_layer_norm, residual=residual,
            )
            for _ in range(num_layers)
        ])
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)

    def forward(
        self,
        graph: torch.Tensor | Dict[str, Any],
        x: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return one graph embedding per batch member, shape ``[B, H]``."""
        if isinstance(graph, dict):
            values = graph["x"].float()
            if values.ndim != 2 or values.shape[1] != self.input_dim:
                raise ValueError(f"Expected sparse x [N,{self.input_dim}], got {tuple(values.shape)}")
            hidden = self.input_norm(
                self.input_dropout(self.input_activation(self.input_projection(values)))
            )
            edge_index, edge_weight = normalize_sparse_edges(
                graph["edge_index"], hidden.shape[0], graph.get("edge_weight"),
                self.add_self_loops,
            )
            for layer in self.layers:
                previous = hidden
                hidden = layer.linear(sparse_message(edge_index, edge_weight, hidden))
                hidden = layer.layer_norm(layer.dropout(layer.activation(hidden)))
                if layer.use_residual:
                    hidden = hidden + previous
            return sparse_graph_pool(
                hidden, graph["batch"].long(), int(graph["num_graphs"]), "mean"
            )

        if x is None:
            raise ValueError("Dense snapshot input requires x")
        validate_static_batch(graph, x, mask)
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected dense x feature dim {self.input_dim}, got {x.shape[-1]}")
        hidden = self.input_norm(
            self.input_dropout(self.input_activation(self.input_projection(x.float())))
        )
        if mask is not None:
            hidden = hidden * mask.float()
        adjacency = normalize_adjacency(graph, add_self_loops=self.add_self_loops)
        for layer in self.layers:
            hidden = layer(adjacency, hidden, mask)
        return masked_mean_pool(hidden, mask)

    def encode_sequence(
        self,
        temporal_batch: torch.Tensor | Dict[str, Any],
        x_seq: Optional[torch.Tensor] = None,
        mask_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply this exact encoder instance chronologically; return ``[B,T,H]``."""
        if isinstance(temporal_batch, dict):
            snapshots = temporal_batch.get("snapshots")
            if not isinstance(snapshots, list) or not snapshots:
                raise ValueError("Sparse temporal batch must contain a non-empty snapshots list")
            return torch.stack([self(snapshot) for snapshot in snapshots], dim=1)
        if x_seq is None or x_seq.ndim != 4:
            raise ValueError("Dense temporal input requires x_seq [B,T,N,F]")
        if temporal_batch.ndim != 4 or temporal_batch.shape[:2] != x_seq.shape[:2]:
            raise ValueError("Dense adjacency and feature sequence shapes do not align")
        embeddings = []
        for timestep in range(x_seq.shape[1]):
            mask = None if mask_seq is None else mask_seq[:, timestep]
            embeddings.append(self(temporal_batch[:, timestep], x_seq[:, timestep], mask))
        return torch.stack(embeddings, dim=1)
