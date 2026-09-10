"""Static PNA regressor for the frozen CAMELS sparse graph experiment."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
from torch_geometric.nn import PNAConv
from torch_geometric.utils import add_remaining_self_loops

from src.models.static_gcn import sparse_graph_pool


CANONICAL_AGGREGATORS = ("mean", "min", "max", "std")
CANONICAL_SCALERS = ("identity", "amplification", "attenuation")


def validate_degree_histogram(deg: torch.Tensor | Sequence[int]) -> torch.Tensor:
    """Return a validated integer PNA degree histogram."""
    value = torch.as_tensor(deg)
    if value.ndim != 1 or value.numel() < 2:
        raise ValueError("PNA degree histogram must be a one-dimensional sequence with at least two bins.")
    if value.dtype.is_floating_point and not torch.equal(value, value.round()):
        raise ValueError("PNA degree histogram values must be integers.")
    value = value.to(dtype=torch.long)
    if (value < 0).any() or int(value.sum()) <= 0:
        raise ValueError("PNA degree histogram must contain non-negative counts with positive total.")
    if int((value * torch.arange(value.numel())).sum()) <= 0:
        raise ValueError("PNA degree histogram must describe at least one positive-degree node.")
    return value


def effective_pna_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Apply one deterministic remaining-self-loop policy without mutating input."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2,E], got {tuple(edge_index.shape)}")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    edge_index = edge_index.long()
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes):
        raise ValueError("edge_index contains an out-of-bounds node index.")
    effective, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    return effective


class PNAMessageLayer(nn.Module):
    """Canonical PNA operator followed by the frozen activation/dropout/norm/residual block."""

    def __init__(self, hidden_dim: int, dropout: float, deg: torch.Tensor,
                 use_layer_norm: bool, residual: bool) -> None:
        super().__init__()
        self.conv = PNAConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            aggregators=list(CANONICAL_AGGREGATORS),
            scalers=list(CANONICAL_SCALERS),
            deg=deg,
            edge_dim=None,
            towers=1,
            pre_layers=1,
            post_layers=1,
            divide_input=False,
            act="relu",
            train_norm=False,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.use_residual = bool(residual)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(x, edge_index)
        x = self.layer_norm(self.dropout(self.activation(x)))
        return x + residual if self.use_residual else x


class StaticPNARegressor(nn.Module):
    """Three-layer static PNA graph regressor; accepts sparse final-snapshot batches only."""

    def __init__(
        self,
        node_features: int = 7,
        hidden_dim: int = 32,
        num_layers: int = 3,
        dropout: float = 0.2,
        graph_pooling: str = "mean",
        deg: torch.Tensor | Sequence[int] | None = None,
        add_self_loops: bool = True,
        use_layer_norm: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if node_features <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("node_features, hidden_dim, and num_layers must be positive.")
        if graph_pooling != "mean":
            raise ValueError("The controlled Static PNA experiment requires graph_pooling='mean'.")
        if deg is None:
            raise ValueError("StaticPNARegressor requires a training-only PNA degree histogram.")
        if not add_self_loops:
            raise ValueError("The controlled Static PNA experiment requires remaining self-loops.")

        degree_histogram = validate_degree_histogram(deg)
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.graph_pooling = graph_pooling
        self.add_self_loops = True
        self.use_layer_norm = use_layer_norm
        self.residual = residual
        self.aggregators = CANONICAL_AGGREGATORS
        self.scalers = CANONICAL_SCALERS
        self.register_buffer("degree_histogram", degree_histogram.clone())

        self.input_projection = nn.Linear(node_features, hidden_dim)
        self.input_activation = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.layers = nn.ModuleList([
            PNAMessageLayer(hidden_dim, dropout, degree_histogram, use_layer_norm, residual)
            for _ in range(num_layers)
        ])
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim >= 2 else hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2 if hidden_dim >= 2 else hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        for module in self.regressor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_nodes(self, graph: Dict[str, Any]) -> torch.Tensor:
        if not isinstance(graph, dict) or graph.get("graph_storage") != "sparse_edge_index":
            raise ValueError("StaticPNARegressor requires a sparse static graph batch; no temporal path exists.")
        x = graph["x"].float()
        if x.ndim != 2 or x.shape[1] != self.node_features:
            raise ValueError(f"Expected sparse x shape [N,{self.node_features}], got {tuple(x.shape)}")
        edge_index = effective_pna_edge_index(graph["edge_index"], x.shape[0])
        x = self.input_norm(self.input_dropout(self.input_activation(self.input_projection(x))))
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

    def forward(self, A: Dict[str, Any], X: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        del X, mask
        nodes = self.encode_nodes(A)
        return self.regressor(sparse_graph_pool(
            nodes, A["batch"].long(), int(A["num_graphs"]), self.graph_pooling
        ))
