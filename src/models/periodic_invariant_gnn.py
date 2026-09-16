"""Controlled periodic physics-informed invariant GNN regressor."""
from __future__ import annotations

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _InvariantLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(hidden_dim * 2 + 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        messages = self.edge_mlp(torch.cat((h[target], h[source], edge_attr), dim=-1))
        summed = torch.zeros_like(h)
        summed.index_add_(0, target, messages)
        counts = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
        counts.index_add_(0, target, torch.ones((target.numel(), 1), dtype=h.dtype, device=h.device))
        return h + self.update(torch.cat((h, summed / counts.clamp_min(1)), dim=-1))


class PeriodicInvariantGNNRegressor(nn.Module):
    """Three-layer 17-wide mean-message GNN using only approved invariants."""
    def __init__(self, hidden_dim: int = 17, num_layers: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        if hidden_dim != 17 or num_layers != 3:
            raise ValueError("controlled architecture is fixed at hidden_dim=17, num_layers=3")
        self.input_embedding = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
        self.layers = nn.ModuleList([_InvariantLayer(hidden_dim, dropout) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 8), nn.ReLU(), nn.Dropout(dropout), nn.Linear(8, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight); nn.init.zeros_(module.bias)

    def forward(self, node_scalar: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                batch: torch.Tensor, num_graphs: int | None = None) -> torch.Tensor:
        if node_scalar.ndim != 2 or node_scalar.shape[1] != 1: raise ValueError("node_scalar must be [N,1]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2: raise ValueError("edge_index must be [2,E]")
        if edge_attr.shape != (edge_index.shape[1], 3): raise ValueError("edge_attr must be [E,3]")
        if batch.ndim != 1 or batch.shape[0] != node_scalar.shape[0]: raise ValueError("batch must be [N]")
        h = self.input_embedding(node_scalar)
        for layer in self.layers: h = layer(h, edge_index.long(), edge_attr)
        n_graphs = int(batch.max().item()) + 1 if num_graphs is None and batch.numel() else (num_graphs or 0)
        pooled = torch.zeros((n_graphs, h.shape[-1]), device=h.device, dtype=h.dtype)
        pooled.index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=n_graphs).to(h.dtype).clamp_min(1).unsqueeze(-1)
        return self.head(pooled / counts)
