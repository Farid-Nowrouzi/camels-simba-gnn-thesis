"""Canonical EvolveGCN-O temporal graph regressor for CAMELS halo graphs.

EvolveGCN-O treats every GCN weight matrix as the input and output of a
matrix-valued recurrent unit.  Unlike EvolveGCN-H, node embeddings never enter
the recurrent update.  Consequently, node identity and node count may change
between snapshots: only the ordered graph sequence is required.

The matrix-GRU equations follow the authors' reference EvolveGCN-O code.  The
surrounding graph convolution, pooling, and regression readout deliberately
match this repository's EvolveGCN-H control.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn as nn

from src.models.evolvegcn_h import normalize_dense_adjacency
from src.models.static_gcn import normalize_sparse_edges, sparse_graph_pool, sparse_message


class MatrixGRUGate(nn.Module):
    """One gate of the matrix-GRU used by canonical EvolveGCN-O."""

    def __init__(self, rows: int, columns: int, activation: nn.Module) -> None:
        super().__init__()
        self.left_input = nn.Parameter(torch.empty(rows, rows))
        self.left_state = nn.Parameter(torch.empty(rows, rows))
        self.bias = nn.Parameter(torch.zeros(rows, columns))
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.left_input)
        nn.init.xavier_uniform_(self.left_state)
        nn.init.zeros_(self.bias)

    def forward(self, recurrent_input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.activation(
            self.left_input @ recurrent_input + self.left_state @ state + self.bias
        )


class MatrixGRUCell(nn.Module):
    """Evolve a rectangular GCN weight matrix without node-embedding input."""

    def __init__(self, rows: int, columns: int) -> None:
        super().__init__()
        self.update = MatrixGRUGate(rows, columns, nn.Sigmoid())
        self.reset = MatrixGRUGate(rows, columns, nn.Sigmoid())
        self.candidate = MatrixGRUGate(rows, columns, nn.Tanh())

    def forward(self, previous_weight: torch.Tensor) -> torch.Tensor:
        # In EvolveGCN-O the prior GCN weight is both recurrent input and state.
        update = self.update(previous_weight, previous_weight)
        reset = self.reset(previous_weight, previous_weight)
        candidate = self.candidate(previous_weight, reset * previous_weight)
        return (1.0 - update) * previous_weight + update * candidate


class EvolveGCNOLayer(nn.Module):
    """One O-formulation evolving graph-convolution layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: bool = True,
        activation_name: Literal["relu", "leaky_relu", "elu"] = "relu",
        dropout: float = 0.0,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        activations = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}
        if activation_name not in activations:
            raise ValueError("activation_name must be relu, leaky_relu, or elu.")
        self.in_features = in_features
        self.out_features = out_features
        self.use_activation = activation
        self.add_self_loops = add_self_loops
        self.initial_weight = nn.Parameter(torch.empty(in_features, out_features))
        self.weight_evolver = MatrixGRUCell(in_features, out_features)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = activations[activation_name]()
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.initial_weight)

    def evolved_weights(self, timesteps: int) -> List[torch.Tensor]:
        """Return the chronological weight states used for a fresh sequence."""
        if timesteps < 1:
            raise ValueError("timesteps must be positive.")
        weight = self.initial_weight
        states: List[torch.Tensor] = []
        for _ in range(timesteps):
            weight = self.weight_evolver(weight)
            states.append(weight)
        return states

    def _finish(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.bias
        if self.use_activation:
            value = self.activation(value)
        return self.dropout(value)

    def forward_sparse(self, snapshots: List[Dict[str, Any]]) -> List[torch.Tensor]:
        if not snapshots:
            raise ValueError("Sparse Evolve input requires at least one snapshot.")
        weights = self.evolved_weights(len(snapshots))
        outputs: List[torch.Tensor] = []
        for graph, weight in zip(snapshots, weights):
            x = graph["x"].float()
            if x.ndim != 2 or x.shape[1] != self.in_features:
                raise ValueError(
                    f"Expected sparse x [total_nodes,{self.in_features}], got {tuple(x.shape)}"
                )
            support = x @ weight
            edge_index, edge_weight = normalize_sparse_edges(
                graph["edge_index"], x.shape[0], graph.get("edge_weight"), self.add_self_loops
            )
            outputs.append(self._finish(sparse_message(edge_index, edge_weight, support)))
        return outputs

    def forward_dense(
        self,
        adjacency: torch.Tensor,
        features: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if adjacency.ndim != 4 or features.ndim != 4:
            raise ValueError("Dense temporal input must have adjacency/features rank 4.")
        if adjacency.shape[:3] != features.shape[:3] or adjacency.shape[2] != adjacency.shape[3]:
            raise ValueError("Dense temporal adjacency and feature shapes do not match.")
        if features.shape[-1] != self.in_features:
            raise ValueError(f"Expected {self.in_features} input features.")
        if mask is not None and tuple(mask.shape) != tuple(features.shape[:-1]) + (1,):
            raise ValueError("Dense temporal mask shape does not match features.")
        weights = self.evolved_weights(features.shape[1])
        outputs = []
        for timestep, weight in enumerate(weights):
            norm = normalize_dense_adjacency(adjacency[:, timestep], self.add_self_loops)
            value = torch.matmul(norm, torch.matmul(features[:, timestep], weight))
            value = self._finish(value)
            if mask is not None:
                value = value * mask[:, timestep]
            outputs.append(value)
        return torch.stack(outputs, dim=1)


class EvolveGCNORegressor(nn.Module):
    """Graph-level Omega_m regressor using canonical EvolveGCN-O weight updates."""

    def __init__(
        self,
        node_features: int = 7,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        activation: Literal["relu", "leaky_relu", "elu"] = "relu",
        temporal_pooling: Literal["mean", "last"] = "mean",
        graph_pooling: Literal["mean", "sum", "mean_max"] = "mean",
        add_self_loops: bool = True,
        summary_feature_dim: int = 0,
        head_type: Literal["linear", "mlp"] = "linear",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if temporal_pooling not in {"mean", "last"}:
            raise ValueError("temporal_pooling must be mean or last.")
        if graph_pooling not in {"mean", "sum", "mean_max"}:
            raise ValueError("graph_pooling must be mean, sum, or mean_max.")
        if head_type not in {"linear", "mlp"}:
            raise ValueError("head_type must be linear or mlp.")
        if summary_feature_dim < 0:
            raise ValueError("summary_feature_dim must be non-negative.")
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.temporal_pooling = temporal_pooling
        self.graph_pooling = graph_pooling
        self.summary_feature_dim = summary_feature_dim
        dimensions = [node_features] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([
            EvolveGCNOLayer(
                dimensions[index], dimensions[index + 1], True, activation,
                dropout, add_self_loops,
            )
            for index in range(num_layers)
        ])
        graph_dim = hidden_dim * (2 if graph_pooling == "mean_max" else 1)
        input_dim = graph_dim + summary_feature_dim
        self.regressor = (
            nn.Linear(input_dim, 1)
            if head_type == "linear"
            else nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                               nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        )

    def _temporal_pool(self, graph_sequence: torch.Tensor) -> torch.Tensor:
        return graph_sequence[:, -1] if self.temporal_pooling == "last" else graph_sequence.mean(1)

    def _regress(
        self, embedding: torch.Tensor, summary_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.summary_feature_dim:
            if summary_features is None or summary_features.shape != (
                embedding.shape[0], self.summary_feature_dim
            ):
                raise ValueError("summary_features has the wrong shape.")
            embedding = torch.cat([embedding, summary_features], dim=-1)
        return self.regressor(embedding)

    def forward(
        self,
        A_seq: torch.Tensor | Dict[str, Any],
        X_seq: Optional[torch.Tensor] = None,
        mask_seq: Optional[torch.Tensor] = None,
        summary_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(A_seq, dict):
            snapshots = A_seq.get("snapshots")
            if not isinstance(snapshots, list) or not snapshots:
                raise ValueError("Sparse temporal batch must contain snapshots.")
            current = snapshots
            for layer in self.layers:
                values = layer.forward_sparse(current)
                current = [dict(graph, x=value) for graph, value in zip(current, values)]
            pooled = [
                sparse_graph_pool(
                    graph["x"], graph["batch"].long(), int(graph["num_graphs"]),
                    self.graph_pooling,
                )
                for graph in current
            ]
            return self._regress(self._temporal_pool(torch.stack(pooled, dim=1)), summary_features)

        if X_seq is None:
            raise ValueError("Dense Evolve input requires X_seq.")
        values = X_seq
        for layer in self.layers:
            values = layer.forward_dense(A_seq, values, mask_seq)
        pooled = EvolveGCNORegressor._dense_graph_pool(values, mask_seq, self.graph_pooling)
        return self._regress(self._temporal_pool(pooled), summary_features)

    @staticmethod
    def _dense_graph_pool(
        values: torch.Tensor,
        mask: Optional[torch.Tensor],
        mode: Literal["mean", "sum", "mean_max"],
    ) -> torch.Tensor:
        if mask is None:
            mean = values.mean(2)
            if mode == "sum":
                return values.sum(2)
            return torch.cat([mean, values.max(2).values], -1) if mode == "mean_max" else mean
        summed = (values * mask).sum(2)
        if mode == "sum":
            return summed
        mean = summed / mask.sum(2).clamp(min=1.0)
        if mode == "mean_max":
            maximum = values.masked_fill(mask <= 0, torch.finfo(values.dtype).min).max(2).values
            return torch.cat([mean, maximum], -1)
        return mean


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
