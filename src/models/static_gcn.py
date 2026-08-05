from __future__ import annotations

"""
static_gcn.py

Static Graph Convolutional Network models for CAMELS-SIMBA halo graphs.

Purpose
-------
This file defines a clean, reproducible static GNN regression model for predicting
the cosmological parameter Omega_m from one graph per universe.

Input format
------------
The model expects a batch of static graph samples:

    A:    Tensor [batch_size, num_nodes, num_nodes]
          Dense adjacency matrix.

    X:    Tensor [batch_size, num_nodes, node_features]
          Node feature matrix.
          Official feature order:
              [log10_Mvir, X, Y, Z, VX, VY, VZ]

    mask: Tensor [batch_size, num_nodes, 1]
          Node mask.
          1 = real halo node
          0 = padded node

Output format
-------------
    prediction: Tensor [batch_size, 1]

Scientific role
---------------
This model is the static GNN baseline.

It uses only one snapshot per universe, usually the final snapshot a = 1.0.
It is compared against:

    1. Mean Omega_m baseline
    2. Static GNN model
    3. Temporal EvolveGCN-H model

Implementation choice
---------------------
This file does not depend on PyTorch Geometric.

It uses dense adjacency matrices directly because the current preprocessing
pipeline already saves each graph as a dense adjacency matrix A [N, N].

The graph convolution is:

    H' = D^(-1/2) A_hat D^(-1/2) H W

where:

    A_hat = A + I

This is the standard GCN-style normalized message passing operation.
"""

import math
from typing import Any, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================
# Utility functions
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a PyTorch model.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def validate_static_batch(
    A: torch.Tensor,
    X: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> None:
    """
    Validate tensor shapes before the forward pass.

    Expected:
        A:    [B, N, N]
        X:    [B, N, F]
        mask: [B, N, 1] or None
    """
    if A.ndim != 3:
        raise ValueError(f"Expected A to have shape [B, N, N], got {tuple(A.shape)}")

    if X.ndim != 3:
        raise ValueError(f"Expected X to have shape [B, N, F], got {tuple(X.shape)}")

    if A.shape[0] != X.shape[0]:
        raise ValueError(
            f"Batch size mismatch: A batch={A.shape[0]}, X batch={X.shape[0]}"
        )

    if A.shape[1] != A.shape[2]:
        raise ValueError(f"Expected square adjacency matrix, got {tuple(A.shape)}")

    if A.shape[1] != X.shape[1]:
        raise ValueError(
            f"Node count mismatch: A nodes={A.shape[1]}, X nodes={X.shape[1]}"
        )

    if mask is not None:
        if mask.ndim != 3:
            raise ValueError(
                f"Expected mask to have shape [B, N, 1], got {tuple(mask.shape)}"
            )

        if mask.shape[0] != X.shape[0] or mask.shape[1] != X.shape[1]:
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} is incompatible with "
                f"X shape {tuple(X.shape)}"
            )

        if mask.shape[2] != 1:
            raise ValueError(f"Expected mask last dimension to be 1, got {mask.shape[2]}")


def normalize_adjacency(
    A: torch.Tensor,
    add_self_loops: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Symmetrically normalize a dense adjacency matrix.

    Input:
        A: Tensor [B, N, N]

    Output:
        A_norm: Tensor [B, N, N]

    Formula:
        A_hat = A + I
        A_norm = D^(-1/2) A_hat D^(-1/2)
    """
    if A.ndim != 3:
        raise ValueError(f"Expected A shape [B, N, N], got {tuple(A.shape)}")

    batch_size, num_nodes, num_nodes_2 = A.shape

    if num_nodes != num_nodes_2:
        raise ValueError(f"Expected square adjacency matrix, got {tuple(A.shape)}")

    A_hat = A.float()

    if add_self_loops:
        identity = torch.eye(
            num_nodes,
            dtype=A_hat.dtype,
            device=A_hat.device,
        ).unsqueeze(0)

        A_hat = A_hat + identity

    degree = A_hat.sum(dim=-1)

    degree_inv_sqrt = torch.pow(degree + eps, -0.5)

    A_norm = (
        degree_inv_sqrt.unsqueeze(-1)
        * A_hat
        * degree_inv_sqrt.unsqueeze(-2)
    )

    return A_norm


def normalize_sparse_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
    add_self_loops: bool = True,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Coalesce and symmetrically normalize sparse edges like dense `A + I`."""
    edge_index = edge_index.long()
    values = (
        torch.ones(edge_index.shape[1], device=edge_index.device, dtype=torch.float32)
        if edge_weight is None else edge_weight.float()
    )
    if add_self_loops:
        nodes = torch.arange(num_nodes, device=edge_index.device)
        edge_index = torch.cat([edge_index, torch.stack([nodes, nodes])], dim=1)
        values = torch.cat([values, torch.ones(num_nodes, device=values.device)])
    sparse = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).coalesce()
    indices = sparse.indices()
    values = sparse.values()
    # edge convention: source -> target; row/target degree matches dense A.sum(-1)
    source, target = indices[0], indices[1]
    degree = torch.zeros(num_nodes, device=values.device, dtype=values.dtype)
    degree.index_add_(0, target, values)
    norm = values * (degree[target] + eps).pow(-0.5) * (degree[source] + eps).pow(-0.5)
    return indices, norm


def sparse_message(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    source, target = edge_index[0], edge_index[1]
    output = torch.zeros_like(features)
    output.index_add_(0, target, features[source] * edge_weight.unsqueeze(-1))
    return output


def sparse_graph_pool(
    features: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    mode: str,
) -> torch.Tensor:
    summed = torch.zeros((num_graphs, features.shape[-1]), device=features.device, dtype=features.dtype)
    summed.index_add_(0, batch, features)
    counts = torch.bincount(batch, minlength=num_graphs).to(features.dtype).clamp(min=1).unsqueeze(-1)
    mean = summed / counts
    if mode == "mean":
        return mean
    max_values = torch.full_like(summed, torch.finfo(features.dtype).min)
    max_values.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(features), features, reduce="amax", include_self=True)
    if mode == "max":
        return max_values
    if mode == "mean_max":
        return torch.cat([mean, max_values], dim=-1)
    raise ValueError(f"Unknown graph pooling mode: {mode}")


def masked_mean_pool(
    node_embeddings: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Mean-pool node embeddings into one graph embedding per universe.

    Input:
        node_embeddings: Tensor [B, N, H]
        mask:            Tensor [B, N, 1] or None

    Output:
        graph_embedding: Tensor [B, H]
    """
    if node_embeddings.ndim != 3:
        raise ValueError(
            f"Expected node_embeddings shape [B, N, H], "
            f"got {tuple(node_embeddings.shape)}"
        )

    if mask is None:
        return node_embeddings.mean(dim=1)

    if mask.ndim != 3:
        raise ValueError(f"Expected mask shape [B, N, 1], got {tuple(mask.shape)}")

    mask = mask.float()

    masked_embeddings = node_embeddings * mask
    node_counts = mask.sum(dim=1).clamp(min=eps)

    graph_embedding = masked_embeddings.sum(dim=1) / node_counts

    return graph_embedding


def masked_max_pool(
    node_embeddings: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Max-pool node embeddings into one graph embedding per universe.

    Input:
        node_embeddings: Tensor [B, N, H]
        mask:            Tensor [B, N, 1] or None

    Output:
        graph_embedding: Tensor [B, H]
    """
    if node_embeddings.ndim != 3:
        raise ValueError(
            f"Expected node_embeddings shape [B, N, H], "
            f"got {tuple(node_embeddings.shape)}"
        )

    if mask is None:
        return node_embeddings.max(dim=1).values

    mask = mask.float()

    very_negative = torch.finfo(node_embeddings.dtype).min
    masked_embeddings = node_embeddings.masked_fill(mask <= 0, very_negative)

    return masked_embeddings.max(dim=1).values


# ============================================================
# Dense GCN layer
# ============================================================

class DenseGCNLayer(nn.Module):
    """
    Dense GCN layer for adjacency matrices saved as [B, N, N].

    Operation:
        H_message = A_norm @ H
        H_out = Linear(H_message)

    Optional:
        activation
        dropout
        layer normalization
        residual connection
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        activation: Optional[nn.Module] = None,
        use_layer_norm: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.use_layer_norm = use_layer_norm
        self.use_residual = residual and input_dim == output_dim

        self.linear = nn.Linear(input_dim, output_dim)

        self.activation = activation if activation is not None else nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = (
            nn.LayerNorm(output_dim)
            if use_layer_norm
            else nn.Identity()
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize layer weights.
        """
        nn.init.xavier_uniform_(self.linear.weight)

        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        A_norm: torch.Tensor,
        H: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            A_norm: normalized adjacency [B, N, N]
            H:      node embeddings [B, N, input_dim]
            mask:   node mask [B, N, 1]

        Returns:
            H_out: node embeddings [B, N, output_dim]
        """
        H_in = H

        H_message = torch.bmm(A_norm, H)
        H_out = self.linear(H_message)
        H_out = self.activation(H_out)
        H_out = self.dropout(H_out)
        H_out = self.layer_norm(H_out)

        if self.use_residual:
            H_out = H_out + H_in

        if mask is not None:
            H_out = H_out * mask.float()

        return H_out


# ============================================================
# Dense GraphSAGE layer
# ============================================================

class DenseGraphSAGELayer(nn.Module):
    """
    Dense GraphSAGE layer for adjacency matrices saved as [B, N, N].

    Operation:
        H_neighbors = mean_{j in N(i)} H_j
        H_combined = concat(H_i, H_neighbors)
        H_out = Linear(H_combined)

    Optional:
        activation
        dropout
        layer normalization
        residual connection
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        activation: Optional[nn.Module] = None,
        use_layer_norm: bool = True,
        residual: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.use_layer_norm = use_layer_norm
        self.use_residual = residual and input_dim == output_dim
        self.eps = eps

        self.linear = nn.Linear(input_dim * 2, output_dim)

        self.activation = activation if activation is not None else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = (
            nn.LayerNorm(output_dim)
            if use_layer_norm
            else nn.Identity()
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize layer weights.
        """
        nn.init.xavier_uniform_(self.linear.weight)

        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        A: torch.Tensor,
        H: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            A:    adjacency matrix [B, N, N]
            H:    node embeddings [B, N, input_dim]
            mask: node mask [B, N, 1]

        Returns:
            H_out: node embeddings [B, N, output_dim]
        """
        H_in = H
        A_float = A.float()

        if mask is not None:
            mask_float = mask.float()
            A_float = A_float * mask_float * mask_float.transpose(1, 2)
            H = H * mask_float

        degree = A_float.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        neighbor_sum = torch.bmm(A_float, H)
        neighbor_mean = neighbor_sum / degree

        combined = torch.cat([H, neighbor_mean], dim=-1)

        H_out = self.linear(combined)
        H_out = self.activation(H_out)
        H_out = self.dropout(H_out)
        H_out = self.layer_norm(H_out)

        if self.use_residual:
            H_out = H_out + H_in

        if mask is not None:
            H_out = H_out * mask.float()

        return H_out


# ============================================================
# Static GCN regressor
# ============================================================

class StaticGCNRegressor(nn.Module):
    """
    Static GCN regressor for CAMELS-SIMBA graph-level Omega_m prediction.

    This model processes one graph per universe.

    Architecture:
        1. Normalize adjacency matrix.
        2. Project node features into hidden dimension.
        3. Apply several dense GCN layers.
        4. Masked graph pooling.
        5. MLP regression head.
    """

    def __init__(
        self,
        node_features: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        graph_pooling: Literal["mean", "max", "mean_max"] = "mean",
        conv_type: Literal["gcn", "graphsage"] = "gcn",
        add_self_loops: bool = True,
        use_layer_norm: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()

        if node_features <= 0:
            raise ValueError("node_features must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        if graph_pooling not in {"mean", "max", "mean_max"}:
            raise ValueError(
                "graph_pooling must be one of: 'mean', 'max', 'mean_max'."
            )

        if conv_type not in {"gcn", "graphsage"}:
            raise ValueError("conv_type must be one of: 'gcn', 'graphsage'.")

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.graph_pooling = graph_pooling
        self.conv_type = conv_type
        self.add_self_loops = add_self_loops
        self.use_layer_norm = use_layer_norm
        self.residual = residual

        self.input_projection = nn.Linear(node_features, hidden_dim)

        self.input_activation = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout)
        self.input_norm = (
            nn.LayerNorm(hidden_dim)
            if use_layer_norm
            else nn.Identity()
        )

        layer_cls = DenseGCNLayer if conv_type == "gcn" else DenseGraphSAGELayer

        self.layers = nn.ModuleList(
            [
                layer_cls(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    dropout=dropout,
                    activation=nn.ReLU(),
                    use_layer_norm=use_layer_norm,
                    residual=residual,
                )
                for _ in range(num_layers)
            ]
        )

        if graph_pooling == "mean_max":
            pooled_dim = hidden_dim * 2
        else:
            pooled_dim = hidden_dim

        self.regressor = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim >= 2 else hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2 if hidden_dim >= 2 else hidden_dim, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize model weights.
        """
        nn.init.xavier_uniform_(self.input_projection.weight)

        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)

        for module in self.regressor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_nodes(
        self,
        A: torch.Tensor | Dict[str, Any],
        X: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode node features using dense GCN message passing.

        Args:
            A:    adjacency matrix [B, N, N]
            X:    node features [B, N, F]
            mask: node mask [B, N, 1]

        Returns:
            H: node embeddings [B, N, hidden_dim]
        """
        if isinstance(A, dict):
            if self.conv_type != "gcn":
                raise ValueError("Sparse Static support preserves the canonical conv_type='gcn' only.")
            graph = A
            X_sparse = graph["x"].float()
            H = self.input_norm(self.input_dropout(self.input_activation(self.input_projection(X_sparse))))
            edge_index, edge_weight = normalize_sparse_edges(
                graph["edge_index"], H.shape[0], graph.get("edge_weight"), self.add_self_loops
            )
            for layer in self.layers:
                H_in = H
                H = layer.linear(sparse_message(edge_index, edge_weight, H))
                H = layer.layer_norm(layer.dropout(layer.activation(H)))
                if layer.use_residual:
                    H = H + H_in
            return H

        if X is None:
            raise ValueError("Dense Static input requires X.")
        validate_static_batch(A=A, X=X, mask=mask)

        if self.conv_type == "gcn":
            A_message = normalize_adjacency(
                A=A,
                add_self_loops=self.add_self_loops,
            )
        else:
            A_message = A.float()

        H = self.input_projection(X.float())
        H = self.input_activation(H)
        H = self.input_dropout(H)
        H = self.input_norm(H)

        if mask is not None:
            H = H * mask.float()

        for layer in self.layers:
            if self.conv_type == "gcn":
                H = layer(A_norm=A_message, H=H, mask=mask)
            else:
                H = layer(A=A_message, H=H, mask=mask)

        return H

    def pool_graph(
        self,
        node_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert node embeddings into one graph embedding.
        """
        if self.graph_pooling == "mean":
            return masked_mean_pool(node_embeddings, mask)

        if self.graph_pooling == "max":
            return masked_max_pool(node_embeddings, mask)

        if self.graph_pooling == "mean_max":
            mean_pool = masked_mean_pool(node_embeddings, mask)
            max_pool = masked_max_pool(node_embeddings, mask)
            return torch.cat([mean_pool, max_pool], dim=-1)

        raise ValueError(f"Unknown graph pooling mode: {self.graph_pooling}")

    def forward(
        self,
        A: torch.Tensor | Dict[str, Any],
        X: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            A:    Tensor [B, N, N]
            X:    Tensor [B, N, F]
            mask: Tensor [B, N, 1]

        Returns:
            prediction: Tensor [B, 1]
        """
        node_embeddings = self.encode_nodes(A=A, X=X, mask=mask)
        if isinstance(A, dict):
            graph_embedding = sparse_graph_pool(
                node_embeddings, A["batch"].long(), int(A["num_graphs"]), self.graph_pooling
            )
        else:
            graph_embedding = self.pool_graph(node_embeddings, mask)
        prediction = self.regressor(graph_embedding)

        return prediction


# ============================================================
# Lightweight smoke test
# ============================================================

def _smoke_test() -> None:
    """
    Small internal test to verify tensor shapes.
    """
    batch_size = 4
    num_nodes = 100
    node_features = 7

    A = torch.randint(
        low=0,
        high=2,
        size=(batch_size, num_nodes, num_nodes),
    ).float()

    # Make adjacency symmetric and remove self-loops.
    A = ((A + A.transpose(1, 2)) > 0).float()

    identity = torch.eye(num_nodes).unsqueeze(0)
    A = A * (1.0 - identity)

    X = torch.rand(batch_size, num_nodes, node_features)
    mask = torch.ones(batch_size, num_nodes, 1)

    model = StaticGCNRegressor(
        node_features=node_features,
        hidden_dim=32,
        num_layers=2,
        dropout=0.1,
        graph_pooling="mean",
    )

    prediction = model(A=A, X=X, mask=mask)

    print("=" * 80)
    print("StaticGCNRegressor smoke test")
    print("=" * 80)
    print(f"A shape:          {tuple(A.shape)}")
    print(f"X shape:          {tuple(X.shape)}")
    print(f"mask shape:       {tuple(mask.shape)}")
    print(f"prediction shape: {tuple(prediction.shape)}")
    print(f"parameters:       {count_parameters(model)}")
    print("=" * 80)

    assert prediction.shape == (batch_size, 1)

    print("✅ Smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
