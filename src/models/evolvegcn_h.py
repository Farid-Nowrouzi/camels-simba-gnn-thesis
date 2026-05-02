from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvolveGCNHLayer(nn.Module):
    """
    CAMELS-adapted EvolveGCN-H layer.

    Original EvolveGCN-H idea:
        Instead of evolving node embeddings through time,
        the model evolves the GCN weight matrix through time.

    Here:
        - Each graph snapshot has node features X_t and adjacency A_t.
        - A GRUCell evolves the GCN weight matrix W_t.
        - Then a dense GCN operation is applied:

              H_t = A_norm_t @ X_t @ W_t

    This version is adapted for our CAMELS-SIMBA format:

        A_seq:    [batch, time, nodes, nodes]
        X_seq:    [batch, time, nodes, features]
        mask_seq: [batch, time, nodes, 1]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout)

        # Initial GCN weight matrix W_0.
        self.initial_weight = nn.Parameter(
            torch.empty(in_features, out_features)
        )

        # GRU hidden state is the flattened GCN weight matrix.
        self.weight_hidden_size = in_features * out_features

        # The GRU receives a graph-level summary vector of size in_features.
        self.weight_gru = nn.GRUCell(
            input_size=in_features,
            hidden_size=self.weight_hidden_size,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.initial_weight)

        for name, param in self.weight_gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    @staticmethod
    def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
        """
        Symmetric adjacency normalization:

            A_norm = D^(-1/2) A D^(-1/2)

        Input:
            A: [batch, nodes, nodes]

        Output:
            A_norm: [batch, nodes, nodes]
        """
        degree = A.sum(dim=-1)

        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        degree_inv_sqrt = torch.where(
            torch.isfinite(degree_inv_sqrt),
            degree_inv_sqrt,
            torch.zeros_like(degree_inv_sqrt),
        )

        D_left = degree_inv_sqrt.unsqueeze(-1)
        D_right = degree_inv_sqrt.unsqueeze(-2)

        return D_left * A * D_right

    @staticmethod
    def masked_mean_pool(X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute graph-level mean feature vector using the node mask.

        Input:
            X:    [batch, nodes, features]
            mask: [batch, nodes, 1]

        Output:
            pooled: [batch, features]
        """
        mask = mask.float()

        masked_X = X * mask
        denominator = mask.sum(dim=1).clamp(min=1.0)

        return masked_X.sum(dim=1) / denominator

    def forward(
        self,
        A_seq: torch.Tensor,
        X_seq: torch.Tensor,
        mask_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for one EvolveGCN-H layer.

        Input:
            A_seq:    [batch, time, nodes, nodes]
            X_seq:    [batch, time, nodes, in_features]
            mask_seq: [batch, time, nodes, 1]

        Output:
            H_seq: [batch, time, nodes, out_features]
        """
        batch_size, num_timesteps, num_nodes, _ = X_seq.shape
        device = X_seq.device

        # Repeat the initial weight matrix for each universe in the batch.
        weight_state = self.initial_weight.reshape(1, -1).repeat(batch_size, 1)
        weight_state = weight_state.to(device)

        outputs = []

        for t in range(num_timesteps):
            A_t = A_seq[:, t, :, :]
            X_t = X_seq[:, t, :, :]
            mask_t = mask_seq[:, t, :, :]

            # Graph-level summary used to evolve the GCN weight matrix.
            graph_summary = self.masked_mean_pool(X_t, mask_t)

            # Evolve flattened GCN weight matrix.
            weight_state = self.weight_gru(graph_summary, weight_state)

            W_t = weight_state.view(
                batch_size,
                self.in_features,
                self.out_features,
            )

            A_norm_t = self.normalize_adjacency(A_t)

            # Dense GCN operation:
            # support = X_t @ W_t
            # output  = A_norm_t @ support
            support = torch.bmm(X_t, W_t)
            H_t = torch.bmm(A_norm_t, support)

            H_t = H_t * mask_t
            H_t = self.dropout(H_t)

            outputs.append(H_t)

        H_seq = torch.stack(outputs, dim=1)

        return H_seq


class EvolveGCNHRegressor(nn.Module):
    """
    CAMELS-SIMBA EvolveGCN-H regressor.

    Task:
        Input:
            A sequence of halo graphs for one universe.

        Output:
            One scalar prediction:
                Omega_m

    Input format:
        A_seq:    [batch, time, nodes, nodes]
        X_seq:    [batch, time, nodes, features]
        mask_seq: [batch, time, nodes, 1]

    Output:
        prediction: [batch, 1]
    """

    def __init__(
        self,
        node_features: int = 7,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        temporal_pooling: Literal["mean", "last"] = "mean",
        graph_pooling: Literal["mean", "sum"] = "mean",
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        if temporal_pooling not in {"mean", "last"}:
            raise ValueError("temporal_pooling must be either 'mean' or 'last'.")

        if graph_pooling not in {"mean", "sum"}:
            raise ValueError("graph_pooling must be either 'mean' or 'sum'.")

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.temporal_pooling = temporal_pooling
        self.graph_pooling = graph_pooling

        layers = []

        for layer_idx in range(num_layers):
            in_dim = node_features if layer_idx == 0 else hidden_dim

            layers.append(
                EvolveGCNHLayer(
                    in_features=in_dim,
                    out_features=hidden_dim,
                    dropout=dropout,
                )
            )

        self.layers = nn.ModuleList(layers)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def masked_graph_pool(
        X_seq: torch.Tensor,
        mask_seq: torch.Tensor,
        mode: Literal["mean", "sum"] = "mean",
    ) -> torch.Tensor:
        """
        Pool node embeddings into graph embeddings.

        Input:
            X_seq:    [batch, time, nodes, hidden_dim]
            mask_seq: [batch, time, nodes, 1]

        Output:
            graph_seq: [batch, time, hidden_dim]
        """
        mask_seq = mask_seq.float()

        masked_X = X_seq * mask_seq
        summed = masked_X.sum(dim=2)

        if mode == "sum":
            return summed

        denominator = mask_seq.sum(dim=2).clamp(min=1.0)
        return summed / denominator

    def temporal_pool(
        self,
        graph_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool graph embeddings across time.

        Input:
            graph_seq: [batch, time, hidden_dim]

        Output:
            universe_embedding: [batch, hidden_dim]
        """
        if self.temporal_pooling == "last":
            return graph_seq[:, -1, :]

        return graph_seq.mean(dim=1)

    def forward(
        self,
        A_seq: torch.Tensor,
        X_seq: torch.Tensor,
        mask_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Input:
            A_seq:    [batch, time, nodes, nodes]
            X_seq:    [batch, time, nodes, node_features]
            mask_seq: [batch, time, nodes, 1]

        Output:
            prediction: [batch, 1]
        """
        H_seq = X_seq

        for layer in self.layers:
            H_seq = layer(A_seq=A_seq, X_seq=H_seq, mask_seq=mask_seq)
            H_seq = F.relu(H_seq)

        graph_seq = self.masked_graph_pool(
            X_seq=H_seq,
            mask_seq=mask_seq,
            mode=self.graph_pooling,
        )

        universe_embedding = self.temporal_pool(graph_seq)

        prediction = self.regressor(universe_embedding)

        return prediction


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable model parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)