from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_dense_adjacency(A: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """
    Symmetric normalization for dense adjacency matrices.

    Input:
        A shape: [B, N, N]

    Output:
        A_norm shape: [B, N, N]

    Formula:
        A_hat = A + I
        A_norm = D^(-1/2) A_hat D^(-1/2)
    """
    if A.dim() != 3:
        raise ValueError(f"Expected A with shape [B, N, N], got {tuple(A.shape)}")

    B, N, N2 = A.shape

    if N != N2:
        raise ValueError(f"Adjacency must be square, got {tuple(A.shape)}")

    A_hat = A.float()

    if add_self_loops:
        I = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, N, N)
        A_hat = A_hat + I

    degree = A_hat.sum(dim=-1)

    degree_inv_sqrt = torch.pow(degree.clamp(min=1e-8), -0.5)

    A_norm = (
        degree_inv_sqrt.unsqueeze(-1)
        * A_hat
        * degree_inv_sqrt.unsqueeze(-2)
    )

    return A_norm


def masked_mean_pool(
    X: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: int = 1,
) -> torch.Tensor:
    """
    Mean-pool node features while respecting masks.

    X:
        [B, N, F]

    mask:
        [B, N, 1]

    Output:
        [B, F]
    """
    if mask is None:
        return X.mean(dim=dim)

    mask = mask.float()

    weighted_sum = (X * mask).sum(dim=dim)
    valid_count = mask.sum(dim=dim).clamp(min=1.0)

    return weighted_sum / valid_count


class EvolveGCNHLayer(nn.Module):
    """
    EvolveGCN-H style dense graph convolution layer.

    Main idea:
        Instead of using a fixed GCN weight matrix at every snapshot,
        the GCN weight matrix evolves over time using a GRUCell.

    At each time step:
        1. summarize current node features
        2. update the GCN weight matrix using GRUCell
        3. apply graph convolution using the evolved weight

    This is suitable for our current CAMELS format:

        A_seq:    [B, T, N, N]
        X_seq:    [B, T, N, F]
        mask_seq: [B, T, N, 1]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.weight_size = in_features * out_features

        self.initial_weight = nn.Parameter(
            torch.empty(in_features, out_features)
        )

        self.bias = nn.Parameter(torch.zeros(out_features))

        self.weight_evolver = nn.GRUCell(
            input_size=in_features,
            hidden_size=self.weight_size,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.initial_weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        A_seq: torch.Tensor,
        X_seq: torch.Tensor,
        mask_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Inputs:
            A_seq:
                [B, T, N, N]

            X_seq:
                [B, T, N, F_in]

            mask_seq:
                [B, T, N, 1]

        Output:
            H_seq:
                [B, T, N, F_out]
        """
        if A_seq.dim() != 4:
            raise ValueError(f"Expected A_seq [B, T, N, N], got {tuple(A_seq.shape)}")

        if X_seq.dim() != 4:
            raise ValueError(f"Expected X_seq [B, T, N, F], got {tuple(X_seq.shape)}")

        B, T, N, N2 = A_seq.shape
        Bx, Tx, Nx, Fin = X_seq.shape

        if B != Bx or T != Tx or N != Nx:
            raise ValueError(
                "A_seq and X_seq dimensions do not match: "
                f"A_seq={tuple(A_seq.shape)}, X_seq={tuple(X_seq.shape)}"
            )

        if N != N2:
            raise ValueError(f"A_seq must contain square adjacency matrices, got {tuple(A_seq.shape)}")

        if Fin != self.in_features:
            raise ValueError(
                f"Expected input feature size {self.in_features}, got {Fin}"
            )

        current_weight = self.initial_weight.reshape(1, -1).expand(B, -1)

        outputs = []

        for t in range(T):
            A_t = A_seq[:, t]
            X_t = X_seq[:, t]

            if mask_seq is not None:
                mask_t = mask_seq[:, t]
            else:
                mask_t = None

            graph_summary = masked_mean_pool(X_t, mask_t, dim=1)

            current_weight = self.weight_evolver(
                graph_summary,
                current_weight,
            )

            W_t = current_weight.reshape(B, self.in_features, self.out_features)

            A_norm = normalize_dense_adjacency(A_t, add_self_loops=True)

            AX = torch.bmm(A_norm, X_t)

            H_t = torch.bmm(AX, W_t) + self.bias

            if self.activation:
                H_t = F.relu(H_t)

            H_t = self.dropout(H_t)

            outputs.append(H_t)

        H_seq = torch.stack(outputs, dim=1)

        return H_seq


class EvolveGCNHRegressor(nn.Module):
    """
    EvolveGCN-H regression model for predicting Omega_m from temporal halo graphs.

    Input:
        A_seq:
            [B, T, N, N]

        X_seq:
            [B, T, N, F]

        mask_seq:
            [B, T, N, 1]

    Output:
        prediction:
            [B, 1]
    """

    def __init__(
        self,
        node_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        temporal_pooling: str = "mean",
    ) -> None:
        super().__init__()

        if num_layers not in [1, 2]:
            raise ValueError("This implementation currently supports num_layers=1 or num_layers=2.")

        if temporal_pooling not in ["mean", "last"]:
            raise ValueError("temporal_pooling must be either 'mean' or 'last'.")

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.temporal_pooling = temporal_pooling

        self.layer1 = EvolveGCNHLayer(
            in_features=node_features,
            out_features=hidden_dim,
            activation=True,
            dropout=dropout,
        )

        if num_layers == 2:
            self.layer2 = EvolveGCNHLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                activation=True,
                dropout=dropout,
            )
        else:
            self.layer2 = None

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        A_seq: torch.Tensor,
        X_seq: torch.Tensor,
        mask_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            prediction with shape [B, 1]
        """
        H_seq = self.layer1(A_seq, X_seq, mask_seq)

        if self.layer2 is not None:
            H_seq = self.layer2(A_seq, H_seq, mask_seq)

        graph_embeddings = []

        T = H_seq.shape[1]

        for t in range(T):
            H_t = H_seq[:, t]

            if mask_seq is not None:
                mask_t = mask_seq[:, t]
            else:
                mask_t = None

            graph_emb_t = masked_mean_pool(H_t, mask_t, dim=1)
            graph_embeddings.append(graph_emb_t)

        graph_embeddings = torch.stack(graph_embeddings, dim=1)

        if self.temporal_pooling == "mean":
            universe_embedding = graph_embeddings.mean(dim=1)
        elif self.temporal_pooling == "last":
            universe_embedding = graph_embeddings[:, -1]
        else:
            raise ValueError(f"Unknown temporal_pooling: {self.temporal_pooling}")

        prediction = self.regressor(universe_embedding)

        return prediction


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable model parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = EvolveGCNHRegressor(
        node_features=7,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        temporal_pooling="mean",
    )

    B = 4
    T = 5
    N = 100
    F_in = 7

    A_seq = torch.randint(0, 2, (B, T, N, N)).float()
    A_seq = torch.maximum(A_seq, A_seq.transpose(-1, -2))

    X_seq = torch.randn(B, T, N, F_in)
    mask_seq = torch.ones(B, T, N, 1)

    prediction = model(A_seq, X_seq, mask_seq)

    print("EvolveGCN-H model test")
    print("=" * 80)
    print("A_seq shape:       ", A_seq.shape)
    print("X_seq shape:       ", X_seq.shape)
    print("mask_seq shape:    ", mask_seq.shape)
    print("prediction shape:  ", prediction.shape)
    print("parameters:        ", count_parameters(model))