from __future__ import annotations

"""
evolvegcn_h.py

EvolveGCN-H style temporal graph regression model for CAMELS-SIMBA halo graphs.

Purpose
-------
This model predicts one cosmological parameter, currently Omega_m, from a temporal
sequence of graph snapshots.

Input format:
    A_seq:
        Dense adjacency sequence with shape [B, T, N, N]

    X_seq:
        Node feature sequence with shape [B, T, N, F]

    mask_seq:
        Node mask sequence with shape [B, T, N, 1]

Output:
    prediction:
        Scalar regression output with shape [B, 1]

Current thesis setting:
    - Universe = one sample
    - Snapshot = one graph at one cosmic time
    - Nodes = halos
    - Node features = [log10(Mvir), X, Y, Z, VX, VY, VZ]
    - Edges = periodic boundary-aware kNN graph
    - Target = Omega_m
"""

from typing import Literal, Optional

import torch
import torch.nn as nn


# ============================================================
# Utility functions
# ============================================================

def normalize_dense_adjacency(
    A: torch.Tensor,
    add_self_loops: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Symmetric normalization for dense adjacency matrices.

    Formula:
        A_hat = A + I
        A_norm = D^(-1/2) A_hat D^(-1/2)

    Args:
        A:
            Dense adjacency matrix with shape [B, N, N].

        add_self_loops:
            If True, add identity matrix before normalization.

        eps:
            Small numerical constant to avoid division by zero.

    Returns:
        Normalized dense adjacency matrix with shape [B, N, N].
    """
    if A.dim() != 3:
        raise ValueError(
            f"Expected A with shape [B, N, N], got {tuple(A.shape)}"
        )

    batch_size, num_nodes, num_nodes_2 = A.shape

    if num_nodes != num_nodes_2:
        raise ValueError(
            f"Adjacency matrices must be square, got {tuple(A.shape)}"
        )

    A_hat = A.float()

    if add_self_loops:
        identity = torch.eye(
            num_nodes,
            device=A.device,
            dtype=A_hat.dtype,
        ).unsqueeze(0).expand(batch_size, num_nodes, num_nodes)

        A_hat = A_hat + identity

    degree = A_hat.sum(dim=-1)
    degree_inv_sqrt = torch.pow(degree.clamp(min=eps), -0.5)

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
    Mean-pool node embeddings while respecting a node mask.

    Args:
        X:
            Tensor to pool. Common shape: [B, N, F].

        mask:
            Optional mask. Common shape: [B, N, 1].
            Values should be 1 for real nodes and 0 for padded nodes.

        dim:
            Dimension over which to pool.

    Returns:
        Pooled tensor.
    """
    if mask is None:
        return X.mean(dim=dim)

    mask = mask.float()

    weighted_sum = (X * mask).sum(dim=dim)
    valid_count = mask.sum(dim=dim).clamp(min=1.0)

    return weighted_sum / valid_count


def validate_temporal_inputs(
    A_seq: torch.Tensor,
    X_seq: torch.Tensor,
    mask_seq: Optional[torch.Tensor],
    expected_features: Optional[int] = None,
) -> None:
    """
    Validate temporal graph tensor shapes before model computation.

    Expected:
        A_seq:    [B, T, N, N]
        X_seq:    [B, T, N, F]
        mask_seq: [B, T, N, 1]
    """
    if A_seq.dim() != 4:
        raise ValueError(
            f"Expected A_seq with shape [B, T, N, N], got {tuple(A_seq.shape)}"
        )

    if X_seq.dim() != 4:
        raise ValueError(
            f"Expected X_seq with shape [B, T, N, F], got {tuple(X_seq.shape)}"
        )

    batch_a, time_a, nodes_a, nodes_a_2 = A_seq.shape
    batch_x, time_x, nodes_x, features_x = X_seq.shape

    if nodes_a != nodes_a_2:
        raise ValueError(
            f"A_seq must contain square adjacency matrices, got {tuple(A_seq.shape)}"
        )

    if batch_a != batch_x or time_a != time_x or nodes_a != nodes_x:
        raise ValueError(
            "A_seq and X_seq dimensions do not match: "
            f"A_seq={tuple(A_seq.shape)}, X_seq={tuple(X_seq.shape)}"
        )

    if expected_features is not None and features_x != expected_features:
        raise ValueError(
            f"Expected node feature size {expected_features}, got {features_x}"
        )

    if mask_seq is not None:
        if mask_seq.dim() != 4:
            raise ValueError(
                f"Expected mask_seq with shape [B, T, N, 1], got {tuple(mask_seq.shape)}"
            )

        batch_m, time_m, nodes_m, mask_features = mask_seq.shape

        if batch_m != batch_a or time_m != time_a or nodes_m != nodes_a:
            raise ValueError(
                "mask_seq dimensions do not match A_seq/X_seq: "
                f"mask_seq={tuple(mask_seq.shape)}, "
                f"A_seq={tuple(A_seq.shape)}, X_seq={tuple(X_seq.shape)}"
            )

        if mask_features != 1:
            raise ValueError(
                f"Expected mask_seq final dimension to be 1, got {mask_features}"
            )


# ============================================================
# EvolveGCN-H layer
# ============================================================

class EvolveGCNHLayer(nn.Module):
    """
    EvolveGCN-H style dense graph convolution layer.

    Main idea:
        Instead of using a fixed GCN weight matrix for every snapshot,
        the GCN weight matrix evolves through time using a GRUCell.

    At each time step:
        1. Create a graph-level summary from the current node features.
        2. Use the graph summary to evolve the GCN weight matrix.
        3. Apply dense graph convolution using the evolved weight.

    This layer is adapted to the CAMELS-SIMBA temporal graph format:

        A_seq:    [B, T, N, N]
        X_seq:    [B, T, N, F_in]
        mask_seq: [B, T, N, 1]
    """

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

        if in_features <= 0:
            raise ValueError("in_features must be positive.")

        if out_features <= 0:
            raise ValueError("out_features must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        if activation_name not in {"relu", "leaky_relu", "elu"}:
            raise ValueError(
                "activation_name must be one of: 'relu', 'leaky_relu', 'elu'."
            )

        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.activation_name = activation_name
        self.add_self_loops = add_self_loops

        self.dropout = nn.Dropout(dropout)
        self.activation_layer = self._build_activation_layer(activation_name)

        self.weight_size = in_features * out_features

        # Initial GCN weight matrix W_0.
        self.initial_weight = nn.Parameter(
            torch.empty(in_features, out_features)
        )

        self.bias = nn.Parameter(torch.zeros(out_features))

        # The GRU hidden state is the flattened GCN weight matrix.
        # The GRU input is a graph-level feature summary.
        self.weight_evolver = nn.GRUCell(
            input_size=in_features,
            hidden_size=self.weight_size,
        )

        self.reset_parameters()

    @staticmethod
    def _build_activation_layer(
        activation_name: Literal["relu", "leaky_relu", "elu"],
    ) -> nn.Module:
        if activation_name == "relu":
            return nn.ReLU()

        if activation_name == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01)

        if activation_name == "elu":
            return nn.ELU()

        raise ValueError(f"Unknown activation_name: {activation_name}")

    def reset_parameters(self) -> None:
        """
        Initialize learnable parameters.
        """
        nn.init.xavier_uniform_(self.initial_weight)
        nn.init.zeros_(self.bias)

        for name, parameter in self.weight_evolver.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def forward(
        self,
        A_seq: torch.Tensor,
        X_seq: torch.Tensor,
        mask_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            A_seq:
                Dense adjacency sequence [B, T, N, N].

            X_seq:
                Node feature sequence [B, T, N, F_in].

            mask_seq:
                Optional node mask sequence [B, T, N, 1].

        Returns:
            H_seq:
                Node embedding sequence [B, T, N, F_out].
        """
        validate_temporal_inputs(
            A_seq=A_seq,
            X_seq=X_seq,
            mask_seq=mask_seq,
            expected_features=self.in_features,
        )

        batch_size, num_timesteps, _, _ = A_seq.shape

        current_weight = self.initial_weight.reshape(1, -1).expand(
            batch_size,
            -1,
        )

        outputs = []

        for timestep in range(num_timesteps):
            A_t = A_seq[:, timestep, :, :]
            X_t = X_seq[:, timestep, :, :]

            if mask_seq is not None:
                mask_t = mask_seq[:, timestep, :, :].float()
            else:
                mask_t = None

            # Summary vector used to evolve the weight matrix.
            graph_summary = masked_mean_pool(
                X=X_t,
                mask=mask_t,
                dim=1,
            )

            current_weight = self.weight_evolver(
                graph_summary,
                current_weight,
            )

            W_t = current_weight.reshape(
                batch_size,
                self.in_features,
                self.out_features,
            )

            A_norm_t = normalize_dense_adjacency(
                A=A_t,
                add_self_loops=self.add_self_loops,
            )

            # Dense GCN operation:
            #   support = X_t @ W_t
            #   H_t = A_norm_t @ support
            support = torch.bmm(X_t, W_t)
            H_t = torch.bmm(A_norm_t, support) + self.bias

            if self.activation:
                H_t = self.activation_layer(H_t)

            H_t = self.dropout(H_t)

            # Very important:
            # keep padded/non-real nodes at zero after convolution.
            if mask_t is not None:
                H_t = H_t * mask_t

            outputs.append(H_t)

        H_seq = torch.stack(outputs, dim=1)

        return H_seq


# ============================================================
# EvolveGCN-H regressor
# ============================================================

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
        node_features: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        activation: Literal["relu", "leaky_relu", "elu"] = "relu",
        temporal_pooling: Literal["mean", "last"] = "mean",
        graph_pooling: Literal["mean", "sum", "mean_max"] = "mean",
        add_self_loops: bool = True,
        summary_feature_dim: int = 0,
        head_type: Literal["mlp", "linear"] = "mlp",
    ) -> None:
        super().__init__()

        if node_features <= 0:
            raise ValueError("node_features must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        if activation not in {"relu", "leaky_relu", "elu"}:
            raise ValueError("activation must be one of: 'relu', 'leaky_relu', 'elu'.")

        if temporal_pooling not in {"mean", "last"}:
            raise ValueError("temporal_pooling must be either 'mean' or 'last'.")

        if graph_pooling not in {"mean", "sum", "mean_max"}:
            raise ValueError(
                "graph_pooling must be one of: 'mean', 'sum', 'mean_max'."
            )

        if summary_feature_dim < 0:
            raise ValueError("summary_feature_dim must be non-negative.")

        if head_type not in {"mlp", "linear"}:
            raise ValueError("head_type must be either 'mlp' or 'linear'.")

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.activation = activation
        self.temporal_pooling = temporal_pooling
        self.graph_pooling = graph_pooling
        self.add_self_loops = add_self_loops
        self.summary_feature_dim = summary_feature_dim
        self.head_type = head_type

        layers = []

        for layer_index in range(num_layers):
            in_dim = node_features if layer_index == 0 else hidden_dim

            layers.append(
                EvolveGCNHLayer(
                    in_features=in_dim,
                    out_features=hidden_dim,
                    activation=True,
                    activation_name=activation,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                )
            )

        self.layers = nn.ModuleList(layers)

        graph_embedding_dim = hidden_dim * 2 if graph_pooling == "mean_max" else hidden_dim
        regressor_input_dim = graph_embedding_dim + summary_feature_dim

        if head_type == "linear":
            self.regressor = nn.Linear(regressor_input_dim, 1)
        else:
            self.regressor = nn.Sequential(
                nn.Linear(regressor_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    @staticmethod
    def masked_graph_pool(
        X_seq: torch.Tensor,
        mask_seq: Optional[torch.Tensor] = None,
        mode: Literal["mean", "sum", "mean_max"] = "mean",
    ) -> torch.Tensor:
        """
        Pool node embeddings into graph embeddings for each time step.

        Args:
            X_seq:
                Node embedding sequence [B, T, N, H].

            mask_seq:
                Optional node mask sequence [B, T, N, 1].

            mode:
                'mean', 'sum', or 'mean_max'.

        Returns:
            graph_seq:
                Graph embedding sequence [B, T, H], or [B, T, 2H] for mean_max.
        """
        if mode not in {"mean", "sum", "mean_max"}:
            raise ValueError("mode must be one of: 'mean', 'sum', 'mean_max'.")

        if mask_seq is None:
            if mode == "sum":
                return X_seq.sum(dim=2)
            if mode == "mean_max":
                mean_pool = X_seq.mean(dim=2)
                max_pool = X_seq.max(dim=2).values
                return torch.cat([mean_pool, max_pool], dim=-1)
            return X_seq.mean(dim=2)

        mask_seq = mask_seq.float()

        masked_X = X_seq * mask_seq
        summed = masked_X.sum(dim=2)

        if mode == "sum":
            return summed

        denominator = mask_seq.sum(dim=2).clamp(min=1.0)
        mean_pool = summed / denominator

        if mode == "mean_max":
            very_negative = torch.finfo(X_seq.dtype).min
            max_input = X_seq.masked_fill(mask_seq <= 0, very_negative)
            max_pool = max_input.max(dim=2).values
            return torch.cat([mean_pool, max_pool], dim=-1)

        return mean_pool

    def temporal_pool(
        self,
        graph_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool graph embeddings across time.

        Args:
            graph_seq:
                [B, T, H]

        Returns:
            universe_embedding:
                [B, H]
        """
        if self.temporal_pooling == "last":
            return graph_seq[:, -1, :]

        if self.temporal_pooling == "mean":
            return graph_seq.mean(dim=1)

        raise ValueError(f"Unknown temporal_pooling: {self.temporal_pooling}")

    def forward(
        self,
        A_seq: torch.Tensor,
        X_seq: torch.Tensor,
        mask_seq: Optional[torch.Tensor] = None,
        summary_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            A_seq:
                Dense adjacency sequence [B, T, N, N].

            X_seq:
                Node feature sequence [B, T, N, F].

            mask_seq:
                Optional node mask sequence [B, T, N, 1].

            summary_features:
                Optional universe-level summary features [B, S].

        Returns:
            prediction:
                Regression prediction [B, 1].
        """
        validate_temporal_inputs(
            A_seq=A_seq,
            X_seq=X_seq,
            mask_seq=mask_seq,
            expected_features=self.node_features,
        )

        H_seq = X_seq

        for layer in self.layers:
            H_seq = layer(
                A_seq=A_seq,
                X_seq=H_seq,
                mask_seq=mask_seq,
            )

        graph_seq = self.masked_graph_pool(
            X_seq=H_seq,
            mask_seq=mask_seq,
            mode=self.graph_pooling,
        )

        universe_embedding = self.temporal_pool(graph_seq)

        if self.summary_feature_dim > 0:
            if summary_features is None:
                raise ValueError(
                    "summary_features must be provided when "
                    "summary_feature_dim > 0."
                )

            if summary_features.dim() != 2:
                raise ValueError(
                    "Expected summary_features with shape [B, S], "
                    f"got {tuple(summary_features.shape)}"
                )

            if summary_features.shape[0] != universe_embedding.shape[0]:
                raise ValueError(
                    "summary_features batch size does not match graph batch: "
                    f"summary_features={tuple(summary_features.shape)}, "
                    f"universe_embedding={tuple(universe_embedding.shape)}"
                )

            if summary_features.shape[1] != self.summary_feature_dim:
                raise ValueError(
                    f"Expected summary feature dim {self.summary_feature_dim}, "
                    f"got {summary_features.shape[1]}"
                )

            universe_embedding = torch.cat(
                [universe_embedding, summary_features.float()],
                dim=-1,
            )

        elif summary_features is not None:
            raise ValueError(
                "summary_features were provided, but summary_feature_dim is 0."
            )

        prediction = self.regressor(universe_embedding)

        return prediction


# ============================================================
# Model utilities
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """
    Count trainable model parameters.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def smoke_test() -> None:
    """
    Lightweight model smoke test.

    This is not a full unit test. It simply verifies that the model can receive
    CAMELS-style tensors and return one scalar prediction per universe.
    """
    print("=" * 80)
    print("EvolveGCN-H smoke test")
    print("=" * 80)

    batch_size = 4
    num_snapshots = 5
    num_nodes = 100
    node_features = 7

    A_seq = torch.randint(
        low=0,
        high=2,
        size=(batch_size, num_snapshots, num_nodes, num_nodes),
    ).float()

    # Make adjacency symmetric.
    A_seq = torch.maximum(A_seq, A_seq.transpose(-1, -2))

    # Remove self-loops from raw adjacency.
    eye = torch.eye(num_nodes).view(1, 1, num_nodes, num_nodes)
    A_seq = A_seq * (1.0 - eye)

    X_seq = torch.randn(
        batch_size,
        num_snapshots,
        num_nodes,
        node_features,
    )

    mask_seq = torch.ones(
        batch_size,
        num_snapshots,
        num_nodes,
        1,
    )

    model = EvolveGCNHRegressor(
        node_features=node_features,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        temporal_pooling="mean",
        graph_pooling="mean",
        add_self_loops=True,
    )

    prediction = model(
        A_seq=A_seq,
        X_seq=X_seq,
        mask_seq=mask_seq,
    )

    print(f"A_seq shape:       {tuple(A_seq.shape)}")
    print(f"X_seq shape:       {tuple(X_seq.shape)}")
    print(f"mask_seq shape:    {tuple(mask_seq.shape)}")
    print(f"prediction shape:  {tuple(prediction.shape)}")
    print(f"parameters:        {count_parameters(model)}")

    expected_shape = (batch_size, 1)

    if tuple(prediction.shape) != expected_shape:
        raise RuntimeError(
            f"Smoke test failed. Expected prediction shape {expected_shape}, "
            f"got {tuple(prediction.shape)}"
        )

    if torch.isnan(prediction).any():
        raise RuntimeError("Smoke test failed. Prediction contains NaN values.")

    if torch.isinf(prediction).any():
        raise RuntimeError("Smoke test failed. Prediction contains Inf values.")

    print("✅ Smoke test passed.")
    print("=" * 80)


if __name__ == "__main__":
    smoke_test()
