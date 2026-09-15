"""Frozen compact periodic 3D CNN for the Top1500 CIC representation."""

from __future__ import annotations

import torch
from torch import nn


EXPECTED_PARAMETERS = 20_017


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class DeterministicAvgPool3d2(nn.Module):
    """Parameter-free non-overlapping 2x2x2 average downsampling."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 5:
            raise ValueError(f"Expected input [B, C, D, H, W], got {tuple(values.shape)}")
        batch, channels, depth, height, width = values.shape
        if depth % 2 or height % 2 or width % 2:
            raise ValueError(
                "DeterministicAvgPool3d2 requires D, H, and W divisible by 2; "
                f"got spatial shape ({depth}, {height}, {width})."
            )
        return values.reshape(
            batch, channels,
            depth // 2, 2,
            height // 2, 2,
            width // 2, 2,
        ).mean(dim=(3, 5, 7))


class StaticCNN3DRegressor(nn.Module):
    """Experiment-1 circular-CNN regressor with a fixed [B,5,32,32,32] contract."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(5, 8, kernel_size=3, stride=1, padding=1, padding_mode="circular", bias=True)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1, padding_mode="circular", bias=True)
        self.pool = DeterministicAvgPool3d2()
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, padding_mode="circular", bias=True)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.head1 = nn.Linear(32, 32)
        self.head2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.2)
        self._initialize_weights()
        if count_parameters(self) != EXPECTED_PARAMETERS:
            raise RuntimeError("Frozen CNN parameter-count contract violated.")

    def _initialize_weights(self) -> None:
        relu_gain = nn.init.calculate_gain("relu")
        for module in (self.conv1, self.conv2, self.conv3, self.head1, self.head2):
            nn.init.xavier_uniform_(module.weight, gain=relu_gain)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output.weight, gain=1.0)
        nn.init.zeros_(self.output.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 5:
            raise ValueError(f"Expected input [B, 5, 32, 32, 32], got {tuple(values.shape)}")
        if tuple(values.shape[1:]) != (5, 32, 32, 32):
            raise ValueError(f"Expected input [B, 5, 32, 32, 32], got {tuple(values.shape)}")
        if not torch.is_floating_point(values):
            raise TypeError("CNN input must have floating dtype.")
        if not torch.isfinite(values).all():
            raise ValueError("CNN input must be finite.")
        values = self.relu(self.conv1(values))
        values = self.relu(self.conv2(values))
        values = self.pool(values)
        values = self.relu(self.conv3(values))
        values = self.flatten(self.global_pool(values))
        values = self.dropout(self.relu(self.head1(values)))
        values = self.dropout(self.relu(self.head2(values)))
        return self.output(values)
