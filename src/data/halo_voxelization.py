"""Frozen periodic CIC halo-catalogue representation for CNN experiment 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from itertools import product
from typing import Sequence

import torch


RAW7_FEATURE_NAMES = ("log10_Mvir", "X", "Y", "Z", "VX", "VY", "VZ")


@dataclass(frozen=True)
class HaloVoxelizationSpec:
    """Scientific specification for the fixed Top1500 final-snapshot field."""

    box_size_mpc_h: float = 25.0
    resolution: int = 32
    deposition: str = "periodic_cic"
    grid_origin: str = "cell_centered"
    channel_order: tuple[str, ...] = (
        "fractional_deposited_halo_count",
        "deposited_log_mass_mark",
        "deposited_VX_mark",
        "deposited_VY_mark",
        "deposited_VZ_mark",
    )
    scaling_divisors: tuple[float, ...] = (1.0, 10.0, 100.0, 100.0, 100.0)
    accumulation_dtype: str = "float64"
    output_dtype: str = "float32"
    source_feature_order: tuple[str, ...] = RAW7_FEATURE_NAMES

    def __post_init__(self) -> None:
        if self.box_size_mpc_h <= 0 or self.resolution <= 0:
            raise ValueError("box_size_mpc_h and resolution must be positive.")
        if self.deposition != "periodic_cic" or self.grid_origin != "cell_centered":
            raise ValueError("Experiment 1 requires cell-centered periodic CIC.")
        if len(self.channel_order) != 5 or len(self.scaling_divisors) != 5:
            raise ValueError("Experiment 1 requires exactly five channels and divisors.")
        if self.source_feature_order != RAW7_FEATURE_NAMES:
            raise ValueError("Experiment 1 requires the frozen raw7 feature order.")
        if self.accumulation_dtype != "float64" or self.output_dtype != "float32":
            raise ValueError("Experiment 1 requires float64 accumulation and float32 output.")


EXPERIMENT_1_SPEC = HaloVoxelizationSpec()


def representation_fingerprint(spec: HaloVoxelizationSpec = EXPERIMENT_1_SPEC) -> str:
    """Hash every field defining the scientifically meaningful representation."""
    payload = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_float64(values: torch.Tensor | Sequence[Sequence[float]]) -> torch.Tensor:
    result = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    if not torch.isfinite(result).all():
        raise ValueError("Halo features must be finite.")
    return result


def wrap_periodic_coordinates(
    coordinates: torch.Tensor | Sequence[Sequence[float]], box_size: float = 25.0
) -> torch.Tensor:
    """Wrap Cartesian coordinates to [0, box_size) in float64."""
    values = _as_float64(coordinates)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Expected coordinates [N, 3], got {tuple(values.shape)}")
    if box_size <= 0:
        raise ValueError("box_size must be positive.")
    return torch.remainder(values, box_size)


def periodic_cic_indices_and_weights(
    coordinates: torch.Tensor | Sequence[Sequence[float]],
    spec: HaloVoxelizationSpec = EXPERIMENT_1_SPEC,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [N,8,3] wrapped cell indices and partition-of-unity CIC weights."""
    wrapped = wrap_periodic_coordinates(coordinates, spec.box_size_mpc_h)
    delta = spec.box_size_mpc_h / spec.resolution
    u = wrapped / delta - 0.5
    base = torch.floor(u).to(torch.int64)
    fraction = u - base.to(torch.float64)
    offsets = torch.tensor(list(product((0, 1), repeat=3)), dtype=torch.int64)
    indices = torch.remainder(base[:, None, :] + offsets[None, :, :], spec.resolution)
    factors = torch.where(offsets[None, :, :].bool(), fraction[:, None, :], 1.0 - fraction[:, None, :])
    weights = factors.prod(dim=-1)
    return indices, weights


def _validate_features(
    features: torch.Tensor | Sequence[Sequence[float]], feature_names: Sequence[str] | None
) -> torch.Tensor:
    if feature_names is not None and tuple(feature_names) != RAW7_FEATURE_NAMES:
        raise ValueError(f"Expected feature order {RAW7_FEATURE_NAMES}, got {tuple(feature_names)}")
    result = _as_float64(features)
    if result.ndim != 2 or result.shape[1] != 7:
        raise ValueError(f"Expected halo features [N, 7], got {tuple(result.shape)}")
    return result


def voxelize_halo_features(
    features: torch.Tensor | Sequence[Sequence[float]],
    valid_mask: torch.Tensor | Sequence[bool] | None = None,
    *,
    feature_names: Sequence[str] | None = RAW7_FEATURE_NAMES,
    spec: HaloVoxelizationSpec = EXPERIMENT_1_SPEC,
) -> torch.Tensor:
    """Deposit frozen raw7 halo features into scaled [5,R,R,R] float32 CIC fields."""
    values = _validate_features(features, feature_names)
    if valid_mask is not None:
        mask = torch.as_tensor(valid_mask, dtype=torch.bool, device="cpu").reshape(-1)
        if mask.numel() != values.shape[0]:
            raise ValueError("valid_mask length must match halo features.")
        values = values[mask]
    if values.shape[0] == 0:
        raise ValueError("At least one valid halo is required.")
    indices, weights = periodic_cic_indices_and_weights(values[:, 1:4], spec)
    linear = ((indices[..., 0] * spec.resolution) + indices[..., 1]) * spec.resolution + indices[..., 2]
    marks = torch.stack((torch.ones(values.shape[0], dtype=torch.float64), values[:, 0],
                         values[:, 4], values[:, 5], values[:, 6]), dim=1)
    field = torch.zeros((5, spec.resolution ** 3), dtype=torch.float64)
    for channel in range(5):
        field[channel].scatter_add_(0, linear.reshape(-1), (weights * marks[:, channel:channel + 1]).reshape(-1))
    field = field.reshape(5, spec.resolution, spec.resolution, spec.resolution)
    divisors = torch.tensor(spec.scaling_divisors, dtype=torch.float64).view(5, 1, 1, 1)
    return (field / divisors).to(dtype=torch.float32)
