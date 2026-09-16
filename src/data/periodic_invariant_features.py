"""Physical periodic-invariant features for the controlled GNN experiment."""
from __future__ import annotations

import torch

from src.data.periodic_geometry import signed_minimum_image_displacement

BOX_SIZE = 25.0
VELOCITY_SCALE = 1000.0  # km/s; Rockstar physical peculiar velocity contract.
EPSILON = 1e-8


def periodic_invariant_features(
    x: torch.Tensor, edge_index: torch.Tensor, *, box_size: float = BOX_SIZE,
    velocity_scale: float = VELOCITY_SCALE, epsilon: float = EPSILON,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw7 ``[N,7]`` plus source->target edges to ``[N,1]``, ``[E,3]``.

    Raw7 is ``[log10(Mvir), X, Y, Z, VX, VY, VZ]``.  Edge columns are,
    in order, distance/L, relative speed/V0, and outward radial relative
    velocity/V0.  No fitted statistics or split-dependent state is used.
    """
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != 7:
        raise ValueError("x must have shape [N, 7]")
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("x must be finite floating-point values")
    if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
        raise TypeError("edge_index must be integer")
    if velocity_scale <= 0 or epsilon <= 0:
        raise ValueError("velocity_scale and epsilon must be positive")
    edge_index = edge_index.long()
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= x.shape[0]):
        raise ValueError("edge_index contains out-of-range nodes")
    source, target = edge_index
    if torch.any(source == target):
        raise ValueError("physical invariant edges must not contain self-loops")
    displacement = signed_minimum_image_displacement(x[source, 1:4], x[target, 1:4], box_size)
    delta_v = x[target, 4:7] - x[source, 4:7]
    radius = torch.linalg.vector_norm(displacement, dim=-1)
    relative_speed = torch.linalg.vector_norm(delta_v, dim=-1)
    radial_velocity = (delta_v * displacement).sum(dim=-1) / radius.clamp_min(epsilon)
    edge_attr = torch.stack((radius / box_size, relative_speed / velocity_scale,
                             radial_velocity / velocity_scale), dim=-1)
    if not torch.isfinite(edge_attr).all():
        raise ValueError("invariant edge features must be finite")
    return x[:, :1], edge_attr
