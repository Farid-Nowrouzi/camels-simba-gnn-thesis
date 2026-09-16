"""Canonical periodic geometry for CAMELS boxes."""
from __future__ import annotations

import math
import torch


def signed_minimum_image_displacement(
    source_pos: torch.Tensor, target_pos: torch.Tensor, box_size: float
) -> torch.Tensor:
    """Return source-to-target minimum-image displacement.

    ``source_pos`` is node j and ``target_pos`` node i, so the result is
    ``x_i - x_j - L * round((x_i - x_j) / L)``.  PyTorch ``round`` uses
    round-half-to-even at exact half-box ties.  Inputs are not mutated.
    """
    if not isinstance(source_pos, torch.Tensor) or not isinstance(target_pos, torch.Tensor):
        raise TypeError("source_pos and target_pos must be torch tensors")
    if source_pos.shape != target_pos.shape or source_pos.ndim < 1 or source_pos.shape[-1] != 3:
        raise ValueError("positions must have equal shape [..., 3]")
    if not source_pos.is_floating_point() or not target_pos.is_floating_point():
        raise TypeError("positions must use floating dtype")
    if not math.isfinite(float(box_size)) or float(box_size) <= 0:
        raise ValueError("box_size must be positive and finite")
    delta = target_pos - source_pos
    return delta - float(box_size) * torch.round(delta / float(box_size))
