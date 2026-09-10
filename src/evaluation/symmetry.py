"""Pure symmetry transforms for CAMELS raw7 periodic-box graph features."""

from __future__ import annotations

import itertools
from collections import OrderedDict
from typing import Mapping, Sequence

import numpy as np
import torch


FEATURE_NAMES = ("log10_Mvir", "X", "Y", "Z", "VX", "VY", "VZ")
POSITION_SLICE = slice(1, 4)
VELOCITY_SLICE = slice(4, 7)
DEFAULT_BOX_SIZE = 25.0
TRANSLATION_FRACTIONS: Mapping[str, tuple[float, float, float]] = OrderedDict(
    (
        ("T1", (0.5, 0.0, 0.0)),
        ("T2", (0.0, 0.5, 0.0)),
        ("T3", (0.0, 0.0, 0.5)),
        ("T4", (0.25, 0.50, 0.75)),
    )
)


def _validate_box_size(box_size: float) -> float:
    value = float(box_size)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("box_size must be finite and positive")
    return value


def _validate_xyz(xyz: torch.Tensor) -> None:
    if not torch.is_tensor(xyz) or xyz.ndim < 1 or xyz.shape[-1] != 3:
        shape = getattr(xyz, "shape", None)
        raise ValueError(f"expected a tensor ending in dimension 3, got {shape}")
    if not xyz.is_floating_point():
        raise TypeError("coordinate and velocity tensors must have floating dtype")


def translation_vectors(box_size: float = DEFAULT_BOX_SIZE) -> OrderedDict[str, tuple[float, float, float]]:
    """Return the four frozen translations in physical box units."""
    box = _validate_box_size(box_size)
    return OrderedDict((key, tuple(box * value for value in fractions))
                       for key, fractions in TRANSLATION_FRACTIONS.items())


def validate_proper_cube_rotation(rotation: torch.Tensor | np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
    """Validate and return a proper 3-D signed-permutation matrix."""
    if torch.is_tensor(rotation):
        array = rotation.detach().cpu().numpy()
    else:
        array = np.asarray(rotation)
    if array.shape != (3, 3):
        raise ValueError(f"rotation must have shape [3,3], got {array.shape}")
    if not np.all(np.isin(array, (-1, 0, 1))):
        raise ValueError("rotation entries must be in {-1,0,1}")
    if not np.all(np.count_nonzero(array, axis=0) == 1) or not np.all(
        np.count_nonzero(array, axis=1) == 1
    ):
        raise ValueError("rotation must have exactly one nonzero per row and column")
    if not np.array_equal(array.T @ array, np.eye(3, dtype=array.dtype)):
        raise ValueError("rotation must be orthogonal")
    if int(round(float(np.linalg.det(array)))) != 1:
        raise ValueError("rotation determinant must be +1")
    return array.astype(np.int8, copy=True)


def proper_cube_rotations() -> OrderedDict[str, np.ndarray]:
    """Generate all 24 rotations in deterministic permutation/sign order."""
    matrices: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.int8)
            matrix[np.arange(3), permutation] = signs
            if int(round(float(np.linalg.det(matrix)))) == 1:
                matrices.append(validate_proper_cube_rotation(matrix))
    if len(matrices) != 24 or len({matrix.tobytes() for matrix in matrices}) != 24:
        raise RuntimeError("proper cube rotation generation did not produce 24 unique matrices")
    return OrderedDict((f"R{index:02d}", matrix) for index, matrix in enumerate(matrices))


def identity_rotation_id() -> str:
    identity = np.eye(3, dtype=np.int8)
    matches = [key for key, matrix in proper_cube_rotations().items()
               if np.array_equal(matrix, identity)]
    if len(matches) != 1:
        raise RuntimeError("identity rotation must occur exactly once")
    return matches[0]


def apply_periodic_translation(
    positions: torch.Tensor,
    delta: Sequence[float] | torch.Tensor,
    box_size: float = DEFAULT_BOX_SIZE,
) -> torch.Tensor:
    """Translate row-vector positions and wrap them into ``[0, box_size)``."""
    _validate_xyz(positions)
    box = _validate_box_size(box_size)
    shift = torch.as_tensor(delta, dtype=positions.dtype, device=positions.device)
    if shift.shape != (3,) or not bool(torch.isfinite(shift).all()):
        raise ValueError("delta must contain exactly three finite values")
    return torch.remainder(positions + shift, box)


def apply_cube_rotation(
    vectors: torch.Tensor,
    rotation: torch.Tensor | np.ndarray | Sequence[Sequence[int]],
    *,
    box_size: float | None = None,
) -> torch.Tensor:
    """Apply ``vectors @ R.T``; modulo-wrap only when ``box_size`` is supplied."""
    _validate_xyz(vectors)
    matrix = validate_proper_cube_rotation(rotation)
    matrix_tensor = torch.as_tensor(matrix, dtype=vectors.dtype, device=vectors.device)
    transformed = vectors @ matrix_tensor.T
    if box_size is not None:
        transformed = torch.remainder(transformed, _validate_box_size(box_size))
    return transformed


def transform_raw7_features(
    features: torch.Tensor,
    *,
    translation: Sequence[float] | torch.Tensor | None = None,
    rotation: torch.Tensor | np.ndarray | Sequence[Sequence[int]] | None = None,
    box_size: float = DEFAULT_BOX_SIZE,
) -> torch.Tensor:
    """Return a transformed clone in the frozen raw7 feature convention."""
    if not torch.is_tensor(features) or features.ndim < 2 or features.shape[-1] != 7:
        shape = getattr(features, "shape", None)
        raise ValueError(f"features must end in the seven raw7 features, got {shape}")
    if not features.is_floating_point():
        raise TypeError("features must have floating dtype")
    if (translation is None) == (rotation is None):
        raise ValueError("provide exactly one of translation or rotation")
    result = features.clone()
    if translation is not None:
        result[..., POSITION_SLICE] = apply_periodic_translation(
            features[..., POSITION_SLICE], translation, box_size
        )
    else:
        result[..., POSITION_SLICE] = apply_cube_rotation(
            features[..., POSITION_SLICE], rotation, box_size=box_size
        )
        result[..., VELOCITY_SLICE] = apply_cube_rotation(
            features[..., VELOCITY_SLICE], rotation
        )
    return result


def rotation_metadata() -> list[dict[str, object]]:
    return [
        {"id": key, "matrix": matrix.astype(int).tolist()}
        for key, matrix in proper_cube_rotations().items()
    ]
