"""Versioned engineered summaries from the final CAMELS halo snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SUMMARY_DEFINITION_VERSION = "u1000_final_summary20_v1"
SNAPSHOT_PROTOCOL_FINAL = "final"
FINAL_SCALE_FACTOR = 1.0
NODE_FEATURE_NAMES = ["log10_Mvir", "X", "Y", "Z", "VX", "VY", "VZ"]
SUMMARY_FEATURE_NAMES = [
    "halo_count",
    "log10_Mvir_mean", "log10_Mvir_std", "log10_Mvir_min", "log10_Mvir_max", "log10_Mvir_median",
    "X_mean", "Y_mean", "Z_mean", "X_std", "Y_std", "Z_std",
    "VX_mean", "VY_mean", "VZ_mean", "VX_std", "VY_std", "VZ_std",
    "speed_mean", "speed_std",
]


def universe_sort_key(universe_id: str) -> tuple[int, str]:
    text = str(universe_id)
    try:
        return (int(text.split("_", 1)[1] if text.lower().startswith("lh_") else text), text)
    except (ValueError, IndexError):
        raise ValueError(f"Unsupported universe ID format: {universe_id!r}") from None


def load_processed_dataset(path: str | Path) -> dict[str, dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {source}")
    try:
        data = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        data = torch.load(source, map_location="cpu")
    if not isinstance(data, dict) or not data:
        raise TypeError("Processed dataset must be a non-empty universe dictionary.")
    return data


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _scale_factor(snapshot: Any) -> float | None:
    if isinstance(snapshot, Mapping):
        for key in ("a", "scale_factor", "snapshot_value", "snapshot", "snapshot_id"):
            if key in snapshot and isinstance(snapshot[key], (int, float)):
                return float(snapshot[key])
    if isinstance(snapshot, (int, float)):
        return float(snapshot)
    return None


def select_snapshot(sample: Mapping[str, Any], snapshot_protocol: str = SNAPSHOT_PROTOCOL_FINAL) -> tuple[np.ndarray, np.ndarray]:
    """Select only a=1.0, matching Static GCN's temporal-final conversion."""
    if snapshot_protocol != SNAPSHOT_PROTOCOL_FINAL:
        raise ValueError(f"Unsupported snapshot protocol: {snapshot_protocol!r}")
    if "Nodes_list" not in sample or "mask_list" not in sample:
        raise KeyError("Final protocol requires Nodes_list and mask_list.")
    nodes, masks = sample["Nodes_list"], sample["mask_list"]
    if len(nodes) == 0 or len(nodes) != len(masks):
        raise ValueError("Nodes_list/mask_list must be non-empty and have equal length.")
    snapshots = sample.get("snapshots")
    if snapshots is None:
        raise ValueError("Final protocol requires snapshot metadata to verify a=1.0.")
    if len(snapshots) != len(nodes):
        raise ValueError("Snapshot metadata length does not match node tensors.")
    scale = _scale_factor(snapshots[-1])
    if scale is None:
        raise ValueError("Final snapshot metadata does not expose a scale factor.")
    if not np.isclose(scale, FINAL_SCALE_FACTOR, atol=1e-8, rtol=0.0):
        raise ValueError(f"Last stored snapshot is not a=1.0: {scale}")
    return _as_numpy(nodes[-1]), _as_numpy(masks[-1])


def summarize_nodes(nodes: Any, mask: Any) -> np.ndarray:
    matrix = _as_numpy(nodes)
    node_mask = _as_numpy(mask)
    if matrix.ndim != 2 or matrix.shape[1] != len(NODE_FEATURE_NAMES):
        raise ValueError(f"Expected node tensor [N, 7], got {matrix.shape}")
    if node_mask.ndim == 2 and node_mask.shape[1] == 1:
        node_mask = node_mask[:, 0]
    if node_mask.ndim != 1 or node_mask.shape[0] != matrix.shape[0]:
        raise ValueError(f"Expected mask [N] or [N,1], got {node_mask.shape}")
    valid = matrix[node_mask > 0]
    if valid.shape[0] == 0:
        raise ValueError("A universe has no valid final-snapshot halos.")
    mass, pos, vel = valid[:, 0], valid[:, 1:4], valid[:, 4:7]
    speed = np.sqrt(np.sum(np.square(vel, dtype=np.float64), axis=1))
    result = np.asarray([
        valid.shape[0],
        mass.mean(), mass.std(ddof=0), mass.min(), mass.max(), np.median(mass),
        *pos.mean(axis=0), *pos.std(axis=0, ddof=0),
        *vel.mean(axis=0), *vel.std(axis=0, ddof=0),
        speed.mean(), speed.std(ddof=0),
    ], dtype=np.float64)
    if result.shape != (20,) or not np.isfinite(result).all():
        raise ValueError("Extracted summary must contain exactly 20 finite values.")
    return result


def extract_summary(sample: Mapping[str, Any], snapshot_protocol: str = SNAPSHOT_PROTOCOL_FINAL) -> np.ndarray:
    nodes, mask = select_snapshot(sample, snapshot_protocol=snapshot_protocol)
    return summarize_nodes(nodes, mask)


def get_target(sample: Mapping[str, Any]) -> float:
    if "target" not in sample:
        raise KeyError("Sample is missing target.")
    value = _as_numpy(sample["target"]).reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]):
        raise ValueError("Target must be one finite scalar.")
    return float(value[0])


def extract_dataset_summaries(
    data: Mapping[str, Mapping[str, Any]], snapshot_protocol: str = SNAPSHOT_PROTOCOL_FINAL
) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids = sorted((str(item) for item in data), key=universe_sort_key)
    features = np.vstack([extract_summary(data[item], snapshot_protocol) for item in ids])
    targets = np.asarray([get_target(data[item]) for item in ids], dtype=np.float64)
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("Summary matrix and targets must be finite.")
    return ids, features, targets


def arrays_for_ids(
    ordered_ids: Sequence[str], all_ids: Sequence[str], features: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Dataset universe IDs are not unique.")
    positions = {universe_id: index for index, universe_id in enumerate(all_ids)}
    missing = [universe_id for universe_id in ordered_ids if universe_id not in positions]
    if missing:
        raise ValueError(f"Manifest IDs absent from dataset: {missing[:20]}")
    indices = [positions[universe_id] for universe_id in ordered_ids]
    return features[indices], targets[indices]
