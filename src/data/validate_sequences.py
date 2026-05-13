from __future__ import annotations

"""
validate_sequences.py

Validation utility for saved CAMELS-SIMBA temporal graph-sequence datasets.

Purpose
-------
This script validates temporal graph datasets before training temporal GNN models.

Expected temporal dataset format:

dataset = {
    "LH_0": {
        "A_list": [Tensor(num_nodes, num_nodes), ...],
        "Nodes_list": [Tensor(num_nodes, num_features), ...],
        "mask_list": [Tensor(num_nodes, 1), ...],
        "target": Tensor scalar,
        "snapshots": [
            {...metadata for snapshot 1...},
            {...metadata for snapshot 2...},
            ...
        ],
        ...
    },
    "LH_1": {...},
    ...
}

Official preprocessing
----------------------
    v2_logmass_minmax_top100_periodic_knn

Scientific definition
---------------------
    Universe       = one simulation sample
    Snapshot       = one graph at one cosmic time
    Nodes          = halos
    Node selection = top num_nodes halos by raw Mvir
    Node features  = [log10_Mvir, X, Y, Z, VX, VY, VZ]
    Edges          = periodic boundary-aware k-nearest-neighbor graph
    Target         = Omega_m
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from src.data.camels_graph_utils import (
    DEFAULT_BOX_SIZE,
    FEATURE_NAMES,
    PREPROCESSING_VERSION,
)


# ============================================================
# General helpers
# ============================================================

def safe_float(value: Any):
    """
    Convert tensor/scalar/list values into a printable float when possible.
    """
    try:
        if torch.is_tensor(value):
            return float(value.detach().cpu().view(-1)[0].item())
        return float(value)
    except Exception:
        return None


def safe_tensor(value: Any):
    """
    Return value if it is a tensor, otherwise None.
    """
    if torch.is_tensor(value):
        return value.detach().cpu()
    return None


def sort_universe_ids(keys: List[Any]) -> List[Any]:
    """
    Sort universe IDs like LH_0, LH_1, ..., LH_100.
    """
    def key_fn(x: Any):
        text = str(x)

        if "_" in text:
            try:
                return int(text.split("_")[-1])
            except ValueError:
                return text

        try:
            return int(text)
        except ValueError:
            return text

    return sorted(keys, key=key_fn)


def load_dataset(path: Path) -> Dict[str, Any]:
    """
    Load temporal graph dataset safely.

    In PyTorch 2.6+, torch.load may default to weights_only=True.
    Our dataset contains dictionaries and tensors, so weights_only=False is
    explicitly used because this is our own trusted processed file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        dataset = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        dataset = torch.load(path, map_location="cpu")

    if not isinstance(dataset, dict):
        raise TypeError(f"Expected dataset to be dict, got {type(dataset)}")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    return dataset


# ============================================================
# Tensor and graph diagnostics
# ============================================================

def tensor_summary(name: str, value: Any) -> Dict[str, Any]:
    """
    Return basic numeric checks for one tensor.
    """
    tensor = safe_tensor(value)

    if tensor is None:
        return {
            "name": name,
            "valid_tensor": False,
            "type": str(type(value)),
        }

    summary: Dict[str, Any] = {
        "name": name,
        "valid_tensor": True,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "nan_count": (
            int(torch.isnan(tensor.float()).sum().item())
            if tensor.is_floating_point()
            else 0
        ),
        "inf_count": (
            int(torch.isinf(tensor.float()).sum().item())
            if tensor.is_floating_point()
            else 0
        ),
    }

    if tensor.numel() > 0 and tensor.is_floating_point():
        summary.update(
            {
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
                "mean": float(tensor.mean().item()),
                "std": (
                    float(tensor.std().item())
                    if tensor.numel() > 1
                    else 0.0
                ),
            }
        )

    return summary


def adjacency_stats(A: Any) -> Dict[str, Any]:
    """
    Compute graph-level diagnostics for an adjacency matrix.
    """
    A_cpu = safe_tensor(A)

    if A_cpu is None:
        return {
            "valid": False,
            "reason": f"A is not a tensor. Type={type(A)}",
        }

    if A_cpu.ndim != 2:
        return {
            "valid": False,
            "reason": f"A is not 2D. Shape={tuple(A_cpu.shape)}",
        }

    if A_cpu.shape[0] != A_cpu.shape[1]:
        return {
            "valid": False,
            "reason": f"A is not square. Shape={tuple(A_cpu.shape)}",
        }

    A_bool = A_cpu > 0
    n = A_cpu.shape[0]

    nonzero = int(A_bool.sum().item())
    diag_nonzero = int((torch.diag(A_cpu) > 0).sum().item())

    degree = A_bool.sum(dim=1).float()

    symmetry_error = float(torch.abs(A_cpu.float() - A_cpu.float().T).sum().item())
    is_symmetric = symmetry_error == 0.0

    estimated_undirected_edges = nonzero // 2 if is_symmetric else None

    return {
        "valid": True,
        "num_nodes": int(n),
        "nonzero_entries": nonzero,
        "estimated_undirected_edges_if_symmetric": estimated_undirected_edges,
        "diag_nonzero": diag_nonzero,
        "is_symmetric": is_symmetric,
        "symmetry_error": symmetry_error,
        "isolated_nodes": int((degree == 0).sum().item()),
        "degree_min": float(degree.min().item()) if degree.numel() > 0 else None,
        "degree_max": float(degree.max().item()) if degree.numel() > 0 else None,
        "degree_mean": float(degree.mean().item()) if degree.numel() > 0 else None,
        "degree_std": float(degree.std().item()) if degree.numel() > 1 else 0.0,
    }


# ============================================================
# Metadata validation
# ============================================================

def validate_top_level_metadata(
    universe_id: str,
    sample: Dict[str, Any],
    expected_normalization: str,
    expected_graph_mode: str,
    expected_periodic_boundary: bool,
    expected_box_size: float,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Validate thesis/science metadata stored at universe level.
    """
    errors: List[str] = []
    warnings: List[str] = []

    metadata_report = {
        "preprocessing_version": sample.get("preprocessing_version"),
        "feature_names": sample.get("feature_names"),
        "mass_feature": sample.get("mass_feature"),
        "node_selection": sample.get("node_selection"),
        "graph_positions": sample.get("graph_positions"),
        "normalization": sample.get("normalization"),
        "graph_mode": sample.get("graph_mode"),
        "periodic_boundary": sample.get("periodic_boundary"),
        "periodic_boundary_knn": sample.get("periodic_boundary_knn"),
        "box_size": sample.get("box_size"),
    }

    if sample.get("preprocessing_version") != PREPROCESSING_VERSION:
        errors.append(
            f"{universe_id}: preprocessing_version={sample.get('preprocessing_version')} "
            f"!= expected {PREPROCESSING_VERSION}"
        )

    if sample.get("feature_names") != FEATURE_NAMES:
        errors.append(
            f"{universe_id}: feature_names={sample.get('feature_names')} "
            f"!= expected {FEATURE_NAMES}"
        )

    if sample.get("mass_feature") != "log10_Mvir":
        errors.append(
            f"{universe_id}: mass_feature={sample.get('mass_feature')} "
            "!= expected log10_Mvir"
        )

    if sample.get("node_selection") != "top_num_nodes_by_raw_Mvir_descending":
        errors.append(
            f"{universe_id}: unexpected node_selection={sample.get('node_selection')}"
        )

    if sample.get("graph_positions") != "raw_physical_XYZ_before_feature_normalization":
        errors.append(
            f"{universe_id}: unexpected graph_positions={sample.get('graph_positions')}"
        )

    if sample.get("normalization") != expected_normalization:
        errors.append(
            f"{universe_id}: normalization={sample.get('normalization')} "
            f"!= expected {expected_normalization}"
        )

    if sample.get("graph_mode") != expected_graph_mode:
        errors.append(
            f"{universe_id}: graph_mode={sample.get('graph_mode')} "
            f"!= expected {expected_graph_mode}"
        )

    if sample.get("periodic_boundary") != expected_periodic_boundary:
        errors.append(
            f"{universe_id}: periodic_boundary={sample.get('periodic_boundary')} "
            f"!= expected {expected_periodic_boundary}"
        )

    if expected_graph_mode == "knn":
        expected_periodic_knn = bool(expected_periodic_boundary)

        if sample.get("periodic_boundary_knn") != expected_periodic_knn:
            errors.append(
                f"{universe_id}: periodic_boundary_knn="
                f"{sample.get('periodic_boundary_knn')} "
                f"!= expected {expected_periodic_knn}"
            )

    box_value = sample.get("box_size")
    if box_value is None:
        errors.append(f"{universe_id}: missing box_size metadata")
    else:
        if abs(float(box_value) - float(expected_box_size)) > 1e-6:
            errors.append(
                f"{universe_id}: box_size={box_value} "
                f"!= expected {expected_box_size}"
            )

    return errors, warnings, metadata_report


def validate_snapshot_metadata(
    universe_id: str,
    snapshot_index: int,
    snapshot_meta: Any,
    expected_graph_mode: str,
    expected_periodic_boundary: bool,
    expected_box_size: float,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Validate metadata for one snapshot.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(snapshot_meta, dict):
        return (
            [f"{universe_id} snapshot {snapshot_index}: snapshot metadata is not a dict"],
            warnings,
            {"valid_metadata": False, "type": str(type(snapshot_meta))},
        )

    metadata_report = {
        "valid_metadata": True,
        "snapshot_value": snapshot_meta.get("snapshot_value"),
        "path": snapshot_meta.get("path"),
        "preprocessing_version": snapshot_meta.get("preprocessing_version"),
        "feature_names": snapshot_meta.get("feature_names"),
        "mass_feature": snapshot_meta.get("mass_feature"),
        "node_selection": snapshot_meta.get("node_selection"),
        "normalization": snapshot_meta.get("normalization"),
        "graph_mode": snapshot_meta.get("graph_mode"),
        "graph_positions": snapshot_meta.get("graph_positions"),
        "periodic_boundary": snapshot_meta.get("periodic_boundary"),
        "periodic_boundary_knn": snapshot_meta.get("periodic_boundary_knn"),
        "box_size": snapshot_meta.get("box_size"),
        "original_num_halos": snapshot_meta.get("original_num_halos"),
        "valid_num_halos": snapshot_meta.get("valid_num_halos"),
        "selected_num_halos_before_padding": snapshot_meta.get(
            "selected_num_halos_before_padding"
        ),
    }

    if snapshot_meta.get("snapshot_value") is None:
        errors.append(
            f"{universe_id} snapshot {snapshot_index}: missing snapshot_value"
        )

    if snapshot_meta.get("preprocessing_version") != PREPROCESSING_VERSION:
        errors.append(
            f"{universe_id} snapshot {snapshot_index}: preprocessing_version="
            f"{snapshot_meta.get('preprocessing_version')} != {PREPROCESSING_VERSION}"
        )

    if snapshot_meta.get("feature_names") is not None:
        if snapshot_meta.get("feature_names") != FEATURE_NAMES:
            errors.append(
                f"{universe_id} snapshot {snapshot_index}: feature_names="
                f"{snapshot_meta.get('feature_names')} != {FEATURE_NAMES}"
            )

    if snapshot_meta.get("mass_feature") != "log10_Mvir":
        errors.append(
            f"{universe_id} snapshot {snapshot_index}: mass_feature="
            f"{snapshot_meta.get('mass_feature')} != log10_Mvir"
        )

    if snapshot_meta.get("node_selection") != "top_num_nodes_by_raw_Mvir_descending":
        errors.append(
            f"{universe_id} snapshot {snapshot_index}: unexpected node_selection="
            f"{snapshot_meta.get('node_selection')}"
        )

    if snapshot_meta.get("graph_positions") != "raw_physical_XYZ_before_feature_normalization":
        errors.append(
            f"{universe_id} snapshot {snapshot_index}: unexpected graph_positions="
            f"{snapshot_meta.get('graph_positions')}"
        )

    if snapshot_meta.get("graph_mode") != expected_graph_mode:
        errors.append(
            f"{universe_id} snapshot {snapshot_index}: graph_mode="
            f"{snapshot_meta.get('graph_mode')} != {expected_graph_mode}"
        )

    if snapshot_meta.get("periodic_boundary") != expected_periodic_boundary:
        errors.append(
            f"{universe_id} snapshot {snapshot_index}: periodic_boundary="
            f"{snapshot_meta.get('periodic_boundary')} != {expected_periodic_boundary}"
        )

    if expected_graph_mode == "knn":
        expected_periodic_knn = bool(expected_periodic_boundary)

        if snapshot_meta.get("periodic_boundary_knn") != expected_periodic_knn:
            errors.append(
                f"{universe_id} snapshot {snapshot_index}: periodic_boundary_knn="
                f"{snapshot_meta.get('periodic_boundary_knn')} "
                f"!= {expected_periodic_knn}"
            )

    box_value = snapshot_meta.get("box_size")
    if box_value is None:
        errors.append(
            f"{universe_id} snapshot {snapshot_index}: missing box_size"
        )
    else:
        if abs(float(box_value) - float(expected_box_size)) > 1e-6:
            errors.append(
                f"{universe_id} snapshot {snapshot_index}: box_size="
                f"{box_value} != {expected_box_size}"
            )

    return errors, warnings, metadata_report


# ============================================================
# Universe-level validation
# ============================================================

def validate_one_snapshot(
    universe_id: str,
    snapshot_index: int,
    A: Any,
    X: Any,
    mask: Any,
    snapshot_meta: Any,
    expected_nodes: int,
    expected_features: int,
    normalization: str,
    expected_graph_mode: str,
    expected_periodic_boundary: bool,
    expected_box_size: float,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Validate one snapshot graph inside one universe sequence.
    """
    errors: List[str] = []
    warnings: List[str] = []

    snapshot_report: Dict[str, Any] = {
        "snapshot_index": snapshot_index,
        "A": tensor_summary("A", A),
        "X": tensor_summary("X", X),
        "mask": tensor_summary("mask", mask),
        "graph": adjacency_stats(A),
    }

    metadata_errors, metadata_warnings, metadata_report = validate_snapshot_metadata(
        universe_id=universe_id,
        snapshot_index=snapshot_index,
        snapshot_meta=snapshot_meta,
        expected_graph_mode=expected_graph_mode,
        expected_periodic_boundary=expected_periodic_boundary,
        expected_box_size=expected_box_size,
    )

    errors.extend(metadata_errors)
    warnings.extend(metadata_warnings)
    snapshot_report["metadata"] = metadata_report

    A_tensor = safe_tensor(A)
    X_tensor = safe_tensor(X)
    mask_tensor = safe_tensor(mask)

    if A_tensor is None:
        errors.append(f"{universe_id} snapshot {snapshot_index}: A is not a tensor")
        return errors, warnings, snapshot_report

    if X_tensor is None:
        errors.append(f"{universe_id} snapshot {snapshot_index}: X is not a tensor")
        return errors, warnings, snapshot_report

    if mask_tensor is None:
        errors.append(f"{universe_id} snapshot {snapshot_index}: mask is not a tensor")
        return errors, warnings, snapshot_report

    if tuple(A_tensor.shape) != (expected_nodes, expected_nodes):
        errors.append(
            f"{universe_id} snapshot {snapshot_index}: A shape {tuple(A_tensor.shape)} "
            f"!= ({expected_nodes}, {expected_nodes})"
        )

    if tuple(X_tensor.shape) != (expected_nodes, expected_features):
        errors.append(
            f"{universe_id} snapshot {snapshot_index}: X shape {tuple(X_tensor.shape)} "
            f"!= ({expected_nodes}, {expected_features})"
        )

    if tuple(mask_tensor.shape) != (expected_nodes, 1):
        errors.append(
            f"{universe_id} snapshot {snapshot_index}: mask shape {tuple(mask_tensor.shape)} "
            f"!= ({expected_nodes}, 1)"
        )

    if A_tensor.is_floating_point():
        if torch.isnan(A_tensor).any():
            errors.append(f"{universe_id} snapshot {snapshot_index}: A contains NaN values")
        if torch.isinf(A_tensor).any():
            errors.append(f"{universe_id} snapshot {snapshot_index}: A contains Inf values")

    if X_tensor.is_floating_point():
        if torch.isnan(X_tensor).any():
            errors.append(f"{universe_id} snapshot {snapshot_index}: X contains NaN values")
        if torch.isinf(X_tensor).any():
            errors.append(f"{universe_id} snapshot {snapshot_index}: X contains Inf values")

    if mask_tensor.is_floating_point():
        if torch.isnan(mask_tensor).any():
            errors.append(f"{universe_id} snapshot {snapshot_index}: mask contains NaN values")
        if torch.isinf(mask_tensor).any():
            errors.append(f"{universe_id} snapshot {snapshot_index}: mask contains Inf values")

    graph_info = snapshot_report["graph"]

    if graph_info.get("valid"):
        if graph_info["nonzero_entries"] == 0:
            errors.append(f"{universe_id} snapshot {snapshot_index}: graph has no edges")

        if graph_info["isolated_nodes"] > 0:
            warnings.append(
                f"{universe_id} snapshot {snapshot_index}: graph has "
                f"{graph_info['isolated_nodes']} isolated nodes"
            )

        if graph_info["is_symmetric"] is False:
            errors.append(
                f"{universe_id} snapshot {snapshot_index}: adjacency is not symmetric"
            )

        if graph_info["diag_nonzero"] != 0:
            errors.append(
                f"{universe_id} snapshot {snapshot_index}: adjacency has "
                f"{graph_info['diag_nonzero']} self-loops"
            )
    else:
        errors.append(
            f"{universe_id} snapshot {snapshot_index}: invalid adjacency: "
            f"{graph_info.get('reason')}"
        )

    if normalization == "minmax":
        x_min = float(X_tensor.min().item())
        x_max = float(X_tensor.max().item())

        tolerance = 1e-5
        if x_min < -tolerance or x_max > 1.0 + tolerance:
            errors.append(
                f"{universe_id} snapshot {snapshot_index}: normalization=minmax "
                f"but X range is [{x_min}, {x_max}], expected approximately [0, 1]"
            )

    real_nodes = int(mask_tensor.sum().item())
    snapshot_report["real_nodes"] = real_nodes

    if real_nodes <= 0:
        errors.append(
            f"{universe_id} snapshot {snapshot_index}: mask indicates zero real nodes"
        )

    mask_unique_values = sorted(
        [float(v) for v in torch.unique(mask_tensor.float()).tolist()]
    )
    snapshot_report["mask_unique_values"] = mask_unique_values

    valid_mask_values = set(mask_unique_values).issubset({0.0, 1.0})
    if not valid_mask_values:
        errors.append(
            f"{universe_id} snapshot {snapshot_index}: mask has values other than 0/1: "
            f"{mask_unique_values}"
        )

    return errors, warnings, snapshot_report


def validate_one_universe(
    universe_id: str,
    sample: Dict[str, Any],
    expected_snapshots: int,
    expected_nodes: int,
    expected_features: int,
    normalization: str,
    expected_periodic_boundary: bool,
    expected_box_size: float,
    expected_graph_mode: str,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Validate one universe temporal graph sequence.
    """
    errors: List[str] = []
    warnings: List[str] = []

    report: Dict[str, Any] = {
        "universe_id": universe_id,
        "errors": errors,
        "warnings": warnings,
    }

    required_keys = [
        "A_list",
        "Nodes_list",
        "mask_list",
        "target",
        "snapshots",
    ]

    for key in required_keys:
        if key not in sample:
            errors.append(f"{universe_id}: missing required key: {key}")

    metadata_errors, metadata_warnings, metadata_report = validate_top_level_metadata(
        universe_id=universe_id,
        sample=sample,
        expected_normalization=normalization,
        expected_graph_mode=expected_graph_mode,
        expected_periodic_boundary=expected_periodic_boundary,
        expected_box_size=expected_box_size,
    )

    errors.extend(metadata_errors)
    warnings.extend(metadata_warnings)
    report["metadata"] = metadata_report

    if any("missing required key" in err for err in errors):
        return errors, warnings, report

    A_list = sample["A_list"]
    X_list = sample["Nodes_list"]
    mask_list = sample["mask_list"]
    snapshots = sample["snapshots"]
    target = sample["target"]

    if not isinstance(A_list, list):
        errors.append(f"{universe_id}: A_list is not a list")
        return errors, warnings, report

    if not isinstance(X_list, list):
        errors.append(f"{universe_id}: Nodes_list is not a list")
        return errors, warnings, report

    if not isinstance(mask_list, list):
        errors.append(f"{universe_id}: mask_list is not a list")
        return errors, warnings, report

    if not isinstance(snapshots, list):
        errors.append(f"{universe_id}: snapshots is not a list")
        return errors, warnings, report

    report["num_snapshots_A"] = len(A_list)
    report["num_snapshots_X"] = len(X_list)
    report["num_snapshots_mask"] = len(mask_list)
    report["num_snapshot_metadata"] = len(snapshots)
    report["target"] = safe_float(target)

    if len(A_list) != expected_snapshots:
        errors.append(
            f"{universe_id}: A_list length {len(A_list)} != expected {expected_snapshots}"
        )

    if len(X_list) != expected_snapshots:
        errors.append(
            f"{universe_id}: Nodes_list length {len(X_list)} != expected {expected_snapshots}"
        )

    if len(mask_list) != expected_snapshots:
        errors.append(
            f"{universe_id}: mask_list length {len(mask_list)} != expected {expected_snapshots}"
        )

    if len(snapshots) != expected_snapshots:
        errors.append(
            f"{universe_id}: snapshots metadata length {len(snapshots)} "
            f"!= expected {expected_snapshots}"
        )

    target_value = safe_float(target)
    if target_value is None:
        errors.append(f"{universe_id}: target is not convertible to float")
    else:
        if not (0.0 < target_value < 1.0):
            warnings.append(
                f"{universe_id}: target value looks unusual for Omega_m: {target_value}"
            )

    snapshot_reports = []
    snapshot_values = []

    min_length = min(
        len(A_list),
        len(X_list),
        len(mask_list),
        len(snapshots),
    )

    for t in range(min_length):
        snapshot_errors, snapshot_warnings, snapshot_report = validate_one_snapshot(
            universe_id=universe_id,
            snapshot_index=t,
            A=A_list[t],
            X=X_list[t],
            mask=mask_list[t],
            snapshot_meta=snapshots[t],
            expected_nodes=expected_nodes,
            expected_features=expected_features,
            normalization=normalization,
            expected_graph_mode=expected_graph_mode,
            expected_periodic_boundary=expected_periodic_boundary,
            expected_box_size=expected_box_size,
        )

        errors.extend(snapshot_errors)
        warnings.extend(snapshot_warnings)
        snapshot_reports.append(snapshot_report)

        snapshot_value = snapshot_report.get("metadata", {}).get("snapshot_value")
        if snapshot_value is not None:
            try:
                snapshot_values.append(float(snapshot_value))
            except Exception:
                warnings.append(
                    f"{universe_id} snapshot {t}: snapshot_value is not numeric: "
                    f"{snapshot_value}"
                )

    if len(snapshot_values) == expected_snapshots:
        if snapshot_values != sorted(snapshot_values):
            errors.append(
                f"{universe_id}: snapshot values are not sorted increasingly: "
                f"{snapshot_values}"
            )

        if len(set(snapshot_values)) != len(snapshot_values):
            errors.append(
                f"{universe_id}: duplicate snapshot values found: {snapshot_values}"
            )

    report["snapshot_values"] = snapshot_values
    report["snapshots"] = snapshot_reports

    return errors, warnings, report


# ============================================================
# Dataset-level validation
# ============================================================

def validate_dataset(
    dataset: Dict[str, Any],
    expected_universes: int,
    expected_snapshots: int,
    expected_nodes: int,
    expected_features: int,
    normalization: str,
    expected_periodic_boundary: bool,
    expected_box_size: float,
    expected_graph_mode: str,
) -> Dict[str, Any]:
    """
    Validate the whole saved temporal dataset.
    """
    all_errors: List[str] = []
    all_warnings: List[str] = []
    universe_reports: Dict[str, Any] = {}

    if not isinstance(dataset, dict):
        raise TypeError(f"Expected dataset to be dict, got {type(dataset)}")

    universe_ids = sort_universe_ids(list(dataset.keys()))

    if len(universe_ids) != expected_universes:
        all_errors.append(
            f"Number of universes {len(universe_ids)} != expected {expected_universes}"
        )

    target_values = []

    for universe_id in universe_ids:
        sample = dataset[universe_id]

        if not isinstance(sample, dict):
            all_errors.append(
                f"{universe_id}: expected dict sample, got {type(sample)}"
            )
            continue

        errors, warnings, report = validate_one_universe(
            universe_id=universe_id,
            sample=sample,
            expected_snapshots=expected_snapshots,
            expected_nodes=expected_nodes,
            expected_features=expected_features,
            normalization=normalization,
            expected_periodic_boundary=expected_periodic_boundary,
            expected_box_size=expected_box_size,
            expected_graph_mode=expected_graph_mode,
        )

        all_errors.extend(errors)
        all_warnings.extend(warnings)
        universe_reports[universe_id] = report

        target_value = report.get("target")
        if target_value is not None:
            target_values.append(float(target_value))

    target_summary = {
        "count": len(target_values),
        "min": min(target_values) if target_values else None,
        "max": max(target_values) if target_values else None,
        "mean": (
            sum(target_values) / len(target_values)
            if target_values
            else None
        ),
    }

    return {
        "passed": len(all_errors) == 0,
        "dataset_type": "temporal_graph_sequences",
        "preprocessing_version_expected": PREPROCESSING_VERSION,
        "feature_names_expected": FEATURE_NAMES,
        "expected_graph_mode": expected_graph_mode,
        "expected_periodic_boundary": expected_periodic_boundary,
        "expected_box_size": expected_box_size,
        "num_universes_found": len(universe_ids),
        "expected_universes": expected_universes,
        "expected_snapshots": expected_snapshots,
        "expected_nodes": expected_nodes,
        "expected_features": expected_features,
        "normalization": normalization,
        "target_summary": target_summary,
        "errors": all_errors,
        "warnings": all_warnings,
        "universes": universe_reports,
    }


# ============================================================
# Report printing and saving
# ============================================================

def print_report(report: Dict[str, Any], max_show: int = 5) -> None:
    """
    Print a readable validation summary.
    """
    print("=" * 90)
    print("CAMELS-SIMBA TEMPORAL SEQUENCE VALIDATION")
    print("=" * 90)

    print(f"Passed:                     {report['passed']}")
    print(f"Dataset type:               {report['dataset_type']}")
    print(f"Expected preprocessing:     {report['preprocessing_version_expected']}")
    print(f"Expected features:          {report['feature_names_expected']}")
    print(f"Expected graph mode:        {report['expected_graph_mode']}")
    print(f"Expected periodic boundary: {report['expected_periodic_boundary']}")
    print(f"Expected box size:          {report['expected_box_size']}")
    print(f"Universes found:            {report['num_universes_found']}")
    print(f"Expected universes:         {report['expected_universes']}")
    print(f"Expected snapshots:         {report['expected_snapshots']}")
    print(f"Expected nodes:             {report['expected_nodes']}")
    print(f"Expected features count:    {report['expected_features']}")
    print(f"Normalization:              {report['normalization']}")

    print("\nTarget summary:")
    print("-" * 90)
    target_summary = report.get("target_summary", {})
    print(f"Count:                      {target_summary.get('count')}")
    print(f"Min Omega_m:                {target_summary.get('min')}")
    print(f"Max Omega_m:                {target_summary.get('max')}")
    print(f"Mean Omega_m:               {target_summary.get('mean')}")

    print("\nErrors:")
    print("-" * 90)
    if report["errors"]:
        for err in report["errors"][:80]:
            print(f"❌ {err}")
        if len(report["errors"]) > 80:
            print(f"... and {len(report['errors']) - 80} more errors")
    else:
        print("✅ No errors found.")

    print("\nWarnings:")
    print("-" * 90)
    if report["warnings"]:
        for warn in report["warnings"][:80]:
            print(f"⚠️ {warn}")
        if len(report["warnings"]) > 80:
            print(f"... and {len(report['warnings']) - 80} more warnings")
    else:
        print("✅ No warnings found.")

    print("\nFirst universe summaries:")
    print("-" * 90)

    shown = 0
    for universe_id, u_report in report["universes"].items():
        if shown >= max_show:
            break

        print(f"\nUniverse: {universe_id}")
        print(f"  Target Omega_m:        {u_report.get('target')}")
        print(f"  A snapshots:           {u_report.get('num_snapshots_A')}")
        print(f"  X snapshots:           {u_report.get('num_snapshots_X')}")
        print(f"  Mask snapshots:        {u_report.get('num_snapshots_mask')}")
        print(f"  Metadata snapshots:    {u_report.get('num_snapshot_metadata')}")
        print(f"  Snapshot values:       {u_report.get('snapshot_values')}")

        metadata = u_report.get("metadata", {})
        print(f"  Preprocessing:         {metadata.get('preprocessing_version')}")
        print(f"  Mass feature:          {metadata.get('mass_feature')}")
        print(f"  Normalization:         {metadata.get('normalization')}")
        print(f"  Graph mode:            {metadata.get('graph_mode')}")
        print(f"  Periodic boundary:     {metadata.get('periodic_boundary')}")
        print(f"  Periodic kNN:          {metadata.get('periodic_boundary_knn')}")
        print(f"  Box size:              {metadata.get('box_size')}")

        snapshots = u_report.get("snapshots", [])
        if snapshots:
            first = snapshots[0]
            first_graph = first.get("graph", {})
            first_x = first.get("X", {})
            first_a = first.get("A", {})
            first_mask = first.get("mask", {})

            print(f"  First A shape:         {first_a.get('shape')}")
            print(f"  First X shape:         {first_x.get('shape')}")
            print(f"  First mask shape:      {first_mask.get('shape')}")
            print(
                f"  First graph degree:    min={first_graph.get('degree_min')} "
                f"mean={first_graph.get('degree_mean')} "
                f"max={first_graph.get('degree_max')}"
            )
            print(f"  Estimated edges:       {first_graph.get('estimated_undirected_edges_if_symmetric')}")
            print(f"  Real nodes:            {first.get('real_nodes')}")
            print(f"  Mask unique values:    {first.get('mask_unique_values')}")

            if "min" in first_x:
                print(
                    f"  First X range:         min={first_x.get('min')} "
                    f"max={first_x.get('max')}"
                )

        shown += 1

    print("\n" + "=" * 90)
    if report["passed"]:
        print("✅ Validation complete. Temporal dataset is structurally and scientifically valid.")
    else:
        print("❌ Validation complete. Temporal dataset has errors that must be fixed.")
    print("=" * 90)


def save_report(report: Dict[str, Any], output_path: Path) -> None:
    """
    Save validation report as JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate saved CAMELS-SIMBA temporal graph-sequence dataset."
    )

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to saved temporal dataset .pt file.",
    )

    parser.add_argument(
        "--expected_universes",
        type=int,
        required=True,
        help="Expected number of universes in the saved dataset.",
    )

    parser.add_argument(
        "--expected_snapshots",
        type=int,
        default=5,
        help="Expected number of snapshots per universe.",
    )

    parser.add_argument(
        "--expected_nodes",
        type=int,
        default=100,
        help="Expected number of nodes per snapshot graph.",
    )

    parser.add_argument(
        "--expected_features",
        type=int,
        default=7,
        help="Expected number of node features.",
    )

    parser.add_argument(
        "--normalization",
        type=str,
        default="minmax",
        choices=["none", "minmax", "zscore"],
        help="Expected normalization mode.",
    )

    parser.add_argument(
        "--graph_mode",
        type=str,
        default="knn",
        choices=["knn", "radius"],
        help="Expected graph construction mode.",
    )

    parser.add_argument(
        "--periodic_boundary",
        action="store_true",
        default=True,
        help="Expect periodic boundary-aware distances. Enabled by default.",
    )

    parser.add_argument(
        "--no_periodic_boundary",
        action="store_false",
        dest="periodic_boundary",
        help="Expect non-periodic Euclidean graph distances.",
    )

    parser.add_argument(
        "--box_size",
        type=float,
        default=DEFAULT_BOX_SIZE,
        help="Expected CAMELS box size. Default: 25.0 h^-1 Mpc.",
    )

    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional path to save validation report as JSON.",
    )

    parser.add_argument(
        "--max_show",
        type=int,
        default=5,
        help="Maximum number of universe summaries to print.",
    )

    args = parser.parse_args()

    dataset_path = Path(args.path)

    print(f"Loading temporal dataset: {dataset_path}")
    dataset = load_dataset(dataset_path)

    report = validate_dataset(
        dataset=dataset,
        expected_universes=args.expected_universes,
        expected_snapshots=args.expected_snapshots,
        expected_nodes=args.expected_nodes,
        expected_features=args.expected_features,
        normalization=args.normalization,
        expected_periodic_boundary=args.periodic_boundary,
        expected_box_size=args.box_size,
        expected_graph_mode=args.graph_mode,
    )

    print_report(report, max_show=args.max_show)

    if args.save_json:
        save_report(report, Path(args.save_json))
        print(f"\nValidation JSON saved to: {args.save_json}")


if __name__ == "__main__":
    main()