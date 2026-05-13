from __future__ import annotations

"""
validate_static_graphs.py

Validation utility for saved CAMELS-SIMBA static graph datasets.

Static dataset structure:

dataset = {
    "LH_0": {
        "A": tensor [num_nodes, num_nodes],
        "X": tensor [num_nodes, 7],
        "mask": tensor [num_nodes, 1],
        "target": tensor scalar,
        "snapshot": {...},
        ...
    },
    "LH_1": {...},
    ...
}

This validator is used before training the static graph baseline.

Official preprocessing:
    v2_logmass_minmax_top100_periodic_knn
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


def json_safe(value: Any) -> Any:
    """
    Convert common non-JSON-safe objects into JSON-safe values.
    """
    if torch.is_tensor(value):
        if value.numel() == 1:
            return safe_float(value)
        return value.detach().cpu().tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [json_safe(v) for v in value]

    return value


def load_dataset(path: Path) -> Dict[str, Any]:
    """
    Load static graph dataset safely.

    Uses weights_only=False when supported, with fallback for older PyTorch.
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
        raise ValueError("Static dataset is empty.")

    return dataset


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


# ============================================================
# Tensor and graph summaries
# ============================================================

def tensor_summary(name: str, tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Return basic numeric checks for one tensor.
    """
    if not torch.is_tensor(tensor):
        return {
            "name": name,
            "valid_tensor": False,
            "type": str(type(tensor)),
            "error": "Object is not a torch.Tensor.",
        }

    tensor_cpu = tensor.detach().cpu()

    summary: Dict[str, Any] = {
        "name": name,
        "valid_tensor": True,
        "shape": list(tensor_cpu.shape),
        "dtype": str(tensor_cpu.dtype),
        "device": str(tensor_cpu.device),
        "numel": int(tensor_cpu.numel()),
        "nan_count": (
            int(torch.isnan(tensor_cpu).sum().item())
            if tensor_cpu.is_floating_point()
            else 0
        ),
        "inf_count": (
            int(torch.isinf(tensor_cpu).sum().item())
            if tensor_cpu.is_floating_point()
            else 0
        ),
    }

    if tensor_cpu.numel() > 0 and tensor_cpu.is_floating_point():
        summary.update(
            {
                "min": float(tensor_cpu.min().item()),
                "max": float(tensor_cpu.max().item()),
                "mean": float(tensor_cpu.mean().item()),
                "std": (
                    float(tensor_cpu.std().item())
                    if tensor_cpu.numel() > 1
                    else 0.0
                ),
            }
        )

    return summary


def adjacency_stats(A: torch.Tensor) -> Dict[str, Any]:
    """
    Compute graph-level diagnostics for an adjacency matrix.
    """
    if not torch.is_tensor(A):
        return {
            "valid": False,
            "reason": f"A is not a torch.Tensor. Type={type(A)}",
        }

    A_cpu = A.detach().cpu()

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

    n = A_cpu.shape[0]

    A_binary = A_cpu > 0

    nonzero_entries = int(A_binary.sum().item())
    diag_nonzero = int((torch.diag(A_cpu) > 0).sum().item())

    degree = A_binary.sum(dim=1).float()

    symmetry_error = float(torch.abs(A_cpu - A_cpu.T).sum().item())
    is_symmetric = symmetry_error == 0.0

    estimated_undirected_edges = nonzero_entries // 2 if is_symmetric else None

    return {
        "valid": True,
        "num_nodes": int(n),
        "nonzero_entries": nonzero_entries,
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

def validate_metadata(
    universe_id: str,
    sample: Dict[str, Any],
    expected_normalization: str,
    expected_graph_mode: str,
    expected_preferred_snapshot: float,
    expected_periodic_boundary: bool,
    expected_box_size: float,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Validate static graph metadata.
    """
    errors: List[str] = []
    warnings: List[str] = []

    snapshot_meta = sample.get("snapshot")

    metadata_report = {
        "preprocessing_version": sample.get("preprocessing_version"),
        "feature_names": sample.get("feature_names"),
        "mass_feature": sample.get("mass_feature"),
        "node_selection": sample.get("node_selection"),
        "graph_positions": sample.get("graph_positions"),
        "normalization": sample.get("normalization"),
        "graph_mode": sample.get("graph_mode"),
        "preferred_snapshot": sample.get("preferred_snapshot"),
        "actual_snapshot_value": sample.get("actual_snapshot_value"),
        "exact_preferred_snapshot_match": sample.get("exact_preferred_snapshot_match"),
        "periodic_boundary": sample.get("periodic_boundary"),
        "periodic_boundary_knn": sample.get("periodic_boundary_knn"),
        "box_size": sample.get("box_size"),
        "snapshot_metadata_exists": isinstance(snapshot_meta, dict),
    }

    # --------------------------------------------------------
    # Top-level metadata checks
    # --------------------------------------------------------
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

    preferred_snapshot = sample.get("preferred_snapshot")
    if preferred_snapshot is None:
        errors.append(f"{universe_id}: missing preferred_snapshot")
    else:
        if abs(float(preferred_snapshot) - float(expected_preferred_snapshot)) > 1e-6:
            errors.append(
                f"{universe_id}: preferred_snapshot={preferred_snapshot} "
                f"!= expected {expected_preferred_snapshot}"
            )

    actual_snapshot = sample.get("actual_snapshot_value")
    if actual_snapshot is None:
        errors.append(f"{universe_id}: missing actual_snapshot_value")
    else:
        if abs(float(actual_snapshot) - float(expected_preferred_snapshot)) > 1e-6:
            warnings.append(
                f"{universe_id}: actual_snapshot_value={actual_snapshot}, "
                f"preferred was {expected_preferred_snapshot}. "
                "This is acceptable only if the preferred snapshot was unavailable."
            )

    exact_match = sample.get("exact_preferred_snapshot_match")
    if exact_match is False:
        warnings.append(
            f"{universe_id}: exact_preferred_snapshot_match=False. "
            "Static graph was built from a fallback snapshot."
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
        errors.append(f"{universe_id}: missing box_size")
    else:
        if abs(float(box_value) - float(expected_box_size)) > 1e-6:
            errors.append(
                f"{universe_id}: box_size={box_value} "
                f"!= expected {expected_box_size}"
            )

    # --------------------------------------------------------
    # Nested snapshot metadata checks
    # --------------------------------------------------------
    if isinstance(snapshot_meta, dict):
        if snapshot_meta.get("preprocessing_version") != PREPROCESSING_VERSION:
            errors.append(
                f"{universe_id}: snapshot metadata preprocessing_version="
                f"{snapshot_meta.get('preprocessing_version')} "
                f"!= {PREPROCESSING_VERSION}"
            )

        if snapshot_meta.get("feature_names") != FEATURE_NAMES:
            errors.append(
                f"{universe_id}: snapshot metadata feature_names="
                f"{snapshot_meta.get('feature_names')} "
                f"!= {FEATURE_NAMES}"
            )

        if snapshot_meta.get("mass_feature") != "log10_Mvir":
            errors.append(
                f"{universe_id}: snapshot metadata mass_feature="
                f"{snapshot_meta.get('mass_feature')} "
                "!= log10_Mvir"
            )

        if snapshot_meta.get("periodic_boundary") != expected_periodic_boundary:
            errors.append(
                f"{universe_id}: snapshot metadata periodic_boundary="
                f"{snapshot_meta.get('periodic_boundary')} "
                f"!= {expected_periodic_boundary}"
            )

        if expected_graph_mode == "knn":
            expected_periodic_knn = bool(expected_periodic_boundary)

            if snapshot_meta.get("periodic_boundary_knn") != expected_periodic_knn:
                errors.append(
                    f"{universe_id}: snapshot metadata periodic_boundary_knn="
                    f"{snapshot_meta.get('periodic_boundary_knn')} "
                    f"!= {expected_periodic_knn}"
                )

        snapshot_exact = snapshot_meta.get("exact_preferred_snapshot_match")
        if snapshot_exact is False:
            warnings.append(
                f"{universe_id}: nested snapshot exact_preferred_snapshot_match=False."
            )

    else:
        warnings.append(f"{universe_id}: snapshot metadata missing or not a dict")

    return errors, warnings, metadata_report


# ============================================================
# Per-universe validation
# ============================================================

def validate_one_universe(
    universe_id: str,
    sample: Dict[str, Any],
    expected_nodes: int,
    expected_features: int,
    normalization: str,
    expected_graph_mode: str,
    expected_preferred_snapshot: float,
    expected_periodic_boundary: bool,
    expected_box_size: float,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Validate one static graph sample.
    """
    errors: List[str] = []
    warnings: List[str] = []

    report: Dict[str, Any] = {
        "universe_id": universe_id,
        "errors": errors,
        "warnings": warnings,
    }

    required_keys = ["A", "X", "mask", "target", "snapshot"]

    for key in required_keys:
        if key not in sample:
            errors.append(f"{universe_id}: missing required key: {key}")

    metadata_errors, metadata_warnings, metadata_report = validate_metadata(
        universe_id=universe_id,
        sample=sample,
        expected_normalization=normalization,
        expected_graph_mode=expected_graph_mode,
        expected_preferred_snapshot=expected_preferred_snapshot,
        expected_periodic_boundary=expected_periodic_boundary,
        expected_box_size=expected_box_size,
    )

    errors.extend(metadata_errors)
    warnings.extend(metadata_warnings)
    report["metadata"] = metadata_report

    if any("missing required key" in err for err in errors):
        return errors, report

    A = sample["A"]
    X = sample["X"]
    mask = sample["mask"]
    target = sample["target"]

    # --------------------------------------------------------
    # Tensor type checks
    # --------------------------------------------------------
    if not torch.is_tensor(A):
        errors.append(f"{universe_id}: A is not a torch.Tensor. Got {type(A)}")

    if not torch.is_tensor(X):
        errors.append(f"{universe_id}: X is not a torch.Tensor. Got {type(X)}")

    if not torch.is_tensor(mask):
        errors.append(f"{universe_id}: mask is not a torch.Tensor. Got {type(mask)}")

    if not torch.is_tensor(A) or not torch.is_tensor(X) or not torch.is_tensor(mask):
        return errors, report

    report["target"] = safe_float(target)
    report["A"] = tensor_summary("A", A)
    report["X"] = tensor_summary("X", X)
    report["mask"] = tensor_summary("mask", mask)
    report["graph"] = adjacency_stats(A)

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------
    if tuple(A.shape) != (expected_nodes, expected_nodes):
        errors.append(
            f"{universe_id}: A shape {tuple(A.shape)} != "
            f"({expected_nodes}, {expected_nodes})"
        )

    if tuple(X.shape) != (expected_nodes, expected_features):
        errors.append(
            f"{universe_id}: X shape {tuple(X.shape)} != "
            f"({expected_nodes}, {expected_features})"
        )

    if tuple(mask.shape) != (expected_nodes, 1):
        errors.append(
            f"{universe_id}: mask shape {tuple(mask.shape)} != "
            f"({expected_nodes}, 1)"
        )

    # --------------------------------------------------------
    # Target checks
    # --------------------------------------------------------
    target_value = safe_float(target)

    if target_value is None:
        errors.append(f"{universe_id}: target is not convertible to float")
    else:
        if not (0.0 < target_value < 1.0):
            warnings.append(
                f"{universe_id}: target value looks unusual for Omega_m: {target_value}"
            )

    # --------------------------------------------------------
    # NaN / Inf checks
    # --------------------------------------------------------
    if X.is_floating_point():
        if torch.isnan(X).any():
            errors.append(f"{universe_id}: X contains NaN values")
        if torch.isinf(X).any():
            errors.append(f"{universe_id}: X contains Inf values")

    if A.is_floating_point():
        if torch.isnan(A).any():
            errors.append(f"{universe_id}: A contains NaN values")
        if torch.isinf(A).any():
            errors.append(f"{universe_id}: A contains Inf values")

    if mask.is_floating_point():
        if torch.isnan(mask).any():
            errors.append(f"{universe_id}: mask contains NaN values")
        if torch.isinf(mask).any():
            errors.append(f"{universe_id}: mask contains Inf values")

    # --------------------------------------------------------
    # Adjacency graph checks
    # --------------------------------------------------------
    graph_info = report["graph"]

    if graph_info.get("valid"):
        if graph_info["nonzero_entries"] == 0:
            errors.append(f"{universe_id}: graph has no edges")

        if graph_info["isolated_nodes"] > 0:
            warnings.append(
                f"{universe_id}: graph has {graph_info['isolated_nodes']} isolated nodes"
            )

        if graph_info["is_symmetric"] is False:
            warnings.append(f"{universe_id}: adjacency is not symmetric")

        if graph_info["diag_nonzero"] != 0:
            warnings.append(
                f"{universe_id}: adjacency has {graph_info['diag_nonzero']} self-loops"
            )

        if expected_graph_mode == "knn":
            degree_mean = graph_info.get("degree_mean")
            if degree_mean is not None and degree_mean <= 0:
                errors.append(f"{universe_id}: kNN graph has invalid mean degree")

    else:
        errors.append(f"{universe_id}: invalid adjacency: {graph_info.get('reason')}")

    # --------------------------------------------------------
    # Normalization checks
    # --------------------------------------------------------
    if normalization == "minmax":
        x_min = float(X.min().item())
        x_max = float(X.max().item())

        tolerance = 1e-5

        if x_min < -tolerance or x_max > 1.0 + tolerance:
            errors.append(
                f"{universe_id}: normalization=minmax but X range is "
                f"[{x_min}, {x_max}], expected approximately [0, 1]"
            )

    # --------------------------------------------------------
    # Mask checks
    # --------------------------------------------------------
    real_nodes = int(mask.sum().item())
    report["real_nodes"] = real_nodes

    if real_nodes <= 0:
        errors.append(f"{universe_id}: mask indicates zero real nodes")

    if real_nodes > expected_nodes:
        errors.append(
            f"{universe_id}: mask real nodes {real_nodes} > expected_nodes {expected_nodes}"
        )

    mask_unique = sorted(torch.unique(mask.detach().cpu()).tolist())
    report["mask_unique_values"] = mask_unique

    if not set(mask_unique).issubset({0.0, 1.0}):
        warnings.append(
            f"{universe_id}: mask contains values other than 0/1: {mask_unique}"
        )

    return errors, report


# ============================================================
# Dataset-level validation
# ============================================================

def validate_dataset(
    dataset: Dict[str, Any],
    expected_universes: int,
    expected_nodes: int,
    expected_features: int,
    normalization: str,
    expected_graph_mode: str,
    expected_preferred_snapshot: float,
    expected_periodic_boundary: bool,
    expected_box_size: float,
) -> Dict[str, Any]:
    """
    Validate the whole saved static graph dataset.
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

    target_values: List[float] = []
    degree_means: List[float] = []
    edge_counts: List[int] = []

    for universe_id in universe_ids:
        sample = dataset[universe_id]

        if not isinstance(sample, dict):
            all_errors.append(f"{universe_id}: expected dict sample, got {type(sample)}")
            continue

        errors, report = validate_one_universe(
            universe_id=str(universe_id),
            sample=sample,
            expected_nodes=expected_nodes,
            expected_features=expected_features,
            normalization=normalization,
            expected_graph_mode=expected_graph_mode,
            expected_preferred_snapshot=expected_preferred_snapshot,
            expected_periodic_boundary=expected_periodic_boundary,
            expected_box_size=expected_box_size,
        )

        all_errors.extend(errors)
        all_warnings.extend(report.get("warnings", []))
        universe_reports[str(universe_id)] = report

        target_value = report.get("target")
        if target_value is not None:
            target_values.append(float(target_value))

        graph_info = report.get("graph", {})
        degree_mean = graph_info.get("degree_mean")
        if degree_mean is not None:
            degree_means.append(float(degree_mean))

        edge_count = graph_info.get("estimated_undirected_edges_if_symmetric")
        if edge_count is not None:
            edge_counts.append(int(edge_count))

    dataset_summary: Dict[str, Any] = {
        "target_count": len(target_values),
        "target_min": min(target_values) if target_values else None,
        "target_max": max(target_values) if target_values else None,
        "target_mean": sum(target_values) / len(target_values) if target_values else None,
        "degree_mean_average": (
            sum(degree_means) / len(degree_means) if degree_means else None
        ),
        "edge_count_min": min(edge_counts) if edge_counts else None,
        "edge_count_max": max(edge_counts) if edge_counts else None,
        "edge_count_mean": (
            sum(edge_counts) / len(edge_counts) if edge_counts else None
        ),
    }

    return {
        "passed": len(all_errors) == 0,
        "dataset_type": "static_final_snapshot_graphs",
        "preprocessing_version_expected": PREPROCESSING_VERSION,
        "feature_names_expected": FEATURE_NAMES,
        "expected_graph_mode": expected_graph_mode,
        "expected_preferred_snapshot": expected_preferred_snapshot,
        "expected_periodic_boundary": expected_periodic_boundary,
        "expected_box_size": expected_box_size,
        "num_universes_found": len(universe_ids),
        "expected_universes": expected_universes,
        "expected_nodes": expected_nodes,
        "expected_features": expected_features,
        "normalization": normalization,
        "dataset_summary": dataset_summary,
        "errors": all_errors,
        "warnings": all_warnings,
        "universes": universe_reports,
    }


# ============================================================
# Reporting
# ============================================================

def print_report(report: Dict[str, Any], max_show: int = 5) -> None:
    """
    Print a readable validation summary.
    """
    print("=" * 90)
    print("CAMELS-SIMBA STATIC GRAPH VALIDATION")
    print("=" * 90)

    print(f"Passed:                     {report['passed']}")
    print(f"Dataset type:               {report['dataset_type']}")
    print(f"Expected preprocessing:     {report['preprocessing_version_expected']}")
    print(f"Expected features:          {report['feature_names_expected']}")
    print(f"Expected graph mode:        {report['expected_graph_mode']}")
    print(f"Expected snapshot:          {report['expected_preferred_snapshot']}")
    print(f"Expected periodic boundary: {report['expected_periodic_boundary']}")
    print(f"Expected box size:          {report['expected_box_size']}")
    print(f"Universes found:            {report['num_universes_found']}")
    print(f"Expected universes:         {report['expected_universes']}")
    print(f"Expected nodes:             {report['expected_nodes']}")
    print(f"Expected features count:    {report['expected_features']}")
    print(f"Normalization:              {report['normalization']}")

    print("\nDataset summary:")
    print("-" * 90)
    summary = report.get("dataset_summary", {})
    print(f"Target count:               {summary.get('target_count')}")
    print(f"Target min:                 {summary.get('target_min')}")
    print(f"Target max:                 {summary.get('target_max')}")
    print(f"Target mean:                {summary.get('target_mean')}")
    print(f"Average degree mean:        {summary.get('degree_mean_average')}")
    print(f"Edge count min:             {summary.get('edge_count_min')}")
    print(f"Edge count max:             {summary.get('edge_count_max')}")
    print(f"Edge count mean:            {summary.get('edge_count_mean')}")

    print("\nErrors:")
    print("-" * 90)

    if report["errors"]:
        for err in report["errors"][:50]:
            print(f"❌ {err}")

        if len(report["errors"]) > 50:
            print(f"... and {len(report['errors']) - 50} more errors")
    else:
        print("✅ No errors found.")

    print("\nWarnings:")
    print("-" * 90)

    if report["warnings"]:
        for warn in report["warnings"][:50]:
            print(f"⚠️ {warn}")

        if len(report["warnings"]) > 50:
            print(f"... and {len(report['warnings']) - 50} more warnings")
    else:
        print("✅ No warnings found.")

    print("\nFirst universe summaries:")
    print("-" * 90)

    shown = 0

    for universe_id, u_report in report["universes"].items():
        if shown >= max_show:
            break

        metadata = u_report.get("metadata", {})
        graph = u_report.get("graph", {})
        x_summary = u_report.get("X", {})
        a_summary = u_report.get("A", {})
        mask_summary = u_report.get("mask", {})

        print(f"\nUniverse: {universe_id}")
        print(f"  Target Omega_m:          {u_report.get('target')}")
        print(f"  Preprocessing:           {metadata.get('preprocessing_version')}")
        print(f"  Mass feature:            {metadata.get('mass_feature')}")
        print(f"  Preferred snapshot:      {metadata.get('preferred_snapshot')}")
        print(f"  Actual snapshot:         {metadata.get('actual_snapshot_value')}")
        print(f"  Exact snapshot match:    {metadata.get('exact_preferred_snapshot_match')}")
        print(f"  Periodic boundary:       {metadata.get('periodic_boundary')}")
        print(f"  Periodic kNN:            {metadata.get('periodic_boundary_knn')}")
        print(f"  Box size:                {metadata.get('box_size')}")
        print(f"  A shape:                 {a_summary.get('shape')}")
        print(f"  X shape:                 {x_summary.get('shape')}")
        print(f"  mask shape:              {mask_summary.get('shape')}")
        print(
            f"  Degree:                  min={graph.get('degree_min')} "
            f"mean={graph.get('degree_mean')} "
            f"max={graph.get('degree_max')}"
        )
        print(
            f"  Estimated edges:         "
            f"{graph.get('estimated_undirected_edges_if_symmetric')}"
        )
        print(f"  X range:                 min={x_summary.get('min')} max={x_summary.get('max')}")
        print(f"  Real nodes:              {u_report.get('real_nodes')}")
        print(f"  Mask unique values:      {u_report.get('mask_unique_values')}")

        shown += 1

    print("\n" + "=" * 90)

    if report["passed"]:
        print("✅ Validation complete. Static dataset is structurally and scientifically valid.")
    else:
        print("❌ Validation complete. Static dataset has errors that must be fixed.")

    print("=" * 90)


def save_report(report: Dict[str, Any], output_path: Path) -> None:
    """
    Save validation report as JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, indent=2)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate saved CAMELS-SIMBA static graph dataset."
    )

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to saved static graph dataset .pt file.",
    )

    parser.add_argument(
        "--expected_universes",
        type=int,
        required=True,
        help="Expected number of universes in the saved dataset.",
    )

    parser.add_argument(
        "--expected_nodes",
        type=int,
        default=100,
        help="Expected number of nodes per static graph.",
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
        "--preferred_snapshot",
        type=float,
        default=1.0,
        help="Expected static snapshot scale factor.",
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

    print(f"Loading static dataset: {dataset_path}")
    dataset = load_dataset(dataset_path)

    report = validate_dataset(
        dataset=dataset,
        expected_universes=args.expected_universes,
        expected_nodes=args.expected_nodes,
        expected_features=args.expected_features,
        normalization=args.normalization,
        expected_graph_mode=args.graph_mode,
        expected_preferred_snapshot=args.preferred_snapshot,
        expected_periodic_boundary=args.periodic_boundary,
        expected_box_size=args.box_size,
    )

    print_report(report, max_show=args.max_show)

    if args.save_json:
        save_report(report, Path(args.save_json))
        print(f"\nValidation JSON saved to: {args.save_json}")


if __name__ == "__main__":
    main()