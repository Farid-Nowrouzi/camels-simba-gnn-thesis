"""
validate_sequences.py

Validation utility for saved CAMELS-SIMBA temporal graph datasets.

This script checks whether a saved temporal dataset has the expected structure:

dataset = {
    "LH_0": {
        "A_list": [...],
        "Nodes_list": [...],
        "mask_list": [...],
        "target": tensor(...),
        "snapshots": [...],
        ...
    },
    "LH_1": {...},
    ...
}

It is used before training to make sure the processed graph sequences are safe,
consistent, and scientifically usable.

Official current preprocessing:
    v2_logmass_minmax_top100_periodic_knn
"""

from __future__ import annotations

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


def tensor_summary(name: str, tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Return basic numeric checks for one tensor.
    """
    tensor_cpu = tensor.detach().cpu()

    summary = {
        "name": name,
        "shape": list(tensor_cpu.shape),
        "dtype": str(tensor_cpu.dtype),
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
    A_cpu = A.detach().cpu()

    if A_cpu.ndim != 2:
        return {
            "valid": False,
            "reason": f"A is not 2D. Shape={tuple(A_cpu.shape)}",
        }

    n = A_cpu.shape[0]
    nonzero = int((A_cpu > 0).sum().item())

    diag_nonzero = (
        int((torch.diag(A_cpu) > 0).sum().item())
        if A_cpu.shape[0] == A_cpu.shape[1]
        else -1
    )

    degree = (A_cpu > 0).sum(dim=1).float()

    symmetry_error = None
    is_symmetric = False

    if A_cpu.shape[0] == A_cpu.shape[1]:
        symmetry_error = float(torch.abs(A_cpu - A_cpu.T).sum().item())
        is_symmetric = symmetry_error == 0.0

    return {
        "valid": True,
        "num_nodes": int(n),
        "nonzero_entries": nonzero,
        "estimated_undirected_edges_if_symmetric": (
            nonzero // 2 if is_symmetric else None
        ),
        "diag_nonzero": diag_nonzero,
        "is_symmetric": is_symmetric,
        "symmetry_error": symmetry_error,
        "isolated_nodes": int((degree == 0).sum().item()),
        "degree_min": float(degree.min().item()) if degree.numel() > 0 else None,
        "degree_max": float(degree.max().item()) if degree.numel() > 0 else None,
        "degree_mean": float(degree.mean().item()) if degree.numel() > 0 else None,
        "degree_std": float(degree.std().item()) if degree.numel() > 1 else 0.0,
    }


def load_dataset(path: Path) -> Dict[str, Any]:
    """
    Load dataset safely.

    In PyTorch 2.6+, torch.load defaults to weights_only=True.
    Our dataset contains dictionaries and tensors, so we explicitly set
    weights_only=False because this is our own trusted file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    return torch.load(path, map_location="cpu", weights_only=False)


def validate_metadata(
    universe_id: str,
    sample: Dict[str, Any],
    expected_normalization: str,
    expected_periodic_boundary: bool,
    expected_box_size: float,
    expected_graph_mode: str,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Validate thesis/science metadata for one universe.
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

    snapshots = sample.get("snapshots")
    if isinstance(snapshots, list) and snapshots:
        first_meta = snapshots[0]

        if first_meta.get("preprocessing_version") != PREPROCESSING_VERSION:
            errors.append(
                f"{universe_id}: first snapshot preprocessing_version="
                f"{first_meta.get('preprocessing_version')} != {PREPROCESSING_VERSION}"
            )

        if first_meta.get("periodic_boundary") != expected_periodic_boundary:
            errors.append(
                f"{universe_id}: first snapshot periodic_boundary="
                f"{first_meta.get('periodic_boundary')} != {expected_periodic_boundary}"
            )

        if expected_graph_mode == "knn":
            expected_periodic_knn = bool(expected_periodic_boundary)
            if first_meta.get("periodic_boundary_knn") != expected_periodic_knn:
                errors.append(
                    f"{universe_id}: first snapshot periodic_boundary_knn="
                    f"{first_meta.get('periodic_boundary_knn')} "
                    f"!= {expected_periodic_knn}"
                )

        if first_meta.get("box_size") is None:
            errors.append(f"{universe_id}: first snapshot missing box_size")
        else:
            if abs(float(first_meta.get("box_size")) - float(expected_box_size)) > 1e-6:
                errors.append(
                    f"{universe_id}: first snapshot box_size="
                    f"{first_meta.get('box_size')} != {expected_box_size}"
                )
    else:
        warnings.append(f"{universe_id}: snapshots metadata missing or empty")

    return errors, warnings, metadata_report


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
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Validate one universe sequence.
    """
    errors: List[str] = []
    warnings: List[str] = []

    report: Dict[str, Any] = {
        "universe_id": universe_id,
        "errors": errors,
        "warnings": warnings,
    }

    required_keys = ["A_list", "Nodes_list", "mask_list", "target"]
    for key in required_keys:
        if key not in sample:
            errors.append(f"Missing required key: {key}")

    metadata_errors, metadata_warnings, metadata_report = validate_metadata(
        universe_id=universe_id,
        sample=sample,
        expected_normalization=normalization,
        expected_periodic_boundary=expected_periodic_boundary,
        expected_box_size=expected_box_size,
        expected_graph_mode=expected_graph_mode,
    )

    errors.extend(metadata_errors)
    warnings.extend(metadata_warnings)
    report["metadata"] = metadata_report

    if errors and any("Missing required key" in err for err in errors):
        return errors, report

    A_list = sample["A_list"]
    X_list = sample["Nodes_list"]
    mask_list = sample["mask_list"]
    target = sample["target"]

    report["num_snapshots_A"] = len(A_list)
    report["num_snapshots_X"] = len(X_list)
    report["num_snapshots_mask"] = len(mask_list)
    report["target"] = safe_float(target)

    if len(A_list) != expected_snapshots:
        errors.append(f"A_list length {len(A_list)} != expected {expected_snapshots}")

    if len(X_list) != expected_snapshots:
        errors.append(f"Nodes_list length {len(X_list)} != expected {expected_snapshots}")

    if len(mask_list) != expected_snapshots:
        errors.append(f"mask_list length {len(mask_list)} != expected {expected_snapshots}")

    target_value = safe_float(target)
    if target_value is None:
        errors.append("Target is not convertible to float.")
    else:
        if not (0.0 < target_value < 1.0):
            warnings.append(f"Target value looks unusual for Omega_m: {target_value}")

    snapshot_reports = []

    for t in range(min(len(A_list), len(X_list), len(mask_list))):
        A = A_list[t]
        X = X_list[t]
        mask = mask_list[t]

        snap_report = {
            "snapshot_index": t,
            "A": tensor_summary("A", A),
            "X": tensor_summary("X", X),
            "mask": tensor_summary("mask", mask),
            "graph": adjacency_stats(A),
        }

        if tuple(A.shape) != (expected_nodes, expected_nodes):
            errors.append(
                f"{universe_id} snapshot {t}: A shape {tuple(A.shape)} != "
                f"({expected_nodes}, {expected_nodes})"
            )

        if tuple(X.shape) != (expected_nodes, expected_features):
            errors.append(
                f"{universe_id} snapshot {t}: X shape {tuple(X.shape)} != "
                f"({expected_nodes}, {expected_features})"
            )

        if tuple(mask.shape) != (expected_nodes, 1):
            errors.append(
                f"{universe_id} snapshot {t}: mask shape {tuple(mask.shape)} != "
                f"({expected_nodes}, 1)"
            )

        if X.is_floating_point():
            if torch.isnan(X).any():
                errors.append(f"{universe_id} snapshot {t}: X contains NaN values.")
            if torch.isinf(X).any():
                errors.append(f"{universe_id} snapshot {t}: X contains infinite values.")

        if A.is_floating_point():
            if torch.isnan(A).any():
                errors.append(f"{universe_id} snapshot {t}: A contains NaN values.")
            if torch.isinf(A).any():
                errors.append(f"{universe_id} snapshot {t}: A contains infinite values.")

        if mask.is_floating_point():
            if torch.isnan(mask).any():
                errors.append(f"{universe_id} snapshot {t}: mask contains NaN values.")
            if torch.isinf(mask).any():
                errors.append(f"{universe_id} snapshot {t}: mask contains infinite values.")

        graph_info = snap_report["graph"]
        if graph_info.get("valid"):
            if graph_info["nonzero_entries"] == 0:
                errors.append(f"{universe_id} snapshot {t}: graph has no edges.")

            if graph_info["isolated_nodes"] > 0:
                warnings.append(
                    f"{universe_id} snapshot {t}: graph has "
                    f"{graph_info['isolated_nodes']} isolated nodes."
                )

            if graph_info["is_symmetric"] is False:
                warnings.append(
                    f"{universe_id} snapshot {t}: adjacency is not symmetric."
                )

            if graph_info["diag_nonzero"] != 0:
                warnings.append(
                    f"{universe_id} snapshot {t}: adjacency has "
                    f"{graph_info['diag_nonzero']} self-loops."
                )

        if normalization == "minmax":
            x_min = float(X.min().item())
            x_max = float(X.max().item())

            tolerance = 1e-5
            if x_min < -tolerance or x_max > 1.0 + tolerance:
                errors.append(
                    f"{universe_id} snapshot {t}: normalization=minmax but X range is "
                    f"[{x_min}, {x_max}], expected approximately [0, 1]."
                )

        snapshot_reports.append(snap_report)

    report["snapshots"] = snapshot_reports

    return errors, report


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
    universe_reports = {}

    if not isinstance(dataset, dict):
        raise TypeError(f"Expected dataset to be dict, got {type(dataset)}")

    universe_ids = sort_universe_ids(list(dataset.keys()))

    if len(universe_ids) != expected_universes:
        all_errors.append(
            f"Number of universes {len(universe_ids)} != expected {expected_universes}"
        )

    for universe_id in universe_ids:
        sample = dataset[universe_id]

        if not isinstance(sample, dict):
            all_errors.append(f"{universe_id}: expected dict sample, got {type(sample)}")
            continue

        errors, report = validate_one_universe(
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
        all_warnings.extend(report.get("warnings", []))
        universe_reports[universe_id] = report

    return {
        "passed": len(all_errors) == 0,
        "preprocessing_version_expected": PREPROCESSING_VERSION,
        "feature_names_expected": FEATURE_NAMES,
        "expected_periodic_boundary": expected_periodic_boundary,
        "expected_box_size": expected_box_size,
        "expected_graph_mode": expected_graph_mode,
        "num_universes_found": len(universe_ids),
        "expected_universes": expected_universes,
        "expected_snapshots": expected_snapshots,
        "expected_nodes": expected_nodes,
        "expected_features": expected_features,
        "normalization": normalization,
        "errors": all_errors,
        "warnings": all_warnings,
        "universes": universe_reports,
    }


def print_report(report: Dict[str, Any], max_show: int = 5) -> None:
    """
    Print a readable validation summary.
    """
    print("=" * 90)
    print("CAMELS-SIMBA TEMPORAL SEQUENCE VALIDATION")
    print("=" * 90)

    print(f"Passed:                    {report['passed']}")
    print(f"Expected preprocessing:    {report['preprocessing_version_expected']}")
    print(f"Expected features:         {report['feature_names_expected']}")
    print(f"Expected graph mode:       {report['expected_graph_mode']}")
    print(f"Expected periodic boundary:{report['expected_periodic_boundary']}")
    print(f"Expected box size:         {report['expected_box_size']}")
    print(f"Universes found:           {report['num_universes_found']}")
    print(f"Expected universes:        {report['expected_universes']}")
    print(f"Expected snapshots:        {report['expected_snapshots']}")
    print(f"Expected nodes:            {report['expected_nodes']}")
    print(f"Expected features count:   {report['expected_features']}")
    print(f"Normalization:             {report['normalization']}")

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

        print(f"\nUniverse: {universe_id}")
        print(f"  Target Omega_m:       {u_report.get('target')}")
        print(f"  A snapshots:          {u_report.get('num_snapshots_A')}")
        print(f"  X snapshots:          {u_report.get('num_snapshots_X')}")
        print(f"  Mask snapshots:       {u_report.get('num_snapshots_mask')}")

        metadata = u_report.get("metadata", {})
        print(f"  Preprocessing:        {metadata.get('preprocessing_version')}")
        print(f"  Mass feature:         {metadata.get('mass_feature')}")
        print(f"  Periodic boundary:    {metadata.get('periodic_boundary')}")
        print(f"  Periodic kNN:         {metadata.get('periodic_boundary_knn')}")
        print(f"  Box size:             {metadata.get('box_size')}")

        snapshots = u_report.get("snapshots", [])
        if snapshots:
            first = snapshots[0]
            print(f"  First A shape:        {first['A']['shape']}")
            print(f"  First X shape:        {first['X']['shape']}")
            print(f"  First mask shape:     {first['mask']['shape']}")
            print(
                f"  First graph degree:   min={first['graph'].get('degree_min')} "
                f"mean={first['graph'].get('degree_mean')} "
                f"max={first['graph'].get('degree_max')}"
            )
            if "min" in first["X"]:
                print(
                    f"  First X range:        min={first['X']['min']} "
                    f"max={first['X']['max']}"
                )

        shown += 1

    print("\n" + "=" * 90)
    if report["passed"]:
        print("✅ Validation complete. Dataset is structurally and scientifically valid.")
    else:
        print("❌ Validation complete. Dataset has errors that must be fixed.")
    print("=" * 90)


def save_report(report: Dict[str, Any], output_path: Path) -> None:
    """
    Save validation report as JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate saved CAMELS-SIMBA temporal graph sequence dataset."
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

    print(f"Loading dataset: {dataset_path}")
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