"""
graph_diagnostics.py

Professional one-graph diagnostics for CAMELS-SIMBA halo graphs.

This script builds one graph from one raw CAMELS-SIMBA snapshot file and prints
plus saves diagnostics about:

- preprocessing version
- node feature matrix
- adjacency matrix
- graph density
- degree statistics
- isolated nodes
- symmetry
- selected physical features
- top-N halo selection by raw Mvir
- log10(Mvir) feature construction
- periodic boundary-aware graph construction
- saved JSON/CSV summaries

This script is used before building full static/temporal datasets.

Official preprocessing:
    v2_logmass_minmax_top100_periodic_knn

Example usage:

python -m src.data.graph_diagnostics \
  --snapshot_path data/raw/CAMELS_SIMBA_100U/LH_0_hlist_1.00000.list \
  --num_nodes 100 \
  --normalization minmax \
  --graph_mode knn \
  --k 8 \
  --periodic_boundary \
  --box_size 25.0 \
  --output_dir outputs/graph_diagnostics/v2_logmass_minmax_top100_periodic_knn/LH_0_snapshot_1.00000 \
  --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.data.camels_graph_utils import (
    DEFAULT_BOX_SIZE,
    FEATURE_NAMES,
    MASS_COLUMN,
    POSITION_COLUMNS,
    PREPROCESSING_VERSION,
    build_positions,
    clean_halo_dataframe,
    process_snapshot,
    read_hlist_file,
    select_top_halos,
)


# ============================================================
# Save helpers
# ============================================================

def save_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Save dictionary as JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    Save list of dictionaries as CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"No rows to save for CSV: {path}")

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Tensor / graph statistics
# ============================================================

def tensor_stats_dict(name: str, tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Return basic tensor statistics as a dictionary.
    """
    tensor_cpu = tensor.detach().cpu()

    stats: Dict[str, Any] = {
        "name": name,
        "shape": list(tensor_cpu.shape),
        "dtype": str(tensor_cpu.dtype),
        "device": str(tensor.device),
        "numel": int(tensor_cpu.numel()),
    }

    if tensor_cpu.numel() == 0:
        stats.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "nan_count": 0,
                "inf_count": 0,
            }
        )
        return stats

    tensor_float = tensor_cpu.float()

    stats.update(
        {
            "min": float(tensor_float.min().item()),
            "max": float(tensor_float.max().item()),
            "mean": float(tensor_float.mean().item()),
            "std": float(tensor_float.std(unbiased=False).item()),
            "nan_count": int(torch.isnan(tensor_float).sum().item()),
            "inf_count": int(torch.isinf(tensor_float).sum().item()),
        }
    )

    return stats


def print_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    """
    Print basic statistics for one tensor.
    """
    stats = tensor_stats_dict(name, tensor)

    print(f"\n{name}")
    print("-" * 80)
    print(f"Shape: {tuple(stats['shape'])}")
    print(f"Dtype: {stats['dtype']}")
    print(f"Device: {stats['device']}")

    if stats["numel"] == 0:
        print("Tensor is empty.")
        return

    print(f"Min:  {stats['min']:.6g}")
    print(f"Max:  {stats['max']:.6g}")
    print(f"Mean: {stats['mean']:.6g}")
    print(f"Std:  {stats['std']:.6g}")
    print(f"NaN count: {stats['nan_count']}")
    print(f"Inf count: {stats['inf_count']}")


def analyze_adjacency(A: torch.Tensor) -> Dict[str, Any]:
    """
    Analyze adjacency matrix statistics.
    """
    A_cpu = A.detach().cpu().float()

    if A_cpu.ndim != 2 or A_cpu.shape[0] != A_cpu.shape[1]:
        raise ValueError(
            f"Expected square adjacency matrix, got shape {tuple(A_cpu.shape)}"
        )

    num_nodes = A_cpu.shape[0]

    total_nonzero = int((A_cpu != 0).sum().item())
    diagonal_nonzero = int((torch.diag(A_cpu) != 0).sum().item())
    diagonal_sum = float(torch.diag(A_cpu).sum().item())

    directed_edge_count = total_nonzero - diagonal_nonzero

    possible_directed_edges = num_nodes * (num_nodes - 1)
    density = (
        directed_edge_count / possible_directed_edges
        if possible_directed_edges > 0
        else 0.0
    )

    degree = A_cpu.sum(dim=1)

    isolated_nodes = int((degree == 0).sum().item())

    symmetry_error = float(torch.abs(A_cpu - A_cpu.T).sum().item())
    is_symmetric = symmetry_error == 0.0

    estimated_undirected_edges = (
        directed_edge_count // 2 if is_symmetric else None
    )

    return {
        "num_nodes": int(num_nodes),
        "total_nonzero_entries": int(total_nonzero),
        "diagonal_nonzero_entries": int(diagonal_nonzero),
        "diagonal_sum": diagonal_sum,
        "directed_edge_count_excluding_diagonal": int(directed_edge_count),
        "estimated_undirected_edges_if_symmetric": estimated_undirected_edges,
        "density": float(density),
        "degree_min": float(degree.min().item()),
        "degree_max": float(degree.max().item()),
        "degree_mean": float(degree.mean().item()),
        "degree_std": float(degree.std(unbiased=False).item()),
        "isolated_nodes": int(isolated_nodes),
        "symmetry_error": float(symmetry_error),
        "is_symmetric": bool(is_symmetric),
    }


# ============================================================
# Feature and selection summaries
# ============================================================

def build_feature_summary_rows(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build feature-level summary rows from processed node feature matrix X.

    These are ML features after preprocessing and normalization.
    """
    X = snapshot["X"].detach().cpu().float()
    mask = snapshot["mask"].detach().cpu().reshape(-1)

    valid_indices = mask > 0
    X_valid = X[valid_indices]

    feature_names = snapshot.get("feature_names", FEATURE_NAMES)

    rows: List[Dict[str, Any]] = []

    for idx, feature_name in enumerate(feature_names):
        column_all = X[:, idx]
        column_valid = X_valid[:, idx]

        rows.append(
            {
                "feature_index": idx,
                "feature_name": feature_name,
                "processed_feature_space": "normalized_or_raw_X_tensor",
                "num_all_nodes": int(X.shape[0]),
                "num_valid_nodes": int(X_valid.shape[0]),
                "all_min": float(column_all.min().item()),
                "all_max": float(column_all.max().item()),
                "all_mean": float(column_all.mean().item()),
                "all_std": float(column_all.std(unbiased=False).item()),
                "valid_min": float(column_valid.min().item()),
                "valid_max": float(column_valid.max().item()),
                "valid_mean": float(column_valid.mean().item()),
                "valid_std": float(column_valid.std(unbiased=False).item()),
                "nan_count": int(torch.isnan(column_all).sum().item()),
                "inf_count": int(torch.isinf(column_all).sum().item()),
            }
        )

    return rows


def build_node_selection_summary(snapshot_path: Path, num_nodes: int) -> Dict[str, Any]:
    """
    Reconstruct raw halo selection summary.

    This confirms:
        - raw Mvir is used for selection
        - selected mass feature is log10(Mvir)
        - top-N selection is available
    """
    df_raw = read_hlist_file(snapshot_path)
    df_clean = clean_halo_dataframe(df_raw)
    df_selected = select_top_halos(
        df_clean,
        num_nodes=num_nodes,
        mass_column=MASS_COLUMN,
    )

    raw_mvir_all = df_clean[MASS_COLUMN].to_numpy(dtype=np.float64)
    raw_mvir_selected = df_selected[MASS_COLUMN].to_numpy(dtype=np.float64)

    positions_selected = build_positions(df_selected)

    summary: Dict[str, Any] = {
        "snapshot_path": str(snapshot_path),
        "requested_num_nodes": int(num_nodes),
        "num_raw_halos_before_cleaning": int(len(df_raw)),
        "num_valid_halos_after_cleaning": int(len(df_clean)),
        "num_selected_halos_before_padding": int(len(df_selected)),
        "selection_rule": "top_num_nodes_by_raw_Mvir_descending",
        "selection_mass_column": MASS_COLUMN,
        "mass_feature_after_selection": "log10_Mvir",
        "padding_needed": bool(len(df_selected) < num_nodes),
        "raw_mvir_all_min": float(raw_mvir_all.min()) if len(raw_mvir_all) else None,
        "raw_mvir_all_max": float(raw_mvir_all.max()) if len(raw_mvir_all) else None,
        "raw_mvir_all_mean": float(raw_mvir_all.mean()) if len(raw_mvir_all) else None,
        "raw_mvir_selected_min": (
            float(raw_mvir_selected.min()) if len(raw_mvir_selected) else None
        ),
        "raw_mvir_selected_max": (
            float(raw_mvir_selected.max()) if len(raw_mvir_selected) else None
        ),
        "raw_mvir_selected_mean": (
            float(raw_mvir_selected.mean()) if len(raw_mvir_selected) else None
        ),
        "log10_mvir_selected_min": (
            float(np.log10(raw_mvir_selected).min()) if len(raw_mvir_selected) else None
        ),
        "log10_mvir_selected_max": (
            float(np.log10(raw_mvir_selected).max()) if len(raw_mvir_selected) else None
        ),
        "log10_mvir_selected_mean": (
            float(np.log10(raw_mvir_selected).mean()) if len(raw_mvir_selected) else None
        ),
        "position_x_min": float(positions_selected[:, 0].min()) if len(positions_selected) else None,
        "position_x_max": float(positions_selected[:, 0].max()) if len(positions_selected) else None,
        "position_y_min": float(positions_selected[:, 1].min()) if len(positions_selected) else None,
        "position_y_max": float(positions_selected[:, 1].max()) if len(positions_selected) else None,
        "position_z_min": float(positions_selected[:, 2].min()) if len(positions_selected) else None,
        "position_z_max": float(positions_selected[:, 2].max()) if len(positions_selected) else None,
    }

    return summary


def build_node_selection_csv_rows(selection_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert node selection summary dictionary into a simple key-value CSV format.
    """
    return [
        {
            "key": key,
            "value": value,
        }
        for key, value in selection_summary.items()
    ]


# ============================================================
# Full report construction
# ============================================================

def build_diagnostics_summary(
    snapshot: Dict[str, Any],
    adjacency_stats: Dict[str, Any],
    feature_summary_rows: List[Dict[str, Any]],
    node_selection_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build full JSON-serializable graph diagnostic report.
    """
    A = snapshot["A"]
    X = snapshot["X"]
    mask = snapshot["mask"]

    summary = {
        "preprocessing_version": snapshot.get("preprocessing_version"),
        "expected_preprocessing_version": PREPROCESSING_VERSION,
        "snapshot_metadata": {
            "snapshot_value": snapshot.get("snapshot_value"),
            "path": snapshot.get("path"),
            "feature_names": snapshot.get("feature_names"),
            "feature_columns": snapshot.get("feature_columns"),
            "raw_feature_columns": snapshot.get("raw_feature_columns"),
            "mass_column": snapshot.get("mass_column"),
            "mass_feature": snapshot.get("mass_feature"),
            "node_selection": snapshot.get("node_selection"),
            "position_columns": snapshot.get("position_columns"),
            "velocity_columns": snapshot.get("velocity_columns"),
            "normalization": snapshot.get("normalization"),
            "graph_mode": snapshot.get("graph_mode"),
            "graph_positions": snapshot.get("graph_positions"),
            "k": snapshot.get("k"),
            "radius": snapshot.get("radius"),
            "periodic_boundary": snapshot.get("periodic_boundary"),
            "periodic_boundary_knn": snapshot.get("periodic_boundary_knn"),
            "box_size": snapshot.get("box_size"),
            "original_num_halos": snapshot.get("original_num_halos"),
            "valid_num_halos": snapshot.get("valid_num_halos"),
            "selected_num_halos_before_padding": snapshot.get(
                "selected_num_halos_before_padding"
            ),
            "num_nodes": snapshot.get("num_nodes"),
        },
        "tensor_summaries": {
            "A": tensor_stats_dict("A", A),
            "X": tensor_stats_dict("X", X),
            "mask": tensor_stats_dict("mask", mask),
        },
        "adjacency_stats": adjacency_stats,
        "feature_summary": feature_summary_rows,
        "node_selection_summary": node_selection_summary,
        "checklist": build_checklist(snapshot, adjacency_stats),
    }

    return summary


def build_checklist(
    snapshot: Dict[str, Any],
    adjacency_stats: Dict[str, Any],
) -> Dict[str, bool]:
    """
    Build pass/fail checklist as booleans.
    """
    X = snapshot["X"]
    A = snapshot["A"]
    mask = snapshot["mask"]

    checklist = {
        "preprocessing_version_matches": (
            snapshot.get("preprocessing_version") == PREPROCESSING_VERSION
        ),
        "feature_names_match_expected": (
            snapshot.get("feature_names") == FEATURE_NAMES
        ),
        "mass_feature_is_log10_mvir": (
            snapshot.get("mass_feature") == "log10_Mvir"
        ),
        "node_selection_is_top_raw_mvir": (
            snapshot.get("node_selection") == "top_num_nodes_by_raw_Mvir_descending"
        ),
        "graph_uses_raw_physical_positions": (
            snapshot.get("graph_positions")
            == "raw_physical_XYZ_before_feature_normalization"
        ),
        "periodic_boundary_enabled": (
            snapshot.get("periodic_boundary") is True
        ),
        "periodic_knn_enabled_when_knn": (
            snapshot.get("graph_mode") != "knn"
            or snapshot.get("periodic_boundary_knn") is True
        ),
        "box_size_is_default": (
            abs(float(snapshot.get("box_size", -1)) - float(DEFAULT_BOX_SIZE)) < 1e-6
        ),
        "adjacency_is_symmetric": bool(adjacency_stats["is_symmetric"]),
        "graph_has_edges": bool(adjacency_stats["total_nonzero_entries"] > 0),
        "no_isolated_nodes": bool(adjacency_stats["isolated_nodes"] == 0),
        "x_has_no_nan": bool(not torch.isnan(X.detach().cpu().float()).any()),
        "x_has_no_inf": bool(not torch.isinf(X.detach().cpu().float()).any()),
        "a_has_no_nan": bool(not torch.isnan(A.detach().cpu().float()).any()),
        "a_has_no_inf": bool(not torch.isinf(A.detach().cpu().float()).any()),
        "mask_has_no_nan": bool(not torch.isnan(mask.detach().cpu().float()).any()),
        "mask_has_no_inf": bool(not torch.isinf(mask.detach().cpu().float()).any()),
        "node_feature_dimension_is_7": bool(X.shape[1] == 7),
        "real_nodes_exist": bool(mask.detach().cpu().sum().item() > 0),
    }

    return checklist


# ============================================================
# Printing
# ============================================================

def print_graph_diagnostics(
    snapshot: Dict[str, Any],
    adjacency_stats: Dict[str, Any],
    feature_summary_rows: List[Dict[str, Any]],
    node_selection_summary: Dict[str, Any],
) -> None:
    """
    Print graph diagnostics for a processed snapshot dictionary.
    """
    A = snapshot["A"]
    X = snapshot["X"]
    mask = snapshot["mask"]

    print("\n" + "=" * 90)
    print("CAMELS-SIMBA GRAPH DIAGNOSTICS")
    print("=" * 90)

    print("\nSnapshot metadata")
    print("-" * 80)
    print(f"Preprocessing version: {snapshot.get('preprocessing_version')}")
    print(f"Expected version:      {PREPROCESSING_VERSION}")
    print(f"Snapshot value:        {snapshot.get('snapshot_value')}")
    print(f"Path:                  {snapshot.get('path')}")
    print(f"Feature names:         {snapshot.get('feature_names')}")
    print(f"Feature columns:       {snapshot.get('feature_columns')}")
    print(f"Raw feature columns:   {snapshot.get('raw_feature_columns')}")
    print(f"Mass column:           {snapshot.get('mass_column')}")
    print(f"Mass feature:          {snapshot.get('mass_feature')}")
    print(f"Node selection:        {snapshot.get('node_selection')}")
    print(f"Position columns:      {snapshot.get('position_columns')}")
    print(f"Velocity columns:      {snapshot.get('velocity_columns')}")
    print(f"Normalization:         {snapshot.get('normalization')}")
    print(f"Graph mode:            {snapshot.get('graph_mode')}")
    print(f"Graph positions:       {snapshot.get('graph_positions')}")
    print(f"k:                     {snapshot.get('k')}")
    print(f"Radius:                {snapshot.get('radius')}")
    print(f"Periodic boundary:     {snapshot.get('periodic_boundary')}")
    print(f"Periodic boundary kNN: {snapshot.get('periodic_boundary_knn')}")
    print(f"Box size:              {snapshot.get('box_size')}")
    print(f"Original halos:        {snapshot.get('original_num_halos')}")
    print(f"Valid halos:           {snapshot.get('valid_num_halos')}")
    print(
        "Selected halos before padding: "
        f"{snapshot.get('selected_num_halos_before_padding')}"
    )
    print(f"Number of nodes requested: {snapshot.get('num_nodes')}")

    print_tensor_stats("Node feature matrix X", X)
    print_tensor_stats("Adjacency matrix A", A)
    print_tensor_stats("Node mask", mask)

    print("\nAdjacency / graph statistics")
    print("-" * 80)

    for key, value in adjacency_stats.items():
        print(f"{key}: {value}")

    print("\nFeature-level quick check")
    print("-" * 80)

    for row in feature_summary_rows:
        print(
            f"{row['feature_index']:02d} | {row['feature_name']:<12} "
            f"valid_min={row['valid_min']:>12.6g} "
            f"valid_max={row['valid_max']:>12.6g} "
            f"valid_mean={row['valid_mean']:>12.6g} "
            f"valid_std={row['valid_std']:>12.6g}"
        )

    print("\nNode selection summary")
    print("-" * 80)
    print(f"Raw halos before cleaning:     {node_selection_summary['num_raw_halos_before_cleaning']}")
    print(f"Valid halos after cleaning:    {node_selection_summary['num_valid_halos_after_cleaning']}")
    print(f"Selected halos before padding: {node_selection_summary['num_selected_halos_before_padding']}")
    print(f"Padding needed:                {node_selection_summary['padding_needed']}")
    print(f"Selection rule:                {node_selection_summary['selection_rule']}")
    print(f"Selected raw Mvir min:         {node_selection_summary['raw_mvir_selected_min']}")
    print(f"Selected raw Mvir max:         {node_selection_summary['raw_mvir_selected_max']}")
    print(f"Selected log10(Mvir) min:      {node_selection_summary['log10_mvir_selected_min']}")
    print(f"Selected log10(Mvir) max:      {node_selection_summary['log10_mvir_selected_max']}")

    checklist = build_checklist(snapshot, adjacency_stats)

    print("\nInterpretation checklist")
    print("-" * 80)

    def show_check(condition: bool, success: str, warning: str) -> None:
        print(success if condition else warning)

    show_check(
        checklist["preprocessing_version_matches"],
        "✅ Preprocessing version matches the current official version.",
        "⚠️ Preprocessing version does NOT match the current official version.",
    )

    show_check(
        checklist["feature_names_match_expected"],
        "✅ Feature names match expected order.",
        "⚠️ Feature names do not match expected order.",
    )

    show_check(
        checklist["mass_feature_is_log10_mvir"],
        "✅ Mass feature is log10_Mvir.",
        "⚠️ Mass feature is not log10_Mvir.",
    )

    show_check(
        checklist["node_selection_is_top_raw_mvir"],
        "✅ Node selection uses top halos by raw Mvir.",
        "⚠️ Node selection rule is unexpected.",
    )

    show_check(
        checklist["graph_uses_raw_physical_positions"],
        "✅ Graph edges are built from raw physical X/Y/Z positions.",
        "⚠️ Graph position rule is unexpected.",
    )

    show_check(
        checklist["periodic_boundary_enabled"],
        "✅ Periodic boundary-aware distances are enabled.",
        "⚠️ Periodic boundary-aware distances are disabled.",
    )

    show_check(
        checklist["periodic_knn_enabled_when_knn"],
        "✅ kNN graph uses periodic boundary-aware distances.",
        "⚠️ kNN graph is not periodic boundary-aware.",
    )

    show_check(
        checklist["box_size_is_default"],
        f"✅ Box size is the CAMELS default: {DEFAULT_BOX_SIZE}.",
        f"⚠️ Box size is not the default CAMELS value: {DEFAULT_BOX_SIZE}.",
    )

    show_check(
        checklist["adjacency_is_symmetric"],
        "✅ Adjacency matrix is symmetric.",
        "⚠️ Adjacency matrix is NOT symmetric.",
    )

    show_check(
        checklist["graph_has_edges"],
        "✅ Graph has edges.",
        "❌ Graph has no edges.",
    )

    show_check(
        checklist["no_isolated_nodes"],
        "✅ No isolated nodes.",
        f"⚠️ Isolated nodes detected: {adjacency_stats['isolated_nodes']}",
    )

    show_check(
        checklist["x_has_no_nan"],
        "✅ No NaN values in node features.",
        "⚠️ NaN values detected in node features.",
    )

    show_check(
        checklist["x_has_no_inf"],
        "✅ No infinite values in node features.",
        "⚠️ Infinite values detected in node features.",
    )

    show_check(
        checklist["node_feature_dimension_is_7"],
        "✅ Node feature dimension is 7.",
        f"⚠️ Node feature dimension is {X.shape[1]}, expected 7.",
    )

    print("\nGraph diagnostics complete.")
    print("=" * 90)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect one CAMELS-SIMBA graph built from one raw snapshot."
    )

    parser.add_argument(
        "--snapshot_path",
        type=str,
        required=True,
        help="Path to one raw CAMELS-SIMBA .list snapshot file.",
    )

    parser.add_argument(
        "--num_nodes",
        type=int,
        default=100,
        help="Number of halos/nodes to keep from the snapshot.",
    )

    parser.add_argument(
        "--normalization",
        type=str,
        default="minmax",
        choices=["none", "minmax", "zscore"],
        help="Feature normalization mode.",
    )

    parser.add_argument(
        "--graph_mode",
        type=str,
        default="knn",
        choices=["knn", "radius"],
        help="Graph construction mode.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of nearest neighbors for kNN graph.",
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Radius threshold for radius graph mode.",
    )

    parser.add_argument(
        "--periodic_boundary",
        action="store_true",
        default=True,
        help="Use periodic boundary-aware distances. Enabled by default.",
    )

    parser.add_argument(
        "--no_periodic_boundary",
        action="store_false",
        dest="periodic_boundary",
        help="Disable periodic boundary-aware distances for ablation/debugging.",
    )

    parser.add_argument(
        "--box_size",
        type=float,
        default=DEFAULT_BOX_SIZE,
        help="Simulation box size. CAMELS default is 25.0 h^-1 Mpc.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional directory where diagnostic JSON/CSV files will be saved.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cpu or cuda.",
    )

    args = parser.parse_args()

    snapshot_path = Path(args.snapshot_path)

    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")

    snapshot = process_snapshot(
        path=snapshot_path,
        num_nodes=args.num_nodes,
        normalization=args.normalization,
        graph_mode=args.graph_mode,
        k=args.k,
        radius=args.radius,
        periodic_boundary=args.periodic_boundary,
        box_size=args.box_size,
        device=args.device,
    )

    adjacency_stats = analyze_adjacency(snapshot["A"])
    feature_summary_rows = build_feature_summary_rows(snapshot)
    node_selection_summary = build_node_selection_summary(
        snapshot_path=snapshot_path,
        num_nodes=args.num_nodes,
    )

    diagnostics_summary = build_diagnostics_summary(
        snapshot=snapshot,
        adjacency_stats=adjacency_stats,
        feature_summary_rows=feature_summary_rows,
        node_selection_summary=node_selection_summary,
    )

    print_graph_diagnostics(
        snapshot=snapshot,
        adjacency_stats=adjacency_stats,
        feature_summary_rows=feature_summary_rows,
        node_selection_summary=node_selection_summary,
    )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_json(
            path=output_dir / "graph_diagnostics_summary.json",
            data=diagnostics_summary,
        )

        save_csv_rows(
            path=output_dir / "04_feature_summary.csv",
            rows=feature_summary_rows,
        )

        save_csv_rows(
            path=output_dir / "10_node_selection_summary.csv",
            rows=build_node_selection_csv_rows(node_selection_summary),
        )

        print("\nDiagnostic files saved:")
        print("-" * 90)
        print(f"Graph diagnostics JSON: {output_dir / 'graph_diagnostics_summary.json'}")
        print(f"Feature summary CSV:    {output_dir / '04_feature_summary.csv'}")
        print(f"Node selection CSV:     {output_dir / '10_node_selection_summary.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()