"""
plot_graph_3d.py

Purpose
-------
Visualize and diagnose one CAMELS-SIMBA halo graph built from one raw snapshot.

This script is used before building full static/temporal datasets.

It helps verify that:
    1. Halo positions are read correctly.
    2. Selected thesis features are correct.
    3. Top-N halo selection by raw Mvir is meaningful.
    4. log10(Mvir) is used for the mass feature.
    5. Graph edges connect physically nearby halos.
    6. Feature normalization does not break graph construction.
    7. Periodic boundary-aware kNN is enabled correctly.
    8. The graph can be shown clearly in reports / meetings.
    9. Degree, mass, edge-distance, and projection diagnostics look reasonable.

Important
---------
Node features may be normalized for ML.
For plotting, this script reconstructs the raw selected physical positions and
raw log10(Mvir), so visualizations remain physically interpretable.

Official preprocessing:
    v2_logmass_minmax_top100_periodic_knn

Example usage
-------------
python -m src.data.plot_graph_3d \
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
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.camels_graph_utils import (
    DEFAULT_BOX_SIZE,
    MASS_COLUMN,
    POSITION_COLUMNS,
    PREPROCESSING_VERSION,
    build_positions,
    clean_halo_dataframe,
    parse_snapshot_value,
    process_snapshot,
    read_hlist_file,
    select_top_halos,
)


# ============================================================
# Basic helpers
# ============================================================

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Safely convert a PyTorch tensor to a NumPy array.
    """
    return tensor.detach().cpu().numpy()


def save_json(path: Path, data: Dict[str, object]) -> None:
    """
    Save a dictionary as a JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def periodic_delta(delta: np.ndarray, box_size: float) -> np.ndarray:
    """
    Apply minimum-image convention for periodic boxes.

    For coordinate difference dx, the periodic displacement is:
        dx_periodic = dx - box_size * round(dx / box_size)
    """
    return delta - box_size * np.round(delta / box_size)


def compute_edge_distances(
    positions: np.ndarray,
    edge_sources: np.ndarray,
    edge_targets: np.ndarray,
    periodic_boundary: bool,
    box_size: float,
) -> np.ndarray:
    """
    Compute physical edge lengths.

    If periodic_boundary=True, use minimum-image convention.
    """
    if len(edge_sources) == 0:
        return np.array([], dtype=np.float32)

    src_pos = positions[edge_sources]
    dst_pos = positions[edge_targets]

    diff = src_pos - dst_pos

    if periodic_boundary:
        diff = periodic_delta(diff, box_size=box_size)

    distances = np.sqrt(np.sum(diff * diff, axis=1))

    return distances.astype(np.float32)


# ============================================================
# Raw selected halo reconstruction
# ============================================================

def load_selected_halos(snapshot_path: str | Path, num_nodes: int):
    """
    Load the raw hlist file, clean it, and select the top-N halos by raw Mvir.

    This mirrors the official preprocessing selection rule.
    """
    df_raw = read_hlist_file(snapshot_path)
    df_clean = clean_halo_dataframe(df_raw)
    df_selected = select_top_halos(
        df_clean,
        num_nodes=num_nodes,
        mass_column=MASS_COLUMN,
    )

    return df_raw, df_clean, df_selected


def extract_raw_selected_positions(
    snapshot_path: str | Path,
    num_nodes: int,
) -> np.ndarray:
    """
    Reconstruct raw physical X/Y/Z positions for the selected halos.

    Why:
        snapshot["X"] may contain normalized ML features.
        For plots, raw physical coordinates are more meaningful.
    """
    _, _, df_selected = load_selected_halos(
        snapshot_path=snapshot_path,
        num_nodes=num_nodes,
    )

    positions = build_positions(df_selected)

    if positions.shape[0] < num_nodes:
        padded = np.zeros((num_nodes, 3), dtype=np.float32)
        padded[: positions.shape[0]] = positions
        positions = padded

    return positions.astype(np.float32)


def extract_raw_selected_logmass(
    snapshot_path: str | Path,
    num_nodes: int,
) -> np.ndarray:
    """
    Reconstruct raw log10(Mvir) for the selected halos.

    This is used for coloring nodes in plots.
    """
    _, _, df_selected = load_selected_halos(
        snapshot_path=snapshot_path,
        num_nodes=num_nodes,
    )

    raw_mvir = df_selected[MASS_COLUMN].to_numpy(dtype=np.float64)

    if np.any(raw_mvir <= 0):
        raise ValueError(
            "Non-positive Mvir found while extracting log10(Mvir). "
            "This should have been removed by clean_halo_dataframe()."
        )

    logmass = np.log10(raw_mvir).astype(np.float32)

    if logmass.shape[0] < num_nodes:
        padded = np.zeros((num_nodes,), dtype=np.float32)
        padded[: logmass.shape[0]] = logmass
        logmass = padded

    return logmass


def build_node_selection_summary(
    snapshot_path: str | Path,
    num_nodes: int,
) -> Dict[str, object]:
    """
    Build a compact summary of raw halo selection.
    """
    df_raw, df_clean, df_selected = load_selected_halos(
        snapshot_path=snapshot_path,
        num_nodes=num_nodes,
    )

    raw_mvir_all = df_clean[MASS_COLUMN].to_numpy(dtype=np.float64)
    raw_mvir_selected = df_selected[MASS_COLUMN].to_numpy(dtype=np.float64)

    summary = {
        "snapshot_path": str(snapshot_path),
        "num_raw_halos": int(len(df_raw)),
        "num_valid_halos_after_cleaning": int(len(df_clean)),
        "num_selected_halos_before_padding": int(len(df_selected)),
        "requested_num_nodes": int(num_nodes),
        "selection_rule": "top_num_nodes_by_raw_Mvir_descending",
        "mass_feature_after_selection": "log10_Mvir",
        "raw_mvir_all_min": float(raw_mvir_all.min()) if len(raw_mvir_all) else None,
        "raw_mvir_all_max": float(raw_mvir_all.max()) if len(raw_mvir_all) else None,
        "raw_mvir_selected_min": float(raw_mvir_selected.min()) if len(raw_mvir_selected) else None,
        "raw_mvir_selected_max": float(raw_mvir_selected.max()) if len(raw_mvir_selected) else None,
        "log10_mvir_selected_min": (
            float(np.log10(raw_mvir_selected).min()) if len(raw_mvir_selected) else None
        ),
        "log10_mvir_selected_max": (
            float(np.log10(raw_mvir_selected).max()) if len(raw_mvir_selected) else None
        ),
    }

    return summary


# ============================================================
# Graph extraction helpers
# ============================================================

def extract_edges_from_adjacency(A: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert adjacency matrix into undirected edge index arrays.

    Returns:
        source_indices, target_indices

    For symmetric adjacency, this keeps only source < target so each edge is
    drawn once.
    """
    A_np = tensor_to_numpy(A)

    sources, targets = np.where(A_np > 0)

    keep = sources < targets

    return sources[keep], targets[keep]


def filter_valid_nodes(
    positions: np.ndarray,
    logmass: np.ndarray,
    A: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """
    Remove padded nodes from positions, logmass, and adjacency.

    Usually CAMELS snapshots have enough halos, but this makes plotting robust.
    """
    mask_np = tensor_to_numpy(mask).reshape(-1) > 0
    valid_indices = np.where(mask_np)[0]

    positions_valid = positions[valid_indices]
    logmass_valid = logmass[valid_indices]

    A_valid = A.detach().cpu()[valid_indices][:, valid_indices]

    return positions_valid, logmass_valid, A_valid


# ============================================================
# Plotting functions
# ============================================================

def plot_3d_graph_static(
    positions: np.ndarray,
    logmass: np.ndarray,
    edge_sources: np.ndarray,
    edge_targets: np.ndarray,
    output_path: Path,
    title: str,
    max_edges_to_draw: int = 600,
) -> None:
    """
    Plot static 3D halo positions and graph edges.

    Node color:
        log10(Mvir)

    Edges:
        kNN / radius graph connections
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    num_edges = len(edge_sources)

    if num_edges > max_edges_to_draw:
        rng = np.random.default_rng(42)
        chosen = rng.choice(num_edges, size=max_edges_to_draw, replace=False)
        draw_sources = edge_sources[chosen]
        draw_targets = edge_targets[chosen]
    else:
        draw_sources = edge_sources
        draw_targets = edge_targets

    for src, dst in zip(draw_sources, draw_targets):
        x_vals = [positions[src, 0], positions[dst, 0]]
        y_vals = [positions[src, 1], positions[dst, 1]]
        z_vals = [positions[src, 2], positions[dst, 2]]

        ax.plot(
            x_vals,
            y_vals,
            z_vals,
            linewidth=0.5,
            alpha=0.25,
        )

    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=logmass,
        s=35,
        alpha=0.9,
    )

    ax.set_title(title)
    ax.set_xlabel("Raw X position [h$^{-1}$ Mpc]")
    ax.set_ylabel("Raw Y position [h$^{-1}$ Mpc]")
    ax.set_zlabel("Raw Z position [h$^{-1}$ Mpc]")

    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.1)
    colorbar.set_label("log10(Mvir)")

    ax.text2D(
        0.02,
        0.95,
        f"Nodes: {len(positions)}\nEdges drawn: {len(draw_sources)} / {num_edges}",
        transform=ax.transAxes,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_3d_graph_interactive(
    positions: np.ndarray,
    logmass: np.ndarray,
    edge_sources: np.ndarray,
    edge_targets: np.ndarray,
    output_path: Path,
    title: str,
    max_edges_to_draw: int = 1200,
) -> bool:
    """
    Save an interactive 3D graph as HTML using Plotly.

    Returns:
        True if saved successfully.
        False if Plotly is unavailable.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("⚠️ Plotly is not installed. Skipping interactive 3D HTML plot.")
        print("   Install with: pip install plotly")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_edges = len(edge_sources)

    if num_edges > max_edges_to_draw:
        rng = np.random.default_rng(42)
        chosen = rng.choice(num_edges, size=max_edges_to_draw, replace=False)
        draw_sources = edge_sources[chosen]
        draw_targets = edge_targets[chosen]
    else:
        draw_sources = edge_sources
        draw_targets = edge_targets

    edge_x = []
    edge_y = []
    edge_z = []

    for src, dst in zip(draw_sources, draw_targets):
        edge_x.extend([positions[src, 0], positions[dst, 0], None])
        edge_y.extend([positions[src, 1], positions[dst, 1], None])
        edge_z.extend([positions[src, 2], positions[dst, 2], None])

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(width=2),
        opacity=0.25,
        name="Graph edges",
        hoverinfo="none",
    )

    node_text = [
        f"Node {i}<br>"
        f"X={positions[i, 0]:.4f}<br>"
        f"Y={positions[i, 1]:.4f}<br>"
        f"Z={positions[i, 2]:.4f}<br>"
        f"log10(Mvir)={logmass[i]:.4f}"
        for i in range(len(positions))
    ]

    node_trace = go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        marker=dict(
            size=5,
            color=logmass,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="log10(Mvir)"),
        ),
        text=node_text,
        hoverinfo="text",
        name="Halos",
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Raw X [h^-1 Mpc]",
            yaxis_title="Raw Y [h^-1 Mpc]",
            zaxis_title="Raw Z [h^-1 Mpc]",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        showlegend=True,
    )

    fig.write_html(output_path)

    return True


def plot_degree_distribution(
    A: torch.Tensor,
    output_path: Path,
    title: str,
) -> None:
    """
    Plot graph degree distribution.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    A_np = tensor_to_numpy(A)
    degree = A_np.sum(axis=1)

    plt.figure(figsize=(10, 6))
    plt.hist(degree, bins=20)
    plt.title(title)
    plt.xlabel("Node degree")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_position_projection(
    positions: np.ndarray,
    logmass: np.ndarray,
    output_path: Path,
    title: str,
    axis_a: int,
    axis_b: int,
    label_a: str,
    label_b: str,
) -> None:
    """
    Plot one 2D projection of the 3D halo distribution.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(
        positions[:, axis_a],
        positions[:, axis_b],
        c=logmass,
        s=35,
        alpha=0.9,
    )

    plt.title(title)
    plt.xlabel(f"Raw {label_a} position [h$^{{-1}}$ Mpc]")
    plt.ylabel(f"Raw {label_b} position [h$^{{-1}}$ Mpc]")
    plt.colorbar(scatter, label="log10(Mvir)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_logmass_histogram(
    logmass: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """
    Plot histogram of selected log10(Mvir).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(logmass, bins=20)
    plt.title(title)
    plt.xlabel("log10(Mvir)")
    plt.ylabel("Number of selected halos")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_edge_distance_distribution(
    edge_distances: np.ndarray,
    output_path: Path,
    title: str,
    periodic_boundary: bool,
) -> None:
    """
    Plot histogram of graph edge distances.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    if len(edge_distances) > 0:
        plt.hist(edge_distances, bins=30)

    distance_mode = "periodic minimum-image" if periodic_boundary else "ordinary Euclidean"

    plt.title(title)
    plt.xlabel(f"Edge distance [{distance_mode}, h$^{{-1}}$ Mpc]")
    plt.ylabel("Number of edges")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_adjacency_matrix(
    A: torch.Tensor,
    output_path: Path,
    title: str,
) -> None:
    """
    Plot adjacency matrix heatmap/image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    A_np = tensor_to_numpy(A)

    plt.figure(figsize=(8, 8))
    plt.imshow(A_np, interpolation="nearest", aspect="equal")
    plt.title(title)
    plt.xlabel("Node index")
    plt.ylabel("Node index")
    plt.colorbar(label="A[i, j]")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


# ============================================================
# Summary and checklist
# ============================================================

def build_plot_summary(
    snapshot: dict,
    positions: np.ndarray,
    logmass: np.ndarray,
    A_valid: torch.Tensor,
    edge_sources: np.ndarray,
    edge_targets: np.ndarray,
    edge_distances: np.ndarray,
    node_selection_summary: Dict[str, object],
) -> Dict[str, object]:
    """
    Build a JSON-serializable summary of graph and visualization diagnostics.
    """
    A_np = tensor_to_numpy(A_valid)
    degree = A_np.sum(axis=1)

    symmetry_error = float(np.abs(A_np - A_np.T).sum())
    is_symmetric = symmetry_error == 0.0

    summary = {
        "preprocessing_version": snapshot.get("preprocessing_version"),
        "expected_preprocessing_version": PREPROCESSING_VERSION,
        "snapshot_path": snapshot.get("path"),
        "snapshot_value": snapshot.get("snapshot_value"),
        "feature_names": snapshot.get("feature_names"),
        "feature_columns": snapshot.get("feature_columns"),
        "mass_column": snapshot.get("mass_column"),
        "mass_feature": snapshot.get("mass_feature"),
        "node_selection": snapshot.get("node_selection"),
        "normalization": snapshot.get("normalization"),
        "graph_mode": snapshot.get("graph_mode"),
        "graph_positions": snapshot.get("graph_positions"),
        "k": snapshot.get("k"),
        "radius": snapshot.get("radius"),
        "periodic_boundary": snapshot.get("periodic_boundary"),
        "periodic_boundary_knn": snapshot.get("periodic_boundary_knn"),
        "box_size": snapshot.get("box_size"),
        "num_valid_nodes_plotted": int(len(positions)),
        "num_undirected_edges": int(len(edge_sources)),
        "adjacency_shape": list(A_valid.shape),
        "adjacency_is_symmetric": bool(is_symmetric),
        "adjacency_symmetry_error": symmetry_error,
        "degree_min": float(degree.min()) if len(degree) else None,
        "degree_max": float(degree.max()) if len(degree) else None,
        "degree_mean": float(degree.mean()) if len(degree) else None,
        "degree_std": float(degree.std()) if len(degree) else None,
        "isolated_nodes": int((degree == 0).sum()) if len(degree) else None,
        "raw_position_x_min": float(positions[:, 0].min()) if len(positions) else None,
        "raw_position_x_max": float(positions[:, 0].max()) if len(positions) else None,
        "raw_position_y_min": float(positions[:, 1].min()) if len(positions) else None,
        "raw_position_y_max": float(positions[:, 1].max()) if len(positions) else None,
        "raw_position_z_min": float(positions[:, 2].min()) if len(positions) else None,
        "raw_position_z_max": float(positions[:, 2].max()) if len(positions) else None,
        "log10_mvir_min": float(logmass.min()) if len(logmass) else None,
        "log10_mvir_max": float(logmass.max()) if len(logmass) else None,
        "log10_mvir_mean": float(logmass.mean()) if len(logmass) else None,
        "edge_distance_min": float(edge_distances.min()) if len(edge_distances) else None,
        "edge_distance_max": float(edge_distances.max()) if len(edge_distances) else None,
        "edge_distance_mean": float(edge_distances.mean()) if len(edge_distances) else None,
        "edge_distance_std": float(edge_distances.std()) if len(edge_distances) else None,
        "node_selection_summary": node_selection_summary,
    }

    return summary


def print_graph_summary(
    snapshot: dict,
    positions: np.ndarray,
    logmass: np.ndarray,
    A_valid: torch.Tensor,
    edge_sources: np.ndarray,
    edge_targets: np.ndarray,
    edge_distances: np.ndarray,
) -> None:
    """
    Print a clean terminal summary.
    """
    A = snapshot["A"]
    X = snapshot["X"]
    mask = snapshot["mask"]

    degree_all = A.sum(dim=1)
    degree_valid = A_valid.sum(dim=1)

    print("=" * 90)
    print("CAMELS-SIMBA ONE-GRAPH VISUAL DIAGNOSTICS")
    print("=" * 90)

    print("\nSnapshot metadata")
    print("-" * 90)
    print(f"Preprocessing version: {snapshot.get('preprocessing_version')}")
    print(f"Expected version:      {PREPROCESSING_VERSION}")
    print(f"Snapshot path:         {snapshot.get('path')}")
    print(f"Snapshot value:        {snapshot.get('snapshot_value')}")
    print(f"Feature names:         {snapshot.get('feature_names')}")
    print(f"Feature columns:       {snapshot.get('feature_columns')}")
    print(f"Position columns:      {snapshot.get('position_columns')}")
    print(f"Mass column:           {snapshot.get('mass_column')}")
    print(f"Mass feature:          {snapshot.get('mass_feature')}")
    print(f"Node selection:        {snapshot.get('node_selection')}")
    print(f"Normalization:         {snapshot.get('normalization')}")
    print(f"Graph mode:            {snapshot.get('graph_mode')}")
    print(f"Graph positions:       {snapshot.get('graph_positions')}")
    print(f"k:                     {snapshot.get('k')}")
    print(f"Radius:                {snapshot.get('radius')}")
    print(f"Periodic boundary:     {snapshot.get('periodic_boundary')}")
    print(f"Periodic boundary kNN: {snapshot.get('periodic_boundary_knn')}")
    print(f"Box size:              {snapshot.get('box_size')}")

    print("\nTensor shapes")
    print("-" * 90)
    print(f"A shape:               {tuple(A.shape)}")
    print(f"X shape:               {tuple(X.shape)}")
    print(f"mask shape:            {tuple(mask.shape)}")

    print("\nGraph statistics")
    print("-" * 90)
    print(f"Requested nodes:        {A.shape[0]}")
    print(f"Valid nodes plotted:    {len(positions)}")
    print(f"Undirected edges:       {len(edge_sources)}")
    print(f"All-degree min:         {degree_all.min().item():.4f}")
    print(f"All-degree max:         {degree_all.max().item():.4f}")
    print(f"All-degree mean:        {degree_all.float().mean().item():.4f}")
    print(f"Valid-degree min:       {degree_valid.min().item():.4f}")
    print(f"Valid-degree max:       {degree_valid.max().item():.4f}")
    print(f"Valid-degree mean:      {degree_valid.float().mean().item():.4f}")
    print(f"Valid-degree std:       {degree_valid.float().std(unbiased=False).item():.4f}")

    print("\nRaw physical position ranges used for plotting")
    print("-" * 90)
    print(f"X min/max:              {positions[:, 0].min():.6g} / {positions[:, 0].max():.6g}")
    print(f"Y min/max:              {positions[:, 1].min():.6g} / {positions[:, 1].max():.6g}")
    print(f"Z min/max:              {positions[:, 2].min():.6g} / {positions[:, 2].max():.6g}")

    print("\nlog10(Mvir) range used for coloring")
    print("-" * 90)
    print(f"log10(Mvir) min:        {logmass.min():.6g}")
    print(f"log10(Mvir) max:        {logmass.max():.6g}")
    print(f"log10(Mvir) mean:       {logmass.mean():.6g}")
    print(f"log10(Mvir) std:        {logmass.std():.6g}")

    print("\nEdge-distance summary")
    print("-" * 90)
    if len(edge_distances) > 0:
        print(f"Distance min:           {edge_distances.min():.6g}")
        print(f"Distance max:           {edge_distances.max():.6g}")
        print(f"Distance mean:          {edge_distances.mean():.6g}")
        print(f"Distance std:           {edge_distances.std():.6g}")
    else:
        print("No edge distances available because graph has no edges.")

    print("\nNormalized ML feature range")
    print("-" * 90)
    print(f"X tensor min:           {X.min().item():.6g}")
    print(f"X tensor max:           {X.max().item():.6g}")

    print("\nChecklist")
    print("-" * 90)

    if snapshot.get("preprocessing_version") == PREPROCESSING_VERSION:
        print("✅ Preprocessing version matches current official version.")
    else:
        print("⚠️ Preprocessing version mismatch.")

    if snapshot.get("mass_feature") == "log10_Mvir":
        print("✅ Mass feature is log10_Mvir.")
    else:
        print("⚠️ Mass feature is not log10_Mvir.")

    if snapshot.get("node_selection") == "top_num_nodes_by_raw_Mvir_descending":
        print("✅ Node selection uses top halos by raw Mvir.")
    else:
        print("⚠️ Node selection rule is unexpected.")

    if snapshot.get("periodic_boundary") is True:
        print("✅ Periodic boundary-aware distance is enabled.")
    else:
        print("⚠️ Periodic boundary-aware distance is disabled.")

    if snapshot.get("graph_mode") == "knn" and snapshot.get("periodic_boundary_knn") is True:
        print("✅ Periodic boundary-aware kNN is enabled.")
    else:
        print("⚠️ Periodic boundary-aware kNN may not be enabled.")

    if torch.isnan(X).any():
        print("❌ NaN values found in X.")
    else:
        print("✅ No NaN values in X.")

    if torch.isinf(X).any():
        print("❌ Infinite values found in X.")
    else:
        print("✅ No infinite values in X.")

    if len(edge_sources) == 0:
        print("❌ No graph edges found.")
    else:
        print("✅ Graph edges exist.")

    isolated_nodes = int((degree_valid == 0).sum().item())
    if isolated_nodes > 0:
        print(f"⚠️ Isolated valid nodes found: {isolated_nodes}")
    else:
        print("✅ No isolated valid nodes.")

    if abs(float(snapshot.get("box_size", -1)) - float(DEFAULT_BOX_SIZE)) < 1e-6:
        print("✅ Box size matches CAMELS default 25.0 h^-1 Mpc.")
    else:
        print("⚠️ Box size differs from CAMELS default.")

    print("=" * 90)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visual diagnostics for one CAMELS-SIMBA halo graph."
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
        help="Number of nearest neighbors if graph_mode='knn'.",
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Connection radius if graph_mode='radius'.",
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
        required=True,
        help="Directory where plots and summaries will be saved.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used for processing. Usually cpu is enough for diagnostics.",
    )

    parser.add_argument(
        "--max_edges_to_draw",
        type=int,
        default=600,
        help="Maximum number of edges to draw in the static 3D plot.",
    )

    parser.add_argument(
        "--max_interactive_edges",
        type=int,
        default=1200,
        help="Maximum number of edges to include in the interactive HTML plot.",
    )

    args = parser.parse_args()

    snapshot_path = Path(args.snapshot_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    raw_positions = extract_raw_selected_positions(
        snapshot_path=snapshot_path,
        num_nodes=args.num_nodes,
    )

    raw_logmass = extract_raw_selected_logmass(
        snapshot_path=snapshot_path,
        num_nodes=args.num_nodes,
    )

    raw_positions, raw_logmass, A_valid = filter_valid_nodes(
        positions=raw_positions,
        logmass=raw_logmass,
        A=snapshot["A"],
        mask=snapshot["mask"],
    )

    edge_sources, edge_targets = extract_edges_from_adjacency(A_valid)

    edge_distances = compute_edge_distances(
        positions=raw_positions,
        edge_sources=edge_sources,
        edge_targets=edge_targets,
        periodic_boundary=args.periodic_boundary,
        box_size=args.box_size,
    )

    node_selection_summary = build_node_selection_summary(
        snapshot_path=snapshot_path,
        num_nodes=args.num_nodes,
    )

    plot_summary = build_plot_summary(
        snapshot=snapshot,
        positions=raw_positions,
        logmass=raw_logmass,
        A_valid=A_valid,
        edge_sources=edge_sources,
        edge_targets=edge_targets,
        edge_distances=edge_distances,
        node_selection_summary=node_selection_summary,
    )

    print_graph_summary(
        snapshot=snapshot,
        positions=raw_positions,
        logmass=raw_logmass,
        A_valid=A_valid,
        edge_sources=edge_sources,
        edge_targets=edge_targets,
        edge_distances=edge_distances,
    )

    snapshot_value = parse_snapshot_value(snapshot_path)

    title_base = (
        f"CAMELS-SIMBA Halo Graph | "
        f"LH snapshot a={snapshot_value:.5f} | "
        f"{args.normalization} | "
        f"{args.graph_mode} | "
        f"k={args.k} | "
        f"periodic={args.periodic_boundary}"
    )

    # 01. Static 3D graph
    plot_3d_graph_static(
        positions=raw_positions,
        logmass=raw_logmass,
        edge_sources=edge_sources,
        edge_targets=edge_targets,
        output_path=output_dir / "01_graph_3d_static.png",
        title=f"3D Halo Graph\n{title_base}",
        max_edges_to_draw=args.max_edges_to_draw,
    )

    # 02. Interactive 3D graph
    interactive_saved = plot_3d_graph_interactive(
        positions=raw_positions,
        logmass=raw_logmass,
        edge_sources=edge_sources,
        edge_targets=edge_targets,
        output_path=output_dir / "02_graph_3d_interactive.html",
        title=f"Interactive 3D Halo Graph | {title_base}",
        max_edges_to_draw=args.max_interactive_edges,
    )

    # 03. Degree distribution
    plot_degree_distribution(
        A=A_valid,
        output_path=output_dir / "03_degree_distribution.png",
        title=f"Degree Distribution\n{title_base}",
    )

    # 06-08. 2D projections
    plot_position_projection(
        positions=raw_positions,
        logmass=raw_logmass,
        output_path=output_dir / "06_xy_projection.png",
        title=f"XY Halo Projection\n{title_base}",
        axis_a=0,
        axis_b=1,
        label_a="X",
        label_b="Y",
    )

    plot_position_projection(
        positions=raw_positions,
        logmass=raw_logmass,
        output_path=output_dir / "07_xz_projection.png",
        title=f"XZ Halo Projection\n{title_base}",
        axis_a=0,
        axis_b=2,
        label_a="X",
        label_b="Z",
    )

    plot_position_projection(
        positions=raw_positions,
        logmass=raw_logmass,
        output_path=output_dir / "08_yz_projection.png",
        title=f"YZ Halo Projection\n{title_base}",
        axis_a=1,
        axis_b=2,
        label_a="Y",
        label_b="Z",
    )

    # 09. log10 mass histogram
    plot_logmass_histogram(
        logmass=raw_logmass,
        output_path=output_dir / "09_log10_mass_histogram.png",
        title=f"Selected Halo log10(Mvir) Distribution\n{title_base}",
    )

    # 11. Edge distance distribution
    plot_edge_distance_distribution(
        edge_distances=edge_distances,
        output_path=output_dir / "11_edge_distance_distribution.png",
        title=f"Graph Edge Distance Distribution\n{title_base}",
        periodic_boundary=args.periodic_boundary,
    )

    # 12. Adjacency matrix
    plot_adjacency_matrix(
        A=A_valid,
        output_path=output_dir / "12_adjacency_matrix.png",
        title=f"Adjacency Matrix\n{title_base}",
    )

    # JSON summary
    save_json(
        path=output_dir / "05_graph_visual_summary.json",
        data=plot_summary,
    )

    print("\nFiles saved:")
    print("-" * 90)
    print(f"01 static 3D graph:          {output_dir / '01_graph_3d_static.png'}")

    if interactive_saved:
        print(f"02 interactive 3D graph:     {output_dir / '02_graph_3d_interactive.html'}")
    else:
        print("02 interactive 3D graph:     skipped because plotly is not installed")

    print(f"03 degree distribution:      {output_dir / '03_degree_distribution.png'}")
    print(f"05 visual summary JSON:      {output_dir / '05_graph_visual_summary.json'}")
    print(f"06 XY projection:            {output_dir / '06_xy_projection.png'}")
    print(f"07 XZ projection:            {output_dir / '07_xz_projection.png'}")
    print(f"08 YZ projection:            {output_dir / '08_yz_projection.png'}")
    print(f"09 log10 mass histogram:     {output_dir / '09_log10_mass_histogram.png'}")
    print(f"11 edge distance histogram:  {output_dir / '11_edge_distance_distribution.png'}")
    print(f"12 adjacency matrix:         {output_dir / '12_adjacency_matrix.png'}")
    print("\nOne-graph visual diagnostics complete.")
    print("=" * 90)


if __name__ == "__main__":
    main()