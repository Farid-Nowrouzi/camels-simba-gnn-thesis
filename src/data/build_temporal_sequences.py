from __future__ import annotations

"""
build_temporal_sequences.py

Build temporal CAMELS-SIMBA graph-sequence datasets from raw halo catalog snapshots.

Purpose
-------
This script creates one temporal graph sequence per universe.

Each universe contains multiple snapshot graphs, usually 5 snapshots:

    LH_0:
        snapshot 1 graph
        snapshot 2 graph
        snapshot 3 graph
        snapshot 4 graph
        snapshot 5 graph

This temporal dataset is used for temporal graph neural networks such as
EvolveGCN-H.

Preprocessing provenance
------------------------
The saved label is generated from the effective CLI configuration.

Scientific definition
---------------------
    Universe       = one simulation sample
    Snapshot       = one graph at one cosmic time
    Nodes          = halos
    Node selection = top num_nodes halos by raw Mvir
    Node features  = [log10_Mvir, X, Y, Z, VX, VY, VZ]
    Edges          = periodic boundary-aware k-nearest-neighbor graph
    Target         = Omega_m

Saved dataset format
--------------------
dataset = {
    "LH_0": {
        "A_list": [Tensor(num_nodes, num_nodes), ...],
        "Nodes_list": [Tensor(num_nodes, 7), ...],
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

Example command
---------------
python -m src.data.build_temporal_sequences \\
  --raw_dir data/raw/CAMELS_SIMBA_100U \\
  --output_path data/processed/temporal_100u_minmax/camels_100u_temporal_logmass_minmax_top100_periodic_knn.pt \\
  --num_universes 100 \\
  --num_snapshots 5 \\
  --num_nodes 100 \\
  --normalization minmax \\
  --graph_mode knn \\
  --k 8 \\
  --periodic_boundary \\
  --box_size 25.0 \\
  --targets_csv outputs/target_inspection_100u.csv \\
  --device cpu
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from src.data.camels_graph_utils import (
    DEFAULT_BOX_SIZE,
    FEATURE_COLUMNS,
    FEATURE_NAMES,
    MASS_COLUMN,
    POSITION_COLUMNS,
    VELOCITY_COLUMNS,
    build_universe_sequence,
    choose_snapshot_files,
    find_universe_files,
    GRAPH_STORAGE_DENSE,
    GRAPH_STORAGE_SPARSE,
    SPARSE_SCHEMA_VERSION,
    preprocessing_version_for_config,
)
from src.data.atomic_dataset import atomic_write_sparse_dataset
from src.data.source_manifest import (
    SOURCE_MANIFEST_POLICY_FULL,
    SOURCE_MANIFEST_POLICY_LEGACY,
    build_full_source_manifest,
    verify_full_source_manifest,
)


# ============================================================
# General utilities
# ============================================================

def parse_universe_id(value: object) -> int:
    """
    Convert universe identifiers into integer IDs.

    Supported examples:
        0       -> 0
        "0"     -> 0
        "LH_0"  -> 0
        "lh_0"  -> 0
    """
    value_str = str(value).strip()

    if value_str.lower().startswith("lh_"):
        value_str = value_str.split("_", 1)[1]

    return int(value_str)


def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """
    Find a column using flexible candidate names.

    This avoids breaking when CSV headers are slightly different.
    """
    normalized_to_original = {
        col.strip().lower(): col for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized_to_original:
            return normalized_to_original[key]

    raise ValueError(
        f"None of the expected columns were found.\n"
        f"Expected one of: {candidates}\n"
        f"Actual columns: {list(df.columns)}"
    )


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    """
    Save metadata/configuration as JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def tensor_to_cpu(value: Any) -> Any:
    """
    Move tensors recursively to CPU before saving.

    This keeps saved .pt datasets portable and prevents GPU-specific loading issues.
    """
    if torch.is_tensor(value):
        return value.detach().cpu()

    if isinstance(value, dict):
        return {key: tensor_to_cpu(item) for key, item in value.items()}

    if isinstance(value, list):
        return [tensor_to_cpu(item) for item in value]

    if isinstance(value, tuple):
        return tuple(tensor_to_cpu(item) for item in value)

    return value


def safe_float(value: Any) -> float:
    """
    Convert tensor/scalar values into Python float.
    """
    if torch.is_tensor(value):
        return float(value.detach().cpu().view(-1)[0].item())

    return float(value)


# ============================================================
# Target loading
# ============================================================

def load_targets_csv(path: str | Path) -> Dict[int, float]:
    """
    Load Omega_m labels from a CSV file.

    Accepted universe columns:
        universe_id
        Universe
        universe
        Universe_ID
        lh_id
        LH

    Accepted Omega_m columns:
        omega_m
        Omega_m
        Omega_M
        target
        Target
        omega_m_value

    Accepted universe ID values:
        0
        1
        2
        LH_0
        LH_1
        LH_2
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Target CSV not found: {path}")

    df = pd.read_csv(path)

    universe_col = find_column(
        df,
        candidates=[
            "universe_id",
            "Universe",
            "universe",
            "Universe_ID",
            "lh_id",
            "LH",
        ],
    )

    omega_col = find_column(
        df,
        candidates=[
            "omega_m",
            "Omega_m",
            "Omega_M",
            "target",
            "Target",
            "omega_m_value",
        ],
    )

    targets: Dict[int, float] = {}

    for _, row in df.iterrows():
        universe_id = parse_universe_id(row[universe_col])
        omega_m = float(row[omega_col])
        targets[universe_id] = omega_m

    print(f"Loaded {len(targets)} Omega_m targets from: {path}")
    print(f"Universe column: {universe_col}")
    print(f"Omega_m column:  {omega_col}")

    if len(targets) == 0:
        raise ValueError(f"No targets were loaded from: {path}")

    return targets


# ============================================================
# Validation helpers
# ============================================================

def validate_common_sequence_metadata(
    universe_key: str,
    sequence: Dict[str, Any],
    normalization: str,
    graph_mode: str,
    periodic_boundary: bool,
    box_size: float,
    preprocessing_version: str,
) -> None:
    """
    Validate scientific metadata for one temporal graph sequence.
    """
    if sequence.get("preprocessing_version") != preprocessing_version:
        raise ValueError(
            f"{universe_key}: unexpected preprocessing version: "
            f"{sequence.get('preprocessing_version')}. "
            f"Expected: {preprocessing_version}"
        )

    if sequence.get("feature_names") != FEATURE_NAMES:
        raise ValueError(
            f"{universe_key}: unexpected feature names: "
            f"{sequence.get('feature_names')}. "
            f"Expected: {FEATURE_NAMES}"
        )

    if sequence.get("mass_feature") != "log10_Mvir":
        raise ValueError(
            f"{universe_key}: unexpected mass feature: "
            f"{sequence.get('mass_feature')}. Expected: log10_Mvir"
        )

    if sequence.get("node_selection") != "top_num_nodes_by_raw_Mvir_descending":
        raise ValueError(
            f"{universe_key}: unexpected node selection: "
            f"{sequence.get('node_selection')}"
        )

    if sequence.get("graph_positions") != "raw_physical_XYZ_before_feature_normalization":
        raise ValueError(
            f"{universe_key}: unexpected graph position rule: "
            f"{sequence.get('graph_positions')}"
        )

    if sequence.get("normalization") != normalization:
        raise ValueError(
            f"{universe_key}: normalization={sequence.get('normalization')} "
            f"!= expected {normalization}"
        )

    if sequence.get("graph_mode") != graph_mode:
        raise ValueError(
            f"{universe_key}: graph_mode={sequence.get('graph_mode')} "
            f"!= expected {graph_mode}"
        )

    if sequence.get("periodic_boundary") != periodic_boundary:
        raise ValueError(
            f"{universe_key}: periodic_boundary={sequence.get('periodic_boundary')} "
            f"!= expected {periodic_boundary}"
        )

    if graph_mode.lower() == "knn":
        expected_periodic_knn = bool(periodic_boundary)
        if sequence.get("periodic_boundary_knn") != expected_periodic_knn:
            raise ValueError(
                f"{universe_key}: periodic_boundary_knn="
                f"{sequence.get('periodic_boundary_knn')} "
                f"!= expected {expected_periodic_knn}"
            )

    if abs(float(sequence.get("box_size")) - float(box_size)) > 1e-6:
        raise ValueError(
            f"{universe_key}: box_size={sequence.get('box_size')} "
            f"!= expected {box_size}"
        )


def validate_temporal_sequence_tensors(
    universe_key: str,
    sequence: Dict[str, Any],
    num_snapshots: int,
    num_nodes: int,
    expected_features: int,
    normalization: str,
) -> None:
    """
    Validate shapes and tensor quality for one universe sequence.
    """
    sparse = sequence.get("graph_storage") == GRAPH_STORAGE_SPARSE
    graph_key = "edge_index_list" if sparse else "A_list"
    required_keys = [graph_key, "Nodes_list", "mask_list", "target", "snapshots"]

    for key in required_keys:
        if key not in sequence:
            raise ValueError(f"{universe_key}: missing required key: {key}")

    graph_list = sequence[graph_key]
    X_list = sequence["Nodes_list"]
    mask_list = sequence["mask_list"]
    snapshots = sequence["snapshots"]

    if len(graph_list) != num_snapshots:
        raise ValueError(
            f"{universe_key}: expected {num_snapshots} adjacency matrices, "
            f"got {len(graph_list)}"
        )

    if len(X_list) != num_snapshots:
        raise ValueError(
            f"{universe_key}: expected {num_snapshots} node-feature matrices, "
            f"got {len(X_list)}"
        )

    if len(mask_list) != num_snapshots:
        raise ValueError(
            f"{universe_key}: expected {num_snapshots} masks, "
            f"got {len(mask_list)}"
        )

    if len(snapshots) != num_snapshots:
        raise ValueError(
            f"{universe_key}: expected {num_snapshots} snapshot metadata entries, "
            f"got {len(snapshots)}"
        )

    target_value = safe_float(sequence["target"])
    if not (0.0 < target_value < 1.0):
        raise ValueError(
            f"{universe_key}: Omega_m target looks invalid: {target_value}"
        )

    snapshot_values = []

    for snapshot_index, (graph, X, mask, snapshot_meta) in enumerate(
        zip(graph_list, X_list, mask_list, snapshots)
    ):
        if not torch.is_tensor(graph):
            raise TypeError(
                f"{universe_key}, snapshot {snapshot_index}: A is not a tensor."
            )

        if not torch.is_tensor(X):
            raise TypeError(
                f"{universe_key}, snapshot {snapshot_index}: X is not a tensor."
            )

        if not torch.is_tensor(mask):
            raise TypeError(
                f"{universe_key}, snapshot {snapshot_index}: mask is not a tensor."
            )

        if sparse:
            if graph.dtype != torch.long or graph.ndim != 2 or graph.shape[0] != 2:
                raise ValueError(f"{universe_key}, snapshot {snapshot_index}: invalid edge_index")
        elif tuple(graph.shape) != (num_nodes, num_nodes):
            raise ValueError(f"{universe_key}, snapshot {snapshot_index}: invalid dense A shape")

        if tuple(X.shape) != (num_nodes, expected_features):
            raise ValueError(
                f"{universe_key}, snapshot {snapshot_index}: "
                f"X shape {tuple(X.shape)} != ({num_nodes}, {expected_features})"
            )

        if tuple(mask.shape) != (num_nodes, 1):
            raise ValueError(
                f"{universe_key}, snapshot {snapshot_index}: "
                f"mask shape {tuple(mask.shape)} != ({num_nodes}, 1)"
            )

        if torch.isnan(graph.float()).any():
            raise ValueError(
                f"{universe_key}, snapshot {snapshot_index}: A contains NaN values."
            )

        if torch.isinf(graph.float()).any():
            raise ValueError(
                f"{universe_key}, snapshot {snapshot_index}: A contains Inf values."
            )

        if torch.isnan(X.float()).any():
            raise ValueError(
                f"{universe_key}, snapshot {snapshot_index}: X contains NaN values."
            )

        if torch.isinf(X.float()).any():
            raise ValueError(
                f"{universe_key}, snapshot {snapshot_index}: X contains Inf values."
            )

        if torch.isnan(mask.float()).any():
            raise ValueError(
                f"{universe_key}, snapshot {snapshot_index}: mask contains NaN values."
            )

        if torch.isinf(mask.float()).any():
            raise ValueError(
                f"{universe_key}, snapshot {snapshot_index}: mask contains Inf values."
            )

        real_nodes = int(mask.sum().item())
        if real_nodes <= 0:
            raise ValueError(
                f"{universe_key}, snapshot {snapshot_index}: mask indicates zero real nodes."
            )

        nonzero_edges = graph.shape[1] if sparse else int((graph > 0).sum().item())
        if nonzero_edges == 0:
            raise ValueError(
                f"{universe_key}, snapshot {snapshot_index}: graph has no edges."
            )

        if sparse:
            pairs = {tuple(pair) for pair in graph.T.tolist()}
            if len(pairs) != graph.shape[1]:
                raise ValueError(f"{universe_key}, snapshot {snapshot_index}: duplicate sparse edges")
            if any(source == target for source, target in pairs):
                raise ValueError(f"{universe_key}, snapshot {snapshot_index}: sparse self-loop")
            if any((target, source) not in pairs for source, target in pairs):
                raise ValueError(f"{universe_key}, snapshot {snapshot_index}: asymmetric sparse edges")
            if graph.numel() and (int(graph.min()) < 0 or int(graph.max()) >= real_nodes):
                raise ValueError(f"{universe_key}, snapshot {snapshot_index}: edge reaches padding")
        else:
            symmetry_error = float(torch.abs(graph.float() - graph.float().T).sum().item())
            if symmetry_error != 0.0:
                raise ValueError(f"{universe_key}, snapshot {snapshot_index}: adjacency is not symmetric")
            if int((torch.diag(graph.float()) > 0).sum().item()) != 0:
                raise ValueError(f"{universe_key}, snapshot {snapshot_index}: adjacency has self-loops")

        if normalization == "minmax":
            x_min = float(X.min().item())
            x_max = float(X.max().item())
            tolerance = 1e-5

            if x_min < -tolerance or x_max > 1.0 + tolerance:
                raise ValueError(
                    f"{universe_key}, snapshot {snapshot_index}: "
                    f"normalization=minmax but X range is [{x_min}, {x_max}], "
                    "expected approximately [0, 1]."
                )

        if not isinstance(snapshot_meta, dict):
            raise TypeError(
                f"{universe_key}, snapshot {snapshot_index}: snapshot metadata is not dict."
            )

        snapshot_value = snapshot_meta.get("snapshot_value")
        if snapshot_value is None:
            raise ValueError(
                f"{universe_key}, snapshot {snapshot_index}: missing snapshot_value."
            )

        snapshot_values.append(float(snapshot_value))

    if snapshot_values != sorted(snapshot_values):
        raise ValueError(
            f"{universe_key}: snapshot values are not sorted increasingly: "
            f"{snapshot_values}"
        )


def validate_full_temporal_sample(
    universe_key: str,
    sequence: Dict[str, Any],
    num_snapshots: int,
    num_nodes: int,
    normalization: str,
    graph_mode: str,
    periodic_boundary: bool,
    box_size: float,
    preprocessing_version: str,
) -> None:
    """
    Run all validations for one temporal sample.
    """
    validate_common_sequence_metadata(
        universe_key=universe_key,
        sequence=sequence,
        normalization=normalization,
        graph_mode=graph_mode,
        periodic_boundary=periodic_boundary,
        box_size=box_size,
        preprocessing_version=preprocessing_version,
    )

    validate_temporal_sequence_tensors(
        universe_key=universe_key,
        sequence=sequence,
        num_snapshots=num_snapshots,
        num_nodes=num_nodes,
        expected_features=len(FEATURE_NAMES),
        normalization=normalization,
    )


# ============================================================
# Main dataset builder
# ============================================================

def build_temporal_dataset(
    raw_dir: str | Path,
    output_path: str | Path,
    num_universes: int,
    num_snapshots: int = 5,
    num_nodes: int = 100,
    normalization: str = "minmax",
    graph_mode: str = "knn",
    k: int = 8,
    radius: Optional[float] = None,
    periodic_boundary: bool = True,
    box_size: float = DEFAULT_BOX_SIZE,
    targets_csv: Optional[str | Path] = None,
    dummy_target: Optional[float] = None,
    device: str = "cpu",
    allow_partial: bool = False,
    graph_storage: str = GRAPH_STORAGE_DENSE,
    overwrite: bool = False,
    force_unsafe_dense: bool = False,
    source_manifest_policy: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build temporal graph sequences from raw CAMELS-SIMBA halo catalogs.

    Each universe becomes one sample containing a sequence of snapshot graphs.

    Parameters
    ----------
    raw_dir:
        Folder containing raw CAMELS-SIMBA halo catalog files.

    output_path:
        Destination .pt file.

    num_universes:
        Number of universes to process, starting from LH_0.

    num_snapshots:
        Number of snapshots per universe.

    num_nodes:
        Number of halo nodes per graph.

    normalization:
        Feature normalization. Official pipeline uses "minmax".

    graph_mode:
        Graph construction mode. Official pipeline uses "knn".

    k:
        Number of neighbors for kNN graph.

    radius:
        Radius threshold if graph_mode="radius".

    periodic_boundary:
        Whether to use periodic boundary-aware distances.

    box_size:
        CAMELS simulation box size.

    targets_csv:
        CSV containing real Omega_m targets.

    dummy_target:
        Dummy target only for testing.

    device:
        Device used while building tensors.

    allow_partial:
        If False, the build fails if any universe fails.
        For thesis experiments, keep this False.
    """
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    if num_universes <= 0:
        raise ValueError("num_universes must be positive.")

    if num_snapshots <= 0:
        raise ValueError("num_snapshots must be positive.")

    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    if graph_storage not in {GRAPH_STORAGE_DENSE, GRAPH_STORAGE_SPARSE}:
        raise ValueError("graph_storage must be dense_adjacency or sparse_edge_index.")
    resolved_manifest_policy = source_manifest_policy or (
        SOURCE_MANIFEST_POLICY_FULL
        if graph_storage == GRAPH_STORAGE_SPARSE
        else SOURCE_MANIFEST_POLICY_LEGACY
    )
    if graph_storage == GRAPH_STORAGE_SPARSE and resolved_manifest_policy != SOURCE_MANIFEST_POLICY_FULL:
        raise ValueError(
            "New sparse builds require source_manifest_policy=full_sha256; "
            "legacy/stat-only provenance is not accepted."
        )
    if graph_storage == GRAPH_STORAGE_DENSE and num_nodes > 512 and not force_unsafe_dense:
        raise ValueError(
            "Dense graph storage above 512 nodes is blocked by the resource guard; "
            "use sparse_edge_index or explicitly pass --force_unsafe_dense."
        )
    estimated_sparse_bytes = num_universes * num_snapshots * (num_nodes * 7 * 4 + 2 * num_nodes * k * 2 * 8)
    free_bytes = shutil.disk_usage(output_path.parent if output_path.parent.exists() else raw_dir).free
    if graph_storage == GRAPH_STORAGE_SPARSE and free_bytes < estimated_sparse_bytes * 2 + 1024 * 1024:
        raise OSError("Insufficient disk space for sparse temporary output plus safety margin.")

    if graph_mode == "knn" and k <= 0:
        raise ValueError("k must be positive when graph_mode='knn'.")

    if graph_mode == "radius" and radius is None:
        raise ValueError("radius must be provided when graph_mode='radius'.")

    if periodic_boundary and (box_size is None or box_size <= 0):
        raise ValueError("box_size must be positive when periodic_boundary=True.")

    if normalization != "minmax":
        print()
        print("WARNING:")
        print(f"  You selected normalization={normalization!r}.")
        print("  The official thesis preprocessing uses normalization='minmax'.")
        print("  Use non-minmax only for ablation/debugging.")
        print()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_manifest = None
    if resolved_manifest_policy == SOURCE_MANIFEST_POLICY_FULL:
        planned_catalogue_paths = []
        for universe_id in range(num_universes):
            planned_catalogue_paths.extend(choose_snapshot_files(
                find_universe_files(raw_dir=raw_dir, universe_id=universe_id),
                num_snapshots=num_snapshots,
            ))
        source_manifest = build_full_source_manifest(
            catalogue_paths=planned_catalogue_paths,
            raw_root=raw_dir,
            target_path=targets_csv,
            target_root=Path(targets_csv).parent if targets_csv is not None else None,
            require_target=targets_csv is not None,
        )
        verify_full_source_manifest(
            source_manifest,
            require_target=targets_csv is not None,
        )

    if targets_csv is not None:
        targets = load_targets_csv(targets_csv)
        target_mode = "real_targets_csv"
    else:
        targets = {}
        target_mode = "dummy_target"

    if targets_csv is None and dummy_target is None:
        raise ValueError(
            "You must provide either --targets_csv for real Omega_m values "
            "or --dummy_target for testing only."
        )

    if dummy_target is not None and targets_csv is None:
        print()
        print("WARNING:")
        print("  You are using --dummy_target.")
        print("  This is acceptable only for testing the pipeline structure.")
        print("  Do not use dummy targets for scientific results.")
        print()

    dataset: Dict[str, Any] = {}
    failed_universes: List[tuple[str, str]] = []
    preprocessing_version = preprocessing_version_for_config(
        num_nodes=num_nodes,
        normalization=normalization,
        graph_mode=graph_mode,
        k=k,
        radius=radius,
        periodic_boundary=periodic_boundary,
        box_size=box_size,
        graph_storage=graph_storage,
    )

    print("=" * 90)
    print("CAMELS-SIMBA TEMPORAL SEQUENCE BUILDER")
    print("=" * 90)
    print(f"Dataset type:          temporal_graph_sequences")
    print(f"Preprocessing version: {preprocessing_version}")
    print(f"Raw directory:         {raw_dir}")
    print(f"Output path:           {output_path}")
    print(f"Number universes:      {num_universes}")
    print(f"Number snapshots:      {num_snapshots}")
    print(f"Number nodes:          {num_nodes}")
    print(f"Normalization:         {normalization}")
    print(f"Graph mode:            {graph_mode}")
    print(f"Graph storage:         {graph_storage}")
    print(f"Overwrite:             {overwrite}")
    print(f"Source manifest:       {resolved_manifest_policy}")
    print(f"k:                     {k}")
    print(f"Radius:                {radius}")
    print(f"Periodic boundary:     {periodic_boundary}")
    print(f"Periodic kNN:          {bool(periodic_boundary and graph_mode.lower() == 'knn')}")
    print(f"Box size:              {box_size}")
    print(f"Device:                {device}")
    print(f"Target mode:           {target_mode}")
    print(f"Allow partial build:   {allow_partial}")
    print()
    print("Feature definition:")
    print(f"  Feature names:       {FEATURE_NAMES}")
    print(f"  Feature columns:     {FEATURE_COLUMNS}")
    print(f"  Mass column:         {MASS_COLUMN}")
    print("  Mass feature:        log10_Mvir")
    print("  Node selection:      top_num_nodes_by_raw_Mvir_descending")
    print("  Graph positions:     raw_physical_XYZ_before_feature_normalization")
    print(f"  Position columns:    {POSITION_COLUMNS}")
    print(f"  Velocity columns:    {VELOCITY_COLUMNS}")
    print("=" * 90)

    for universe_id in range(num_universes):
        universe_key = f"LH_{universe_id}"

        if universe_id in targets:
            target = targets[universe_id]
        else:
            target = dummy_target

        if target is None:
            error_message = "Missing Omega_m target"
            failed_universes.append((universe_key, error_message))
            print(f"[FAILED] {universe_key}: {error_message}")
            continue

        try:
            sequence = build_universe_sequence(
                raw_dir=raw_dir,
                universe_id=universe_id,
                target=target,
                num_snapshots=num_snapshots,
                num_nodes=num_nodes,
                normalization=normalization,
                graph_mode=graph_mode,
                k=k,
                radius=radius,
                periodic_boundary=periodic_boundary,
                box_size=box_size,
                device=device,
                graph_storage=graph_storage,
            )

            validate_full_temporal_sample(
                universe_key=universe_key,
                sequence=sequence,
                num_snapshots=num_snapshots,
                num_nodes=num_nodes,
                normalization=normalization,
                graph_mode=graph_mode,
                periodic_boundary=periodic_boundary,
                box_size=box_size,
                preprocessing_version=preprocessing_version,
            )

            sequence = tensor_to_cpu(sequence)
            dataset[universe_key] = sequence

            first_a = sequence["edge_index_list"][0] if graph_storage == GRAPH_STORAGE_SPARSE else sequence["A_list"][0]
            first_x = sequence["Nodes_list"][0]
            first_mask = sequence["mask_list"][0]

            snapshot_values = [
                item["snapshot_value"] for item in sequence["snapshots"]
            ]

            target_value = safe_float(sequence["target"])

            print(
                f"[OK] {universe_key} | "
                f"snapshots={snapshot_values} | "
                f"A={tuple(first_a.shape)} | "
                f"X={tuple(first_x.shape)} | "
                f"mask={tuple(first_mask.shape)} | "
                f"target={target_value:.6f} | "
                f"version={sequence['preprocessing_version']} | "
                f"periodic={sequence['periodic_boundary']} | "
                f"box={sequence['box_size']}"
            )

        except Exception as exc:
            error_message = str(exc)
            failed_universes.append((universe_key, error_message))
            print(f"[FAILED] {universe_key}: {error_message}")

    if len(dataset) == 0:
        raise RuntimeError(
            "No universes were successfully processed. "
            "Check raw_dir, file names, column mapping, and targets."
        )

    if failed_universes and not allow_partial:
        failed_text = "\n".join(
            f"  - {universe_key}: {error_message}"
            for universe_key, error_message in failed_universes
        )

        raise RuntimeError(
            "Temporal dataset build failed for one or more universes.\n"
            "For thesis experiments, partial temporal datasets should not be saved.\n"
            "Fix the failed universes first, or use --allow_partial only for debugging.\n\n"
            f"Failed universes:\n{failed_text}"
        )

    if not allow_partial and len(dataset) != num_universes:
        raise RuntimeError(
            f"Expected {num_universes} successful universes, "
            f"but built {len(dataset)}."
        )

    target_values = [
        safe_float(sample["target"])
        for sample in dataset.values()
    ]

    metadata = {
        "dataset_type": "temporal_graph_sequences",
        "preprocessing_version": preprocessing_version,
        "raw_dir": str(raw_dir),
        "output_path": str(output_path),
        "num_universes_requested": num_universes,
        "num_universes_successful": len(dataset),
        "num_universes_failed": len(failed_universes),
        "allow_partial": allow_partial,
        "num_snapshots": num_snapshots,
        "num_nodes": num_nodes,
        "normalization": normalization,
        "graph_mode": graph_mode,
        "k": k,
        "radius": radius,
        "periodic_boundary": periodic_boundary,
        "periodic_boundary_knn": bool(periodic_boundary and graph_mode.lower() == "knn"),
        "box_size": box_size,
        "targets_csv": str(targets_csv) if targets_csv is not None else None,
        "target_mode": target_mode,
        "used_dummy_target": dummy_target is not None,
        "dummy_target": dummy_target,
        "device_used_for_building": device,
        "saved_device": "cpu",
        "feature_names": FEATURE_NAMES,
        "feature_columns": FEATURE_COLUMNS,
        "mass_column": MASS_COLUMN,
        "mass_feature": "log10_Mvir",
        "node_selection": "top_num_nodes_by_raw_Mvir_descending",
        "position_columns": POSITION_COLUMNS,
        "velocity_columns": VELOCITY_COLUMNS,
        "graph_positions": "raw_physical_XYZ_before_feature_normalization",
        "target_summary": {
            "count": len(target_values),
            "min": min(target_values),
            "max": max(target_values),
            "mean": sum(target_values) / len(target_values),
        },
        "failed_universes": failed_universes,
        "dataset_schema_version": SPARSE_SCHEMA_VERSION if graph_storage == GRAPH_STORAGE_SPARSE else "legacy_dense_v1",
        "graph_storage": graph_storage,
        "source_suite": "CAMELS-SIMBA",
        "ordered_universe_ids": list(dataset),
        "ordered_universe_ids_hash": hashlib.sha256("".join(f"{key}\n" for key in dataset).encode()).hexdigest(),
        "snapshot_ids": [item["snapshot_value"] for item in next(iter(dataset.values()))["snapshots"]],
        "top_n": num_nodes,
        "selection_method": "raw_Mvir_desc_stable_then_tie_key_asc",
        "tie_breaking_policy": "authoritative_halo_id_ascending_else_original_row_index",
        "target_normalization": "none",
        "edge_policy": "directed_k_choices_symmetrized_unique_no_builder_self_loops",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "pyg_version": None,
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_policy": resolved_manifest_policy,
    }

    config_material = {key: metadata[key] for key in (
        "num_universes_requested", "num_snapshots", "num_nodes", "normalization",
        "graph_mode", "k", "radius", "periodic_boundary", "box_size", "graph_storage",
    )}
    metadata["builder_config_hash"] = hashlib.sha256(
        json.dumps(config_material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    catalogue_paths = [
        snapshot["path"]
        for sample in dataset.values()
        for snapshot in sample["snapshots"]
    ]
    if resolved_manifest_policy == SOURCE_MANIFEST_POLICY_FULL:
        if source_manifest is None:
            raise RuntimeError("Full-SHA256 source manifest was not created before preprocessing.")
        verification = verify_full_source_manifest(
            source_manifest,
            require_target=targets_csv is not None,
        )
        target_entries = [
            entry for entry in source_manifest["entries"]
            if entry["source_role"] == "target_table"
        ]
        metadata.update({
            "source_manifest": source_manifest,
            "source_manifest_schema_version": source_manifest["schema_version"],
            "source_manifest_entry_count": source_manifest["entry_count"],
            "source_manifest_catalogue_count": source_manifest["catalogue_count"],
            "source_manifest_target_source_count": source_manifest["target_source_count"],
            "source_manifest_sha256": source_manifest["manifest_sha256"],
            "source_manifest_hash": source_manifest["manifest_sha256"],
            "source_root_identity": source_manifest["source_root_identity"],
            "source_manifest_verification": verification,
            "target_source_relative_path": target_entries[0]["relative_path"] if target_entries else None,
            "target_source_sha256": target_entries[0]["sha256"] if target_entries else None,
        })
    else:
        source_rows = []
        for source_value in catalogue_paths:
            source_path = Path(source_value)
            stat = source_path.stat()
            source_rows.append(f"{source_path}\t{stat.st_size}\t{stat.st_mtime_ns}\n")
        metadata["source_manifest_hash"] = hashlib.sha256("".join(source_rows).encode()).hexdigest()
        metadata["source_manifest_verification"] = {
            "verified": False,
            "verification_result": "legacy_unverified_stat_only",
        }
    selection_hashes = [
        meta["selection_hash_sha256"]
        for sample in dataset.values() for meta in sample["snapshots"]
    ]
    metadata["selected_halo_hash"] = hashlib.sha256(
        "".join(f"{value}\n" for value in selection_hashes).encode()
    ).hexdigest()
    metadata["raw_catalogue_roots"] = [str(raw_dir)]
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        metadata["git_commit"] = "unknown"

    node_counts = [int(meta["num_real_nodes"]) for sample in dataset.values() for meta in sample["snapshots"]]
    edge_counts = [
        int(edges.shape[1]) for sample in dataset.values() for edges in sample.get("edge_index_list", [])
    ]
    metadata["node_padding_statistics"] = {
        "real_min": min(node_counts), "real_max": max(node_counts),
        "real_mean": sum(node_counts) / len(node_counts),
        "padded_total": sum(num_nodes - count for count in node_counts),
    }
    metadata["edge_statistics"] = (
        {"directed_min": min(edge_counts), "directed_max": max(edge_counts), "directed_mean": sum(edge_counts) / len(edge_counts)}
        if edge_counts else {"representation": "dense_adjacency"}
    )

    if graph_storage == GRAPH_STORAGE_SPARSE:
        metadata = atomic_write_sparse_dataset(
            dataset, output_path, metadata,
            validate=lambda value: [
                validate_full_temporal_sample(
                    key, sample, num_snapshots, num_nodes, normalization,
                    graph_mode, periodic_boundary, box_size,
                    preprocessing_version,
                ) for key, sample in value.items()
            ],
            overwrite=overwrite,
        )
    else:
        torch.save(dataset, output_path)

    metadata_path = output_path.with_suffix(".metadata.json")
    if graph_storage == GRAPH_STORAGE_DENSE:
        save_json(metadata_path, metadata)

    print()
    print("=" * 90)
    print("TEMPORAL DATASET BUILD COMPLETE")
    print("=" * 90)
    print(f"Saved dataset:        {output_path}")
    print(f"Saved metadata:       {metadata_path}")
    print(f"Successful universes: {len(dataset)}")
    print(f"Failed universes:     {len(failed_universes)}")
    print(f"Preprocessing:        {preprocessing_version}")
    print(f"Feature names:        {FEATURE_NAMES}")
    print("Mass feature:         log10_Mvir")
    print("Node selection:       top_num_nodes_by_raw_Mvir_descending")
    print("Graph positions:      raw_physical_XYZ_before_feature_normalization")
    print(f"Normalization:        {normalization}")
    print(f"Graph mode:           {graph_mode}")
    print(f"Temporal snapshots:   {num_snapshots}")
    print(f"Periodic boundary:    {periodic_boundary}")
    print(f"Periodic kNN:         {bool(periodic_boundary and graph_mode.lower() == 'knn')}")
    print(f"Box size:             {box_size}")
    print(f"Saved device:         CPU")

    if failed_universes:
        print()
        print("Failed universe details:")
        for universe_key, error_message in failed_universes:
            print(f"  - {universe_key}: {error_message}")

    print("=" * 90)

    return dataset


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CAMELS-SIMBA temporal graph-sequence dataset."
    )

    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Raw CAMELS-SIMBA halo catalog directory.",
    )
    parser.add_argument(
        "--graph_storage", choices=[GRAPH_STORAGE_DENSE, GRAPH_STORAGE_SPARSE],
        default=GRAPH_STORAGE_DENSE,
        help="Legacy dense adjacency (default) or optional sparse edge_index schema.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing sparse output explicitly.")
    parser.add_argument("--force_unsafe_dense", action="store_true", help="Permit dense storage above the guarded Top-N threshold.")
    parser.add_argument(
        "--source_manifest_policy",
        choices=[SOURCE_MANIFEST_POLICY_FULL, SOURCE_MANIFEST_POLICY_LEGACY],
        default=None,
        help="Sparse builds require full_sha256; dense legacy builds default to legacy_stat_only.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output .pt file for the temporal graph-sequence dataset.",
    )

    parser.add_argument(
        "--num_universes",
        type=int,
        required=True,
        help="Number of universes to process, starting from LH_0.",
    )

    parser.add_argument(
        "--num_snapshots",
        type=int,
        default=5,
        help="Number of snapshots per universe. Default: 5.",
    )

    parser.add_argument(
        "--num_nodes",
        type=int,
        default=100,
        help="Number of halo nodes per graph. Default: 100.",
    )

    parser.add_argument(
        "--normalization",
        type=str,
        default="minmax",
        choices=["none", "minmax", "zscore"],
        help=(
            "Feature normalization mode. "
            "Official thesis pipeline uses 'minmax'."
        ),
    )

    parser.add_argument(
        "--graph_mode",
        type=str,
        default="knn",
        choices=["knn", "radius"],
        help="Graph construction mode. Official thesis pipeline uses 'knn'.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of neighbors for kNN graph. Default: 8.",
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Radius value if graph_mode='radius'.",
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
        help="Disable periodic boundary-aware distances. Use only for ablation/debugging.",
    )

    parser.add_argument(
        "--box_size",
        type=float,
        default=DEFAULT_BOX_SIZE,
        help="CAMELS simulation box size. Default: 25.0 h^-1 Mpc.",
    )

    parser.add_argument(
        "--targets_csv",
        type=str,
        default=None,
        help="CSV file containing real Omega_m targets.",
    )

    parser.add_argument(
        "--dummy_target",
        type=float,
        default=None,
        help="Dummy target for pipeline testing only. Do not use for scientific runs.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used while building tensors. Usually 'cpu'.",
    )

    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help=(
            "Allow saving a dataset even if some universes fail. "
            "Use only for debugging, not final thesis experiments."
        ),
    )

    args = parser.parse_args()

    build_temporal_dataset(
        raw_dir=args.raw_dir,
        output_path=args.output_path,
        num_universes=args.num_universes,
        num_snapshots=args.num_snapshots,
        num_nodes=args.num_nodes,
        normalization=args.normalization,
        graph_mode=args.graph_mode,
        k=args.k,
        radius=args.radius,
        periodic_boundary=args.periodic_boundary,
        box_size=args.box_size,
        targets_csv=args.targets_csv,
        dummy_target=args.dummy_target,
        device=args.device,
        allow_partial=args.allow_partial,
        graph_storage=args.graph_storage,
        overwrite=args.overwrite,
        force_unsafe_dense=args.force_unsafe_dense,
        source_manifest_policy=args.source_manifest_policy,
    )


if __name__ == "__main__":
    main()
