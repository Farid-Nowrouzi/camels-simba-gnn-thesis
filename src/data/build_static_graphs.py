from __future__ import annotations

"""
build_static_graphs.py

Build static CAMELS-SIMBA graph datasets from raw halo catalog snapshots.

Purpose
-------
This script creates one static graph per universe using ONE selected snapshot,
usually the final snapshot at scale factor a = 1.00000.

This is used for the static baseline comparison against temporal models.

Static sample format:

dataset = {
    "LH_0": {
        "A": tensor [num_nodes, num_nodes],
        "X": tensor [num_nodes, 7],
        "mask": tensor [num_nodes, 1],
        "target": tensor scalar,
        "snapshot": {...metadata...},
        ...
    },
    "LH_1": {...},
    ...
}

Official preprocessing:
    v2_logmass_minmax_top100_periodic_knn

Node selection:
    top num_nodes halos by raw Mvir

Node features:
    [log10_Mvir, X, Y, Z, VX, VY, VZ]

Graph construction:
    kNN from raw physical X/Y/Z positions with periodic boundary-aware distances

Example command:

python -m src.data.build_static_graphs \
  --raw_dir data/raw/CAMELS_SIMBA_100U \
  --output_path data/processed/static_2u_logmass_minmax_top100_periodic_knn/camels_2u_static_logmass_minmax_top100_periodic_knn.pt \
  --num_universes 2 \
  --num_nodes 100 \
  --normalization minmax \
  --graph_mode knn \
  --k 8 \
  --periodic_boundary \
  --box_size 25.0 \
  --targets_csv outputs/target_inspection_20u.csv \
  --device cpu
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

from src.data.camels_graph_utils import (
    DEFAULT_BOX_SIZE,
    FEATURE_COLUMNS,
    FEATURE_NAMES,
    MASS_COLUMN,
    POSITION_COLUMNS,
    PREPROCESSING_VERSION,
    VELOCITY_COLUMNS,
    find_universe_files,
    parse_snapshot_value,
    process_snapshot,
)


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

    return targets


def save_json(path: str | Path, data: Dict[str, object]) -> None:
    """
    Save metadata/configuration as JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def choose_static_snapshot_file(
    files: List[Path],
    preferred_snapshot: float = 1.0,
) -> Path:
    """
    Choose one snapshot file for the static graph.

    Default:
        Prefer scale factor a = 1.00000.

    If exact preferred snapshot is not found, choose the latest available
    snapshot by scale factor.

    This keeps static baselines comparable and interpretable.
    """
    if not files:
        raise ValueError("No snapshot files provided.")

    snapshot_pairs = []
    for path in files:
        value = parse_snapshot_value(path)
        snapshot_pairs.append((path, value))

    # First try exact match to preferred snapshot.
    for path, value in snapshot_pairs:
        if value is not None and abs(float(value) - float(preferred_snapshot)) < 1e-6:
            return path

    # Otherwise choose latest available snapshot.
    valid_pairs = [
        (path, value)
        for path, value in snapshot_pairs
        if value is not None
    ]

    if not valid_pairs:
        raise ValueError("Could not parse snapshot values from files.")

    latest_path, _ = max(valid_pairs, key=lambda item: item[1])

    return latest_path


def build_static_dataset(
    raw_dir: str | Path,
    output_path: str | Path,
    num_universes: int,
    num_nodes: int = 100,
    normalization: str = "minmax",
    graph_mode: str = "knn",
    k: int = 8,
    radius: Optional[float] = None,
    preferred_snapshot: float = 1.0,
    periodic_boundary: bool = True,
    box_size: float = DEFAULT_BOX_SIZE,
    targets_csv: Optional[str | Path] = None,
    dummy_target: Optional[float] = None,
    device: str = "cpu",
) -> Dict[str, object]:
    """
    Build static graph dataset from raw CAMELS-SIMBA halo catalogs.

    One universe becomes one final-snapshot graph.

    Returns:
        dataset dictionary and saves it to output_path.
    """
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    if periodic_boundary and (box_size is None or box_size <= 0):
        raise ValueError("box_size must be positive when periodic_boundary=True.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    dataset: Dict[str, object] = {}
    failed_universes = []

    print("=" * 90)
    print("CAMELS-SIMBA STATIC GRAPH BUILDER")
    print("=" * 90)
    print(f"Preprocessing version: {PREPROCESSING_VERSION}")
    print(f"Raw directory:         {raw_dir}")
    print(f"Output path:           {output_path}")
    print(f"Number universes:      {num_universes}")
    print(f"Preferred snapshot:    {preferred_snapshot}")
    print(f"Number nodes:          {num_nodes}")
    print(f"Normalization:         {normalization}")
    print(f"Graph mode:            {graph_mode}")
    print(f"k:                     {k}")
    print(f"Radius:                {radius}")
    print(f"Periodic boundary:     {periodic_boundary}")
    print(f"Box size:              {box_size}")
    print(f"Device:                {device}")
    print(f"Target mode:           {target_mode}")
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

    if dummy_target is not None:
        print()
        print("WARNING:")
        print("  You are using --dummy_target.")
        print("  This is acceptable only for testing the pipeline structure.")
        print("  Do not use dummy targets for scientific results.")
        print()

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
            all_files = find_universe_files(
                raw_dir=raw_dir,
                universe_id=universe_id,
            )

            snapshot_path = choose_static_snapshot_file(
                files=all_files,
                preferred_snapshot=preferred_snapshot,
            )

            snapshot = process_snapshot(
                path=snapshot_path,
                num_nodes=num_nodes,
                normalization=normalization,
                graph_mode=graph_mode,
                k=k,
                radius=radius,
                periodic_boundary=periodic_boundary,
                box_size=box_size,
                device=device,
            )

            target_tensor = torch.tensor(
                float(target),
                dtype=torch.float32,
                device=torch.device(device),
            )

            # --------------------------------------------------------
            # Safety checks for official preprocessing version.
            # --------------------------------------------------------
            if snapshot.get("preprocessing_version") != PREPROCESSING_VERSION:
                raise ValueError(
                    f"Unexpected preprocessing version for {universe_key}: "
                    f"{snapshot.get('preprocessing_version')}. "
                    f"Expected: {PREPROCESSING_VERSION}"
                )

            if snapshot.get("feature_names") != FEATURE_NAMES:
                raise ValueError(
                    f"Unexpected feature names for {universe_key}: "
                    f"{snapshot.get('feature_names')}. "
                    f"Expected: {FEATURE_NAMES}"
                )

            if snapshot.get("mass_feature") != "log10_Mvir":
                raise ValueError(
                    f"Unexpected mass feature for {universe_key}: "
                    f"{snapshot.get('mass_feature')}. Expected: log10_Mvir"
                )

            if snapshot.get("node_selection") != "top_num_nodes_by_raw_Mvir_descending":
                raise ValueError(
                    f"Unexpected node selection for {universe_key}: "
                    f"{snapshot.get('node_selection')}"
                )

            if (
                snapshot.get("graph_positions")
                != "raw_physical_XYZ_before_feature_normalization"
            ):
                raise ValueError(
                    f"Unexpected graph position rule for {universe_key}: "
                    f"{snapshot.get('graph_positions')}"
                )

            if snapshot.get("periodic_boundary") != periodic_boundary:
                raise ValueError(
                    f"Unexpected periodic_boundary for {universe_key}: "
                    f"{snapshot.get('periodic_boundary')}. "
                    f"Expected: {periodic_boundary}"
                )

            if graph_mode.lower() == "knn":
                expected_periodic_knn = bool(periodic_boundary)
                if snapshot.get("periodic_boundary_knn") != expected_periodic_knn:
                    raise ValueError(
                        f"Unexpected periodic_boundary_knn for {universe_key}: "
                        f"{snapshot.get('periodic_boundary_knn')}. "
                        f"Expected: {expected_periodic_knn}"
                    )

            if float(snapshot.get("box_size")) != float(box_size):
                raise ValueError(
                    f"Unexpected box_size for {universe_key}: "
                    f"{snapshot.get('box_size')}. Expected: {box_size}"
                )

            A = snapshot["A"]
            X = snapshot["X"]
            mask = snapshot["mask"]

            dataset[universe_key] = {
                "A": A,
                "X": X,
                "mask": mask,
                "target": target_tensor,
                "snapshot": {
                    "path": snapshot["path"],
                    "snapshot_value": snapshot["snapshot_value"],
                    "preprocessing_version": snapshot["preprocessing_version"],
                    "feature_names": snapshot["feature_names"],
                    "mass_feature": snapshot["mass_feature"],
                    "node_selection": snapshot["node_selection"],
                    "normalization": snapshot["normalization"],
                    "graph_mode": snapshot["graph_mode"],
                    "graph_positions": snapshot["graph_positions"],
                    "k": snapshot["k"],
                    "radius": snapshot["radius"],
                    "original_num_halos": snapshot["original_num_halos"],
                    "valid_num_halos": snapshot["valid_num_halos"],
                    "selected_num_halos_before_padding": snapshot[
                        "selected_num_halos_before_padding"
                    ],
                    "periodic_boundary": snapshot["periodic_boundary"],
                    "periodic_boundary_knn": snapshot["periodic_boundary_knn"],
                    "box_size": snapshot["box_size"],
                },
                "feature_columns": FEATURE_COLUMNS,
                "feature_names": FEATURE_NAMES,
                "position_columns": POSITION_COLUMNS,
                "velocity_columns": VELOCITY_COLUMNS,
                "mass_column": MASS_COLUMN,
                "mass_feature": "log10_Mvir",
                "node_selection": "top_num_nodes_by_raw_Mvir_descending",
                "normalization": normalization,
                "graph_mode": graph_mode,
                "graph_positions": "raw_physical_XYZ_before_feature_normalization",
                "num_nodes": num_nodes,
                "preferred_snapshot": preferred_snapshot,
                "actual_snapshot_value": snapshot["snapshot_value"],
                "preprocessing_version": PREPROCESSING_VERSION,
                "periodic_boundary": periodic_boundary,
                "periodic_boundary_knn": bool(periodic_boundary and graph_mode.lower() == "knn"),
                "box_size": box_size,
            }

            print(
                f"[OK] {universe_key} | "
                f"snapshot={snapshot['snapshot_value']} | "
                f"A={tuple(A.shape)} | "
                f"X={tuple(X.shape)} | "
                f"mask={tuple(mask.shape)} | "
                f"target={float(target_tensor):.6f} | "
                f"version={PREPROCESSING_VERSION} | "
                f"periodic={periodic_boundary} | "
                f"box={box_size}"
            )

        except Exception as exc:
            failed_universes.append((universe_key, str(exc)))
            print(f"[FAILED] {universe_key}: {exc}")

    if len(dataset) == 0:
        raise RuntimeError(
            "No universes were successfully processed. "
            "Check raw_dir, file names, column mapping, and targets."
        )

    torch.save(dataset, output_path)

    metadata = {
        "dataset_type": "static_final_snapshot_graphs",
        "preprocessing_version": PREPROCESSING_VERSION,
        "raw_dir": str(raw_dir),
        "output_path": str(output_path),
        "num_universes_requested": num_universes,
        "num_universes_successful": len(dataset),
        "num_universes_failed": len(failed_universes),
        "num_nodes": num_nodes,
        "preferred_snapshot": preferred_snapshot,
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
        "device": device,
        "feature_names": FEATURE_NAMES,
        "feature_columns": FEATURE_COLUMNS,
        "mass_column": MASS_COLUMN,
        "mass_feature": "log10_Mvir",
        "node_selection": "top_num_nodes_by_raw_Mvir_descending",
        "position_columns": POSITION_COLUMNS,
        "velocity_columns": VELOCITY_COLUMNS,
        "graph_positions": "raw_physical_XYZ_before_feature_normalization",
        "failed_universes": failed_universes,
    }

    metadata_path = output_path.with_suffix(".metadata.json")
    save_json(metadata_path, metadata)

    print()
    print("=" * 90)
    print("STATIC DATASET BUILD COMPLETE")
    print("=" * 90)
    print(f"Saved dataset:        {output_path}")
    print(f"Saved metadata:       {metadata_path}")
    print(f"Successful universes: {len(dataset)}")
    print(f"Failed universes:     {len(failed_universes)}")
    print(f"Preprocessing:        {PREPROCESSING_VERSION}")
    print(f"Feature names:        {FEATURE_NAMES}")
    print("Mass feature:         log10_Mvir")
    print("Node selection:       top_num_nodes_by_raw_Mvir_descending")
    print("Graph positions:      raw_physical_XYZ_before_feature_normalization")
    print(f"Preferred snapshot:   {preferred_snapshot}")
    print(f"Periodic boundary:    {periodic_boundary}")
    print(f"Periodic kNN:         {bool(periodic_boundary and graph_mode.lower() == 'knn')}")
    print(f"Box size:             {box_size}")

    if failed_universes:
        print()
        print("Failed universe details:")
        for universe_key, error_message in failed_universes:
            print(f"  - {universe_key}: {error_message}")

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CAMELS-SIMBA static final-snapshot graph dataset."
    )

    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--num_universes", type=int, required=True)
    parser.add_argument("--num_nodes", type=int, default=100)

    parser.add_argument(
        "--preferred_snapshot",
        type=float,
        default=1.0,
        help="Preferred static snapshot scale factor. Default: 1.0.",
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
    )

    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--radius", type=float, default=None)

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
        help="CAMELS box size. Default: 25.0 h^-1 Mpc.",
    )

    parser.add_argument("--targets_csv", type=str, default=None)
    parser.add_argument("--dummy_target", type=float, default=None)

    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    build_static_dataset(
        raw_dir=args.raw_dir,
        output_path=args.output_path,
        num_universes=args.num_universes,
        num_nodes=args.num_nodes,
        preferred_snapshot=args.preferred_snapshot,
        normalization=args.normalization,
        graph_mode=args.graph_mode,
        k=args.k,
        radius=args.radius,
        periodic_boundary=args.periodic_boundary,
        box_size=args.box_size,
        targets_csv=args.targets_csv,
        dummy_target=args.dummy_target,
        device=args.device,
    )


if __name__ == "__main__":
    main()